import argparse

import numpy as np
from mpi4py import MPI

from btx.interfaces.psana_interface import PsanaInterface
from btx.processing.ipca import IPCA

from matplotlib import pyplot as plt


class FeatureExtractor:

    """
    Class to manage feature extraction on image data subject to initialization
    parameters.
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        num_components=10,
        block_size=10,
        num_images=10,
        init_with_pca=False,
        benchmark_mode=False,
        downsample=False,
        output_dir="",
    ):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)

        self.reduced_indices = np.array([])

        self.init_with_pca = init_with_pca
        self.benchmark_mode = benchmark_mode
        self.output_dir = output_dir
        self.downsample = downsample

        det_shape = self.psi.det.shape()
        self.d = np.prod(det_shape)

        if self.downsample:
            # need a better way to do this, rounded closest cf?
            self.bin_factor = 16
            self.d = int(self.d / self.bin_factor**2)

        # ensure that requested number of images is valid
        self.num_images = num_images
        if self.num_images > self.psi.max_events:
            self.num_images = self.psi.max_events
            print(
                f"Requested number of images too large,\
                    reduced to {self.num_images}"
            )

        if self.benchmark_mode:
            self.num_images = min(120, self.psi.max_events)
            self.q = num_components
            self.m = 20
        else:
            # ensure that requested dimension is valid
            self.q = num_components
            if self.q > self.num_images:
                self.q = self.num_images
                print(
                    f"Requested number of components too large,\
                        reduced to {self.q}"
                )

            # ensure block size is valid
            self.m = block_size
            if self.m > self.num_images:
                self.m = self.num_images
                print(f"Requested block size too large, reduced to {self.m}")

    def distribute_indices(self):
        d = self.d
        size = self.size

        # determine boundary indices between ranks
        split_indices = np.zeros(size)
        for r in range(size):
            num_per_rank = d // size
            if r < (d % size):
                num_per_rank += 1
            split_indices[r] = num_per_rank

        trailing_indices = np.cumsum(split_indices)
        split_indices = np.append(np.array([0]), trailing_indices).astype(int)

        self.split_indices = split_indices

    def fetch_formatted_images(self, n, fetch_all_features=False):
        """
        Retrieve and format n images from run.

        Paramters
        ---------
        n : int
            number of images to retrieve and format

        Return
        ------
        formatted_imgs: ndarray, shape (d x n)
            n formatted (d x 1) images from run

        Notes
        -----
        The PsanaInterface instance self.psi has an internal counter which is
        updated on calls to get_images, ensuring that images are retrieved
        sequentially using this method.
        """
        d = self.d
        rank = self.rank

        # get start index, end index
        start_index = 0 if fetch_all_features else self.split_indices[rank]
        end_index = d if fetch_all_features else self.split_indices[rank + 1]

        # may have to rewrite eventually when number of images becomes large,
        # i.e. streamed setting, either that or downsample aggressively
        imgs = self.psi.get_images(n, assemble=False)

        if self.downsample:
            imgs = bin_data(imgs, self.bin_factor)

        formatted_imgs = np.reshape(imgs, (n, d)).T

        return formatted_imgs[start_index:end_index, :]

    def run_ipca(self):
        """
        Perform iPCA on run subject to initialization parameters.
        """
        d = self.d
        m = self.m
        q = self.q

        self.distribute_indices()
        split_indices = self.split_indices

        self.ipca = IPCA(d, q, m, split_indices)

        parsed_images = 0
        num_images = self.num_images

        if self.init_with_pca and not self.benchmark_mode:
            img_block = self.fetch_formatted_images(q)
            self.ipca.initialize_model(img_block)

            parsed_images = q

        # divide remaning number of images into blocks
        rem_imgs = num_images - parsed_images
        block_sizes = np.array(
            [m] * np.floor(rem_imgs / m).astype(int)
            + ([rem_imgs % m] if rem_imgs % m else [])
        )

        # update model with remaining blocks
        for block_size in block_sizes:
            img_block = self.fetch_formatted_images(block_size)
            self.ipca.update_model(img_block)

        if self.benchmark_mode:
            self.ipca.save_interval_data(self.output_dir)

    def verify_model_accuracy(self):
        d = self.d
        m = self.m
        q = self.q
        n = self.num_images

        self.comm.Barrier()
        U, S, mu, var = self.ipca.get_model()

        if self.rank == 0:
            # store current event index from self.psi and reset
            event_index = self.psi.counter
            self.psi.counter = 0

            try:
                print("Verifying Model Accuracy\n------------------------\n")
                print(f"q = {q}")
                print(f"d = {d}")
                print(f"n = {n}")
                print(f"m = {m}")
                print("\n")

                # run svd on centered image batch
                print("\nGathering images for batch PCA...")
                X = self.fetch_formatted_images(n, fetch_all_features=True)

                print("Performing batch PCA...")
                mu_pca = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
                var_pca = np.reshape(np.var(X, ddof=1, axis=1), (X.shape[0], 1))

                mu_n = np.tile(mu_pca, n)
                X_centered = X - mu_n

                U_pca, S_pca, _ = np.linalg.svd(X_centered, full_matrices=False)
                print("\n")

                q_pca = min(q, n)

                # calculate compression loss, normalized if given flag
                norm = True
                print(
                    "iPCA {norm}Compression Loss: {loss:.6g}".format(
                        norm="Normalized " if norm else "",
                        loss=compression_loss(X, U[:, :q_pca], normalized=norm),
                    )
                )
                print(
                    "PCA {norm}Compression Loss: {loss:.6g}".format(
                        norm="Normalized " if norm else "",
                        loss=compression_loss(X, U_pca[:, :q_pca], normalized=norm),
                    )
                )
                print("\n")

                print(f"iPCA Total Variance: {np.sum(var):.6g}")
                print(f"PCA Total Variance: {np.sum(var_pca):.6g}")
                print("\n")

                ipca_exp_var = (np.sum(S[:q_pca] ** 2) / (n - 1)) / np.sum(var)
                print(f"iPCA Explained Variance: {ipca_exp_var:.6g}")
                pca_exp_var = (np.sum(S_pca[:q_pca] ** 2) / (n - 1)) / np.sum(var_pca)
                print(f"PCA Explained Variance: {pca_exp_var:.6g}")
                print("\n")

                print("iPCA Singular Values: \n")
                print(S)
                print("\n")

                print("PCA Singular Values: \n")
                print(S_pca[:q_pca])
                print("\n")

                mean_inner_prod = np.inner(mu.flatten(), mu_pca.flatten()) / (
                    np.linalg.norm(mu) * np.linalg.norm(mu_pca)
                )

                print(f"Normalized Mean Inner Product: {mean_inner_prod:6g}")
                print("\n")

                print("Basis Inner Product: \n")
                print(np.diagonal(np.abs(U[:, :q_pca].T @ U_pca[:, :q_pca])))

                ipca_mu_u = np.hstack((mu / np.linalg.norm(mu), U[:, :q_pca]))
                pca_mu_u = np.hstack(
                    (mu_pca / np.linalg.norm(mu_pca), U_pca[:, :q_pca])
                )

                b = plt.imshow(np.abs(ipca_mu_u.T @ pca_mu_u))
                plt.colorbar(b)
                # plt.savefig(f"fig_{q}_{self.size}.png")
                plt.clf()
                plt.show()

                self.ipca.report_interval_data()

            finally:
                # reset counter
                self.psi.counter = event_index


def compare_basis_vectors(U_1, U_2, q):
    """
    Quantitatively compare the first q basis vectors of U and U_prime.

    Parameters
    ----------
    U_1 : ndarray, shape (d x a), a >= q
        first matrix of orthonormal basis vectors
    U_2 : ndarray, shape (d x b), b >= q
        second matrix of orthonormal basis vectors
    q : int
        number of vectors to compare

    Returns
    -------
    acc : float, 0 <= acc <= 1
        quantitative measure of distance between basis vectors
    """
    if q > min(U_1.shape[1], U_2.shape[1]):
        print("Desired number of vectors is greater than matrix dimension.")
        return 0.0

    acc = np.trace(np.abs(U_1[:, :q].T @ U_2[:, :q])) / q
    return acc


def compression_loss(X, U, normalized=False):
    """
    Calculate the compression loss between centered observation matrix X
    and its rank-q reconstruction.

    Parameters
    ----------
    X : ndarray, shape (d x n)
        flattened, centered image data from n run events
    U : ndarray, shape (d x q)
        first q singular vectors of X, forming orthonormal basis
        of q-dimensional subspace of R^d

    Returns
    -------
    Ln : float
        compression loss of X
    """
    _, n = X.shape

    UX = U.T @ X
    UUX = U @ UX

    Ln = ((np.linalg.norm(X - UUX, "fro")) ** 2) / n

    if normalized:
        Ln /= np.linalg.norm(X, "fro") ** 2

    return Ln


def bin_data(arr, bin_factor, det_shape=None):
    """
    Bin detector data by bin_factor through averaging.
    Retrieved from
    https://github.com/apeck12/cmtip/blob/main/cmtip/prep_data.py

    :param arr: array shape (n_images, n_panels, panel_shape_x, panel_shape_y)
      or if det_shape is given of shape (n_images, 1, n_pixels_per_image)
    :param bin_factor: how may fold to bin arr by
    :param det_shape: tuple of detector shape, optional
    :return arr_binned: binned data of same dimensions as arr
    """
    # reshape as needed
    if det_shape is not None:
        arr = np.array([arr[i].reshape(det_shape) for i in range(arr.shape[0])])

    n, p, y, x = arr.shape

    # ensure that original shape is divisible by bin factor
    assert y % bin_factor == 0
    assert x % bin_factor == 0

    # bin each panel of each image
    binned_arr = (
        arr.reshape(
            n,
            p,
            int(y / bin_factor),
            bin_factor,
            int(x / bin_factor),
            bin_factor,
        )
        .mean(-1)
        .mean(3)
    )

    # if input data were flattened, reflatten
    if det_shape is not None:
        flattened_size = np.prod(np.array(binned_arr.shape[1:]))
        binned_arr = binned_arr.reshape((binned_arr.shape[0], 1) + (flattened_size,))
    return binned_arr


# For command line use #


def parse_input():
    """
    Parse command line input.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", help="Experiment name", required=True, type=str)
    parser.add_argument("-r", "--run", help="Run number", required=True, type=int)
    parser.add_argument(
        "-d",
        "--det_type",
        help="Detector name, e.g epix10k2M or jungfrau4M",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-c",
        "--num_components",
        help="Number of principal components to compute",
        required=False,
        type=int,
    )
    parser.add_argument(
        "-m",
        "--block_size",
        help="Desired block size",
        required=False,
        type=int,
    )
    parser.add_argument(
        "-n", "--num_images", help="Number of images", required=False, type=int
    )

    parser.add_argument(
        "--output_dir",
        help="Path to output directory.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--init_with_pca",
        help="Initialize with PCA",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--benchmark_mode",
        help="Run algorithm in benchmark mode.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--downsample", help="Downsample.", required=False, action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":

    params = parse_input()
    kwargs = {k: v for k, v in vars(params).items() if v is not None}

    fe = FeatureExtractor(**kwargs)
    fe.run_ipca()
    fe.verify_model_accuracy()

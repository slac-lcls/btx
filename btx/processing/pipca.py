import os, csv, argparse

import numpy as np
from mpi4py import MPI

from matplotlib import pyplot as plt
from matplotlib import colors

from btx.misc.shortcuts import TaskTimer

from btx.interfaces.psana_interface import (
    PsanaInterface,
    bin_data,
    bin_pixel_index_map,
    retrieve_pixel_index_map,
    assemble_image_stack_batch,
)

class PiPCA:

    """Parallelized Incremental Principal Component Analysis."""

    def __init__(
        self,
        exp,
        run,
        det_type,
        start_offset=0,
        num_images=10,
        num_components=10,
        batch_size=10,
        priming=False,
        benchmarking=False,
        downsample=False,
        bin_factor=2,
        output_dir="",
    ):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
        self.psi.counter = start_offset

        self.priming = priming
        self.benchmarking = benchmarking
        self.downsample = downsample
        self.bin_factor = bin_factor
        self.output_dir = output_dir

        (
            self.num_images,
            self.num_components,
            self.batch_size,
            self.num_features,
        ) = self.set_params(num_images, num_components, batch_size, bin_factor)

        # compute number of counts in and start indices over ranks
        self.split_indices, self.split_counts = distribute_indices(
            self.num_features, self.size
        )

        # attribute for storing interval data
        self.task_durations = dict({})

        self.num_incorporated_images = 0
        self.outliers, self.pc_data = [], []

    def get_params(self):
        """
        Method to retrieve iPCA params.

        Returns
        -------
        num_incorporated_images : int
            number of images used to build model
        num_components : int
            number of components maintained in model
        batch_size : int
            batch size used in model updates
        num_features : int
            dimensionality of incorporated images
        """
        return (
            self.num_incorporated_images,
            self.num_components,
            self.batch_size,
            self.num_features,
        )

    def set_params(self, num_images, num_components, batch_size, bin_factor):
        """
        Method to initialize iPCA parameters.

        Parameters
        ----------
        num_images : int
            Desired number of images to incorporate into model.
        num_components : int
            Desired number of components for model to maintain.
        batch_size : int
            Desired size of image block to be incorporated into model at each update.
        bin_factor : int
            Factor to bin data by.

        Returns
        -------
        n : int
            Number of images to incorporate into model.
        q : int
            Number of components for model to maintain.
        m : int
            Size of image block to be incorporated into model at each update.
        d : int
            Number of features (dimension) in each image.
        """
        max_events = self.psi.max_events
        benchmarking = self.benchmarking
        downsample = self.downsample

        # set n, q, and m
        n = num_images
        q = num_components
        m = batch_size

        if benchmarking:
            min_n = max(int(4 * q), 40)
            n = min(min_n, max_events)
            m = q

            print(f"In benchmarking mode, setting q = {q}, n = {n}, m = {m}.")
        else:
            n = min(n, max_events) if n != -1 else max_events
            q = min(q, n)
            m = min(m, n)

        # set d
        det_shape = self.psi.det.shape()
        d = np.prod(det_shape).astype(int)

        if downsample:
            if det_shape[-1] % bin_factor or det_shape[-2] % bin_factor:
                print("Invalid bin factor, toggled off downsampling.")
                self.downsample = False
            else:
                d = int(d / bin_factor**2)

        return n, q, m, d

    def run(self):
        """
        Perform iPCA on run subject to initialization parameters.
        """
        m = self.batch_size
        num_images = self.num_images

        # initialize and prime model, if specified
        if self.priming and not self.benchmarking:
            img_batch = self.get_formatted_images(
                self.num_components, 0, self.num_features
            )
            self.prime_model(img_batch)
        else:
            self.U = np.zeros((self.split_counts[self.rank], self.num_components))
            self.S = np.ones(self.num_components)
            self.mu = np.zeros((self.split_counts[self.rank], 1))
            self.total_variance = np.zeros((self.split_counts[self.rank], 1))

        # divide remaining number of images into batches
        # will become redundant in a streaming setting, need to change
        rem_imgs = num_images - self.num_incorporated_images
        batch_sizes = np.array(
            [m] * np.floor(rem_imgs / m).astype(int)
            + ([rem_imgs % m] if rem_imgs % m else [])
        )

        # update model with remaining batches
        for batch_size in batch_sizes:
            self.fetch_and_update_model(batch_size)

        if self.benchmarking:
            self.save_interval_data()

    def get_formatted_images(self, n, start_index, end_index):
        """
        Fetch n - x image segments from run, where x is the number of 'dead' images.

        Parameters
        ----------
        n : int
            number of images to retrieve
        start_index : int
            start index of subsection of data to retrieve
        end_index : int
            end index of subsection of data to retrieve

        Returns
        -------
        ndarray, shape (end_index-start_index, n-x)
            n-x retrieved image segments of dimension end_index-start_index
        """

        bin_factor = self.bin_factor
        downsample = self.downsample

        # may have to rewrite eventually when number of images becomes large,
        # i.e. streamed setting, either that or downsample aggressively
        imgs = self.psi.get_images(n, assemble=False)

        if downsample:
            imgs = bin_data(imgs, bin_factor)

        imgs = imgs[
            [i for i in range(imgs.shape[0]) if not np.isnan(imgs[i : i + 1]).any()]
        ]

        num_valid_imgs, p, x, y = imgs.shape
        formatted_imgs = np.reshape(imgs, (num_valid_imgs, p * x * y)).T

        return formatted_imgs[start_index:end_index, :]

    def prime_model(self, X):
        """
        Initialiize model on sample of data using batch PCA.

        Parameters
        ----------
        X : ndarray, shape (d x m)
            set of m (d x 1) observations
        """

        if self.rank == 0:
            print(f"Priming model with {self.num_components} samples...")

        self.mu, self.total_variance = self.calculate_sample_mean_and_variance(X)

        centered_data = X - np.tile(self.mu, self.num_components)
        self.U, self.S, _ = np.linalg.svd(centered_data, full_matrices=False)

        self.num_incorporated_images += self.num_components

    def fetch_and_update_model(self, n):
        """
        Fetch images and update model.

        Parameters
        ----------
        n : int
            number of images to incorporate
        """

        rank = self.rank
        start_index, end_index = self.split_indices[rank], self.split_indices[rank + 1]

        img_batch = self.get_formatted_images(n, start_index, end_index)

        self.update_model(img_batch)

    def update_model(self, X):
        """
        Update model with new batch of observations using iPCA.

        Parameters
        ----------
        X : ndarray, shape (d x m)
            batch of m (d x 1) observations

        Notes
        -----
        Implementation of iPCA algorithm from [1].

        References
        ----------
        [1] Ross DA, Lim J, Lin RS, Yang MH. Incremental learning for robust visual tracking.
        International journal of computer vision. 2008 May;77(1):125-41.
        """
        _, m = X.shape
        n = self.num_incorporated_images
        q = self.num_components

        with TaskTimer(self.task_durations, "total update"):

            if self.rank == 0:
                print(
                    "Factoring {m} sample{s} into {n} sample, {q} component model...".format(
                        m=m, s="s" if m > 1 else "", n=n, q=q
                    )
                )

            with TaskTimer(self.task_durations, "record pc data"):
                if n > 0:
                    self.record_loadings(X)

            with TaskTimer(self.task_durations, "update mean and variance"):
                mu_n = self.mu
                mu_m, s_m = self.calculate_sample_mean_and_variance(X)

                self.total_variance = self.update_sample_variance(
                    self.total_variance, s_m, mu_n, mu_m, n, m
                )
                self.mu = self.update_sample_mean(mu_n, mu_m, n, m)

            with TaskTimer(
                self.task_durations, "center data and compute augment vector"
            ):
                X_centered = X - np.tile(mu_m, m)
                mean_augment_vector = np.sqrt(n * m / (n + m)) * (mu_m - mu_n)

                X_augmented = np.hstack((X_centered, mean_augment_vector))

            with TaskTimer(self.task_durations, "first matrix product U@S"):
                US = self.U @ np.diag(self.S)

            with TaskTimer(self.task_durations, "QR concatenate"):
                A = np.hstack((US, X_augmented))

            with TaskTimer(self.task_durations, "parallel QR"):
                Q_r, U_tilde, S_tilde = self.parallel_qr(A)

            # concatenating first preserves the memory contiguity
            # of U_prime and thus self.U
            with TaskTimer(self.task_durations, "compute local U_prime"):
                self.U = Q_r @ U_tilde[:, :q]
                self.S = S_tilde[:q]

            self.num_incorporated_images += m


    def calculate_sample_mean_and_variance(self, imgs):
        """
        Compute the sample mean and variance of a flattened stack of n images.

        Parameters
        ----------
        imgs : ndarray, shape (d x n)
            horizonally stacked batch of flattened images

        Returns
        -------
        mu_m : ndarray, shape (d x 1)
            mean of imgs
        su_m : ndarray, shape (d x 1)
            sample variance of imgs (1 dof)
        """
        d, m = imgs.shape

        mu_m = np.reshape(np.mean(imgs, axis=1), (d, 1))
        s_m = np.zeros((d, 1))

        if m > 1:
            s_m = np.reshape(np.var(imgs, axis=1, ddof=1), (d, 1))

        return mu_m, s_m

    def parallel_qr(self, A):
        """
        Perform parallelized qr factorization on input matrix A.

        Parameters
        ----------
        A : ndarray, shape (_ x q+m+1)
            Input data to be factorized.

        Returns
        -------
        q_fin : ndarray, shape (_, q+m+1)
            Q_{r,1} from TSQR algorithm, where r = self.rank + 1
        U_tilde : ndarray, shape (q+m+1, q+m+1)
            Q_{r,2} from TSQR algorithm, where r = self.rank + 1
        S_tilde : ndarray, shape (q+m+1)
            R_tilde from TSQR algorithm, where r = self.rank + 1

        Notes
        -----
        Parallel QR algorithm implemented from [1], with additional elements from [2]
        sprinkled in to record elements for iPCA using SVD, etc.

        References
        ----------
        [1] Benson AR, Gleich DF, Demmel J. Direct QR factorizations for tall-and-skinny
        matrices in MapReduce architectures. In2013 IEEE international conference on
        big data 2013 Oct 6 (pp. 264-272). IEEE.

        [2] Ross DA, Lim J, Lin RS, Yang MH. Incremental learning for robust visual tracking.
        International journal of computer vision. 2008 May;77(1):125-41.

        [3] Maulik, R., & Mengaldo, G. (2021, November). PyParSVD: A streaming, distributed and
        randomized singular-value-decomposition library. In 2021 7th International Workshop on
        Data Analysis and Reduction for Big Scientific Data (DRBSD-7) (pp. 19-25). IEEE.
        """
        _, x = A.shape
        q = self.num_components
        m = x - q - 1

        with TaskTimer(self.task_durations, "qr - local qr"):
            Q_r1, R_r = np.linalg.qr(A, mode="reduced")

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - r_tot gather"):
            if self.rank == 0:
                R = np.empty((self.size * (q + m + 1), q + m + 1))
            else:
                R = None

            self.comm.Gather(R_r, R, root=0)

        if self.rank == 0:
            with TaskTimer(self.task_durations, "qr - global qr"):
                Q_2, R_tilde = np.linalg.qr(R, mode="reduced")

            # compute SVD of R_tilde, from iPCA algorithm
            with TaskTimer(self.task_durations, "qr - global svd"):
                U_tilde, S_tilde, _ = np.linalg.svd(R_tilde)
        else:
            U_tilde = np.empty((q + m + 1, q + m + 1))
            S_tilde = np.empty(q + m + 1)
            Q_2 = None

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - scatter q_tot"):
            Q_r2 = np.empty((q + m + 1, q + m + 1))
            self.comm.Scatter(Q_2, Q_r2, root=0)

        with TaskTimer(self.task_durations, "qr - local matrix build"):
            Q_r = Q_r1 @ Q_r2

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - bcast S_tilde"):
            self.comm.Bcast(S_tilde, root=0)

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - bcast U_tilde"):
            self.comm.Bcast(U_tilde, root=0)

        return Q_r, U_tilde, S_tilde

    def update_sample_mean(self, mu_n, mu_m, n, m):
        """
        Compute combined mean of two blocks of data.

        Parameters
        ----------
        mu_n : ndarray, shape (d x 1)
            mean of first block of data
        mu_m : ndarray, shape (d x 1)
            mean of second block of data
        n : int
            number of observations in first block of data
        m : int
            number of observations in second block of data

        Returns
        -------
        mu_nm : ndarray, shape (d x 1)
            combined mean of both blocks of input data
        """
        mu_nm = mu_m

        if n != 0:
            mu_nm = (1 / (n + m)) * (n * mu_n + m * mu_m)

        return mu_nm

    def update_sample_variance(self, s_n, s_m, mu_n, mu_m, n, m):
        """
        Compute combined sample variance of two blocks
        of data described by input parameters.

        Parameters
        ----------
        s_n : ndarray, shape (d x 1)
            sample variance of first block of data
        s_m : ndarray, shape (d x 1)
            sample variance of second block of data
        mu_n : ndarray, shape (d x 1)
            mean of first block of data
        mu_m : ndarray, shape (d x 1)
            mean of second block of data
        n : int
            number of observations in first block of data
        m : int
            number of observations in second block of data

        Returns
        -------
        s_nm : ndarray, shape (d x 1)
            combined sample variance of both blocks of data described by input parameters
        """
        s_nm = s_m

        if n != 0:
            s_nm = (
                           ((n - 1) * s_n + (m - 1) * s_m) + (n * m * (mu_n - mu_m) ** 2) / (n + m)
                   ) / (n + m - 1)

        return s_nm

    def get_model(self):
        """
        Method to retrieve model parameters.

        Returns
        -------
        U_tot : ndarray, shape (d x q)
            iPCA principal axes from model.
        S_tot : ndarray, shape (1 x q)
            iPCA singular values from model.
        mu_tot : ndarray, shape (1 x d)
            Data mean computed from all input images.
        var_tot : ndarray, shape (1 x d)
            Sample data variance computed from all input images.
        """
        if self.rank == 0:
            U_tot = np.empty(self.num_features * self.num_components)
            mu_tot = np.empty((self.num_features, 1))
            var_tot = np.empty((self.num_features, 1))
        else:
            U_tot, mu_tot, var_tot = None, None, None

        start_indices = self.split_indices[:-1]

        self.comm.Gatherv(
            self.U.flatten(),
            [
                U_tot,
                self.split_counts * self.num_components,
                start_indices * self.num_components,
                MPI.DOUBLE,
            ],
            root=0,
        )

        if self.rank == 0:
            U_tot = np.reshape(U_tot, (self.num_features, self.num_components))

        self.comm.Gatherv(
            self.mu,
            [
                mu_tot,
                self.split_counts * self.num_components,
                start_indices,
                MPI.DOUBLE,
            ],
            root=0,
        )
        self.comm.Gatherv(
            self.total_variance,
            [
                var_tot,
                self.split_counts * self.num_components,
                start_indices,
                MPI.DOUBLE,
            ],
            root=0,
        )

        S_tot = self.S

        return U_tot, S_tot, mu_tot, var_tot

    def get_loss_stats(self):

        if self.rank == 0:
            print(self.outliers)

    def record_loadings(self, X):
        """
        Method to store all loadings, ΣV^T, from present batch using past
        model iteration.

        Parameters
        ----------
        X : ndarray, shape (_ x m)
            Local subdivision of current image data batch.
        """
        _, m = X.shape
        n, d = self.num_incorporated_images, self.num_features

        start_indices = self.split_indices[:-1]

        U, _, mu, _ = self.get_model()

        if self.rank == 0:
            X_tot = np.empty((d, m))
        else:
            X_tot = None

        self.comm.Gatherv(
            X.flatten(),
            [
                X_tot,
                self.split_counts * m,
                start_indices * m,
                MPI.DOUBLE,
            ],
            root=0,
        )

        if self.rank == 0:

            X_tot = np.reshape(X_tot, (d, m))
            cb = X_tot - np.tile(mu, (1, m))

            pcs = U.T @ cb
            self.pc_data = (
                np.concatenate((self.pc_data, pcs), axis=1)
                if len(self.pc_data)
                else pcs
            )

            pc_dist = np.linalg.norm(pcs[:5], axis=0)
            std = np.std(pc_dist)
            mu = np.mean(pc_dist)

            batch_outliers = np.where(np.abs(pc_dist - mu) > std)[0] + n - m

            self.outliers = (
                np.concatenate((self.outliers, batch_outliers), axis=0)
                if len(self.outliers)
                else batch_outliers
            )

    def compression_loss(self, X, U, normalized=False):
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

        return L

    def display_image(self, idx, output_dir="", save_image=False):
        """
        Method to retrieve single image from run subject to model binning constraints.

        Parameters
        ----------
        idx : int
            Run index of image to be retrieved.
        output_dir : str, optional
            File path to output directory, by default ""
        save_image : bool, optional
            Whether to save image to file, by default False
        """

        U, S, mu, var = self.get_model()

        if self.rank != 0:
            return

        bin_factor = 1
        if self.downsample:
            bin_factor = self.bin_factor

        n, q, m, d = self.get_params()

        a, b, c = self.psi.det.shape()
        b = int(b / bin_factor)
        c = int(c / bin_factor)

        fig, ax = plt.subplots(1)

        counter = self.psi.counter
        self.psi.counter = idx
        img = self.get_formatted_images(1, 0, d)
        self.psi.counter = counter

        img = img - mu
        img = np.reshape(img, (a, b, c))

        pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(self.psi.run))
        binned_pim = bin_pixel_index_map(pixel_index_map, bin_factor)

        img = assemble_image_stack_batch(img, binned_pim)

        #vmin = np.min(img.flatten())
        vmax = np.max(img.flatten())
        ax.imshow(
            img,
            norm=colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=0, vmax=vmax),
            interpolation=None
        )

        if save_image:
            plt.savefig(output_dir)

        plt.show()

    def verify_model_accuracy(self):
        """
        Run benchmarking to verify model accuracy.
        """
        self.comm.Barrier()
        U, S, mu, var = self.get_model()

        if self.rank != 0:
            return

        d = self.num_features
        m = self.batch_size
        q = self.num_components
        num_images = self.num_images

        # store current event index from self.psi and reset
        event_index = self.psi.counter
        self.psi.counter = event_index - num_images

        try:
            print("\nVerifying Model Accuracy\n------------------------\n")
            print(f"q = {q}")
            print(f"d = {d}")
            print(f"n = {num_images}")
            print(f"m = {m}")
            print("\n")

            # run svd on centered image batch
            print("Gathering images for batch PCA...")
            X = self.get_formatted_images(num_images, 0, d)
            y, x = X.shape

            print("Performing batch PCA...")
            mu_pca = np.reshape(np.mean(X, axis=1), (y, 1))
            var_pca = np.reshape(np.var(X, ddof=1, axis=1), (y, 1))

            mu_n = np.tile(mu_pca, x)
            X_centered = X - mu_n

            U_pca, S_pca, _ = np.linalg.svd(X_centered, full_matrices=False)
            print("\n")

            q_pca = min(q, x)


            # calculate compression loss, normalized if given flag
            norm = True
            norm_str = "Normalized " if norm else ""

            ipca_loss = self.compression_loss(X, U[:, :q_pca], normalized=norm)
            print(f"iPCA {norm_str}Compression Loss: {ipca_loss}")

            pca_loss = self.compression_loss(X, U_pca[:, :q_pca], normalized=norm)
            print(f"PCA {norm_str}Compression Loss: {pca_loss}")

            print("\n")
            ipca_tot_var = np.sum(var)
            pca_tot_var = np.sum(var_pca)

            print(f"iPCA Total Variance: {ipca_tot_var}")
            print(f"PCA Total Variance: {pca_tot_var}")
            print("\n")

            ipca_exp_var = (np.sum(S[:q_pca] ** 2) / (x - 1)) / np.sum(var)
            print(f"iPCA Explained Variance: {ipca_exp_var}")

            pca_exp_var = (np.sum(S_pca[:q_pca] ** 2) / (x - 1)) / np.sum(var_pca)
            print(f"PCA Explained Variance: {pca_exp_var}")
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

            print("Normalized Mean Inner Product: " + f"{mean_inner_prod}")
            print("\n")

            print("Basis Inner Product: \n")
            print(np.diagonal(np.abs(U[:, :q_pca].T @ U_pca[:, :q_pca])))

            ipca_mu_u = np.hstack((mu / np.linalg.norm(mu), U[:, :q_pca]))
            pca_mu_u = np.hstack((mu_pca / np.linalg.norm(mu_pca), U_pca[:, :q_pca]))

            b = plt.imshow(np.abs(ipca_mu_u.T @ pca_mu_u))
            plt.colorbar(b)
            plt.savefig(f"fig_{q}_{self.size}.png")
            plt.show()

            print("\n")
            self.report_interval_data()

        finally:
            # reset counter
            self.psi.counter = event_index


    def report_interval_data(self):
        """
        Method to print out iPCA time interval data.
        """

        if self.rank != 0:
            return

        task_durations = self.task_durations

        if not len(task_durations):
            print("No model updates or initialization recorded.")
            return

        # log data
        print("\n")
        for key in list(task_durations.keys()):
            interval_mean = np.mean(task_durations[key])
            print(
                "Mean per-batch compute time of step "
                + f"'{key}': {interval_mean:.4g}s"
            )

    def save_interval_data(self):
        """
        Save time interval data gathered during iPCA.

        Parameters
        ----------
        save_data : bool, optional
            if True, save interval data to file in self.output_dir, by default False
        """

        if self.rank != 0:
            return

        q = self.num_components
        d = self.num_features
        n = self.num_incorporated_images
        m = self.batch_size

        size = self.size
        dir_path = self.output_dir
        task_durations = self.task_durations

        file_name = "task_" + str(q) + str(d) + str(n) + str(size) + ".csv"

        with open(
            os.path.join(dir_path, file_name),
            "x",
            newline="",
            encoding="utf-8",
        ) as f:

            if len(task_durations):
                writer = csv.writer(f)

                writer.writerow(["q", q])
                writer.writerow(["d", d])
                writer.writerow(["n", n])
                writer.writerow(["ranks", size])
                writer.writerow(["m", m])

                keys = list(task_durations.keys())
                values = list(task_durations.values())
                values_transposed = np.array(values).T

                writer.writerow(keys)
                writer.writerows(values_transposed)


def distribute_indices(d, size):
    """_summary_

    Parameters
    ----------
    d : int
        total number of dimensions
    size : int
        number of ranks in world

    Returns
    -------
    split_indices : ndarray, shape (size+1 x 1)
        division indices between ranks
    split_counts : ndarray, shape (size x 1)
        number of dimensions allocated per rank
    """

    total_indices = 0
    split_indices, split_counts = [0], []

    for r in range(size):
        num_per_rank = d // size
        if r < (d % size):
            num_per_rank += 1

        split_counts.append(num_per_rank)

        total_indices += num_per_rank
        split_indices.append(total_indices)

    split_indices = np.array(split_indices)
    split_counts = np.array(split_counts)

    return split_indices, split_counts


#### for command line use ###


def parse_input():
    """
    Parse command line input.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", help="Experiment name.", required=True, type=str)
    parser.add_argument("-r", "--run", help="Run number.", required=True, type=int)
    parser.add_argument(
        "-d",
        "--det_type",
        help="Detector name, e.g epix10k2M or jungfrau4M.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--start_offset",
        help="Run index of first image to be incorporated into iPCA model.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_components",
        help="Number of principal components to compute and maintain.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help="Size of image batch incorporated in each model update.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_images",
        help="Total number of images to be incorporated into model.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output directory for recording interval data.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--priming",
        help="Initialize model with PCA.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--benchmarking",
        help="Run algorithm in benchmarking mode.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--downsample",
        help="Enable downsampling of images.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--bin_factor",
        help="Bin factor if using downsizing.",
        required=False,
        type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":

    params = parse_input()
    kwargs = {k: v for k, v in vars(params).items() if v is not None}

    pipca = PiPCA(**kwargs)
    pipca.run()
    # fe.verify_model_accuracy()
    pipca.get_loss_stats()

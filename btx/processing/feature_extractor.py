import argparse
from inspect import formatargspec
import numpy as np

from btx.interfaces.psana_interface import *
from btx.processing.ipca import *

from matplotlib import pyplot as plt


class FeatureExtractor:
    
    """
    Class to manage feature extraction on image data subject to initialization parameters.
    """

    def __init__(self, exp, run, det_type, num_components=50, block_size=10, num_images=100, init_with_pca=False, benchmark_mode=False, output_dir=''):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)

        self.d = np.prod(self.psi.det.shape())
        self.reduced_indices = np.array([])

        self.init_with_pca = init_with_pca
        self.benchmark_mode = benchmark_mode
        
        self.output_dir = output_dir

        # ensure that requested number of images is valid
        self.num_images = num_images
        if self.num_images > self.psi.max_events:
            self.num_images = self.psi.max_events
            print(f'Requested number of images too large, reduced to {self.num_images}')

        if self.benchmark_mode:
            self.num_images = min(100, self.num_images)
            self.q = num_components
            self.m = 20
        else:
            # ensure that requested dimension is valid
            self.q = num_components
            if self.q > self.num_images:
                self.q = self.num_images
                print(f'Requested number of components too large, reduced to {self.q}')

            # ensure block size is valid
            self.m = block_size
            if self.m > self.num_images:
                self.m = self.num_images
                print(f'Requested block size too large, reduced to {self.m}')

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

        split_indices = np.append(np.array([0]), np.cumsum(split_indices)).astype(int)

        self.split_indices = split_indices

    def fetch_formatted_images(self, n):
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
        The PsanaInterface instance self.psi has an internal counter which is updated on calls 
        to get_images, ensuring that images are retrieved sequentially using this method.
        """
        d_original = np.prod(self.psi.det.shape())

        # get ranks start index, end index, and number of features to parse
        start_index = self.split_indices[self.rank]
        end_index = self.split_indices[self.rank+1]
        num_features =  end_index - start_index

        # may have to rewrite eventually when number of images becomes large, i.e. online
        imgs = self.psi.get_images(n, assemble=False)
        rank_imgs = np.empty((num_features, n))

        for i in range(n):
            
            formatted_imgs = np.reshape(imgs[i], (d_original, 1))

            if self.reduced_indices.size:
                formatted_imgs = formatted_imgs[self.reduced_indices]

            rank_imgs[:, i:i+1] = formatted_imgs[start_index:end_index]

        return rank_imgs

    def generate_reduced_indices(self, new_dim):
        """
        Sample, without replacement, and store new_dim indices from unassembed image dimension.

        Parameters
        ----------
        new_dim : int
            number of sampled indices to be drawn
        """
        det_pixels = np.prod(self.psi.det.shape())

        if det_pixels < new_dim:
            print('Detector dimension must be greater than or equal to reduced dimension.')
            return

        self.reduced_indices = np.random.choice(det_pixels, new_dim, replace=False)
        self.d = new_dim

    def remove_reduced_indices(self):
        """
        Remove stored reduced indices.
        """
        det_pixels = np.prod(self.psi.det.shape())

        self.reduced_indices = np.array([])
        self.d = det_pixels

    def run_ipca(self):
        """
        Perform iPCA on run subject to initialization parameters.
        """
        d = self.d
        m = self.m
        q = self.q
        rank = self.rank

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
        remaining_imgs = num_images - parsed_images
        block_sizes = np.array([m] * int(remaining_imgs / m) + ([remaining_imgs % m] if remaining_imgs % m else []))

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

        U, S, mu, var = self.ipca.get_model()

        if self.rank == 0:
            # store current event index and reset to get same image batch
            event_index = self.psi.counter
            self.psi.counter = 0

            try: 
                print('Verifying Model Accuracy\n------------------------\n')
                print(f'q = {q}')
                print(f'd = {d}')
                print(f'n = {n}')
                print(f'm = {m}')
                print('\n')

                # run svd on centered image batch
                print('Gathering images for batch PCA...')
                X = self.fetch_formatted_images(n)

                print('Performing batch PCA...')
                mu_pca = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
                var_pca = np.reshape(np.var(X, ddof=1, axis=1), (X.shape[0], 1))
                mu_n = np.tile(mu_pca, n)
                X_centered = X - mu_n
                U_pca, S_pca, _ = np.linalg.svd(X_centered, full_matrices=False)
                print('\n')

                print(f'iPCA Compression Loss: {compression_loss(X, U)}')
                print(f'PCA Compression Loss: {compression_loss(X, U_pca[:, :q])}')
                print('\n')

                print(f'iPCA Total Variance: {np.sum(var)}')
                print(f'PCA Total Variance: {np.sum(var_pca)}')
                print('\n')

                print(f'iPCA Explained Variance: {(np.sum(S[:q]**2) / (n-1)) / np.sum(var)}')
                print(f'PCA Explained Variance: {(np.sum(S_pca[:q]**2) / (n-1)) / np.sum(var_pca)}')
                print('\n')

                print(f'iPCA Singular Values: \n')
                print(S)
                print('\n')

                print(f'PCA Singular Values: \n')
                print(S_pca[:q])
                print('\n')

                print(f'Normalized Mean Inner Product: {np.inner(mu.flatten(), mu_pca.flatten()) / (np.linalg.norm(mu) * np.linalg.norm(mu_pca))}')
                print('\n')

                print('Basis Inner Product: \n')
                print(np.diagonal(np.abs(U.T @ U_pca[:, :q])))

                b = plt.imshow(np.abs(np.hstack((mu / np.linalg.norm(mu), U)).T @ np.hstack((mu_pca / np.linalg.norm(mu_pca), U_pca[:, :q]))))
                plt.colorbar(b)
                plt.savefig(f'fig.png')
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
        print('Desired number of vectors is greater than matrix dimension.')
        return 0.
    
    acc = np.trace(np.abs(U_1[:, :q].T @ U_2[:, :q])) / q
    return acc

def compression_loss(X, U):
    """
    Calculate the compression loss between centered observation matrix X and its rank-q reconstruction.

    Parameters
    ----------
    X : ndarray, shape (d x n)
        flattened, centered image data from n run events
    U : ndarray, shape (d x q)
        first q singular vectors of X, forming orthonormal basis of q-dimensional subspace of R^d
    
    Returns
    -------
    Ln : float
        compression loss of X
    """
    d, n = X.shape

    UX = U.T @ X
    UUX = U @ UX

    Ln = ((np.linalg.norm(X - UUX, 'fro')) ** 2) / n
    return Ln 

#### For command line use ####
            
def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M',  required=True, type=str)

    parser.add_argument('--output_dir', help='Path to output directory.', required=False, type=str)
    parser.add_argument('-c', '--num_components', help='Number of principal components to compute', required=False, type=int)
    parser.add_argument('-m', '--block_size', help='Desired block size', required=False, type=int)
    parser.add_argument('-n', '--num_images', help='Number of images', required=False, type=int)
    parser.add_argument('-i', '--init_with_pca', help='Initialize with PCA', required=False, action='store_true')
    parser.add_argument('-b', '--benchmark_mode', help='Run algorithm in benchmark mode.', required=False, action='store_true')

    return parser.parse_args()
 
if __name__ == '__main__':

    params = parse_input()
    kwargs = {k: v for k, v in vars(params).items() if v is not None}

    fe = FeatureExtractor(**kwargs)
    fe.run_ipca()
    # fe.verify_model_accuracy()

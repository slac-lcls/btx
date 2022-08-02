import argparse
import numpy as np

from btx.interfaces.psana_interface import *
from btx.processing.ipca_t import *


class FeatureExtractorT:
    
    """
    Class to manage feature extraction on image data subject to initialization parameters.
    """

    def __init__(self, exp, run, det_type, q=50, block_size=10, num_images=100, init_with_pca=False):
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)

        self.d = np.prod(self.psi.det.shape())
        self.reduced_indices = np.array([])

        self.init_with_pca = init_with_pca

        # ensure that requested number of images is valid
        self.num_images = num_images
        if self.num_images > self.psi.max_events:
            self.num_images = self.psi.max_events
            print(f'Requested number of images too large, reduced to {self.num_images}')

        # ensure that requested dimension is valid
        self.q = q
        if self.q > self.num_images:
            self.q = self.num_images
            print(f'Requested number of components too large, reduced to {self.q}')

        # ensure block size is valid
        self.m = block_size
        if self.m > self.num_images:
            self.m = self.num_images
            print(f'Requested block size too large, reduced to {self.m}')

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
        PsanaInterface has internal counter that's updated on retrieval of images, ensuring
        that images are retrieved sequentially using this method.
        """
        d = self.d

        imgs = self.psi.get_images(n, assemble=False)
        print(imgs.shape)
        print(self.reduced_indices.size)

        if self.reduced_indices.size:
            reduced_imgs = np.empty((n, d))

            for i in range(n):
                reduced_imgs[i:i+1] = imgs[i][self.reduced_indices]
            
            return reduced_imgs

        return imgs

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
        parsed_images = 0

        self.ipca = IPCAT(d, q)

        if self.init_with_pca:
            img_block = self.fetch_formatted_images(q)
            self.ipca.initialize_model(img_block)

            parsed_images = q

        while parsed_images <= self.num_images:

            if parsed_images == self.num_images or (parsed_images % m == 0 and parsed_images != 0):
                current_block_size = parsed_images % m if parsed_images % m else m

                img_block = self.fetch_formatted_images(current_block_size)
                self.ipca.update_model(img_block)
            
            parsed_images += 1

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
    if q > min(U_1.shape[0], U_2.shape[0]):
        print('Desired number of vectors is greater than matrix dimension.')
        return 0.
    
    acc = np.trace(np.abs(U_1[:, :q] @ U_2[:, :q].T)) / q
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
    n, d = X.shape
    XU = X @ U.T
    XUU = XU @ U

    Ln = ((np.linalg.norm(X - XUU, 'fro')) ** 2) / n
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

    parser.add_argument('--components', help='Number of principal components to compute', required=False, type=int)
    parser.add_argument('--block_size', help='Desired block size', required=False, type=int)
    parser.add_argument('--num_images', help='Number of images', required=False, type=int)
    parser.add_argument('--init_with_pca', help='Initialize with PCA', required=False, action='store_true')

    return parser.parse_args()
 
if __name__ == '__main__':
    
    params = parse_input()
    fe = FeatureExtractor(exp=params.exp, run=params.run, det_type=params.det_type, q=params.components, block_size=params.block_size, num_images=params.num_images)
    fe.run_ipca()
    fe.ipca.report_interval_data()
    

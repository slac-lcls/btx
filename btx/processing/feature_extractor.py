from btx.interfaces.psana_interface import *

from btx.processing.ipca import *


class FeatureExtractor:
    
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
        formatted_imgs = np.empty((d, n))

        for i in range(n):
            if self.reduced_indices.size:
                formatted_imgs[:, i:i+1] = np.reshape(imgs[i], (d, 1))[self.reduced_indices]
            else:
                formatted_imgs[:, i:i+1] = np.reshape(imgs[i], (d, 1))

        return formatted_imgs

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

        self.ipca = IPCA(d, q)

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

        
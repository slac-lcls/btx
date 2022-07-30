from btx.interfaces.psana_interface import *

from btx.processing.ipca import *


class IPCAWrapper:

    def __init__(self, exp, run, det_type, q=50, block_size=10, num_images=100, init_with_pca=False):
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)

        self.d = np.prod(self.psi.det.shape())
        self.reduced_indices = np.array([])

        self.q = q
        self.m = block_size
        self.num_images = num_images
        self.init_with_pca = init_with_pca
    
    def initialize_ipca(self):
        ipca = IPCA(d, q)
        
        if self.init_with_pca:
            ipca.initialize_model()

    def fetch_formatted_images(self, n):
        d = self.d

        imgs = self.psi.get_images(n, assemble=False)

        formatted_imgs = np.empty((d, n))

        for i in range(n):
            if self.reduced_indices.size:
                formatted_imgs[:, i] = np.reshape(imgs[i], (d, 1))[self.reduced_indices]
            else:
                formatted_imgs[:, i] = np.reshape(imgs[i], (d, 1))

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
        d = self.d
        m = self.m
        q = self.q

        self.ipca = IPCA(d, q)

        parsed_images = 0

        if self.init_with_pca:
            img_block = self.fetch_formatted_images(q)
            self.ipca.initialize_model(img_block)

            parsed_images = q

        max_images = min(self.num_images, self.psi.max_events)

        while parsed_images <= max_images:

            if parsed_images == max_images or parsed_images % m == 0:
                current_block_size = parsed_images % m if parsed_images % m else m

                img_block = self.fetch_formatted_images(current_block_size)
                self.ipca.update_model(img_block)
            
            parsed_images += 1

        
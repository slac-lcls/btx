import numpy as np
import h5py
import os
import sys
import glob
from btx.interfaces.ipsana import *

class MaskInterface:
    
    def __init__(self, exp, run, det_type):
        self.exp = exp # experiment name, str
        self.run = run # run number, int
        self.det_type = det_type # detector name, str
        self.mask = None
        
        self.psi = PsanaInterface(exp=self.exp, run=self.run, det_type=self.det_type) 
        self.pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(self.run))

    def mask_border_pixels(self, n_edge):
        """
        Mask the edge of each panel with a border n_edge pixels deep.
        
        Parameters
        ----------
        n_edge : int
            number of pixels along each panel edge to mask
        """
        panel_mask = np.zeros(self.mask.shape[-2:]).astype(int)
        panel_mask[n_edge:-n_edge,n_edge:-n_edge] = 1
    
        if len(self.mask.shape) == 2:
            self.mask *= panel_mask
        else:
            for panel in range(self.mask.shape[0]):
                self.mask[panel] *= panel_mask

    def generate_from_psana_run(self, thresholds, n_images=1, n_edge=0):
        """
        Generate a mask by extracting n_images random images from the indicated run,
        thresholding each, and then merging them. A value of 0 indicates a bad pixel.
        If n_edge is supplied, the border pixels of each panel will be masked as well.

        Parameters
        ----------
        thresholds : tuple, 2d
            (lower, upper) thresholds for pixel value
        n_images : int
            number of images to threshold
        n_edge : int
            depth of border in pixels to mask for each panel
        """
        # retrieve random events, excluding first of run due to Rayonix oddity
        imgs = np.zeros((n_images,) + self.psi.det.shape())
        indices = np.random.randint(1, high=self.psi.max_events, size=n_images)
        for i,idx in enumerate(indices):
            evt = self.psi.runner.event(self.psi.times[idx])
            imgs[i] = self.psi.det.calib(evt=evt)
    
        # apply thresholds and set valid pixels to 1
        imgs[(imgs < thresholds[0]) | (imgs > thresholds[1])] = 0
        mask = np.prod(imgs, axis=0)
        mask[mask!=0] = 1
        mask = mask.astype(int)
        
        if self.mask is None:
            self.mask = mask
        else:
            self.mask *= mask

        if n_edge!=0:
            self.mask_border_pixels(n_edge)

    def retrieve_from_mrxv(self, mrxv_path='/cds/sw/package/autosfx/mrxv/masks', dataset='/entry_1/data_1/mask'):
        """
        Retrieve the latest mask from mrxv.

        Parameters
        ----------
        dataset : str
            internal path to dataset, only relevant for mask_format='crystfel'
        """
        try:
            mask_file = glob.glob(os.path.join(mrxv_path, f"{self.det_type}_latest.*"))[0]
            assert os.path.isfile(mask_file)
            print(f"Retrieving mask file {mask_file}")
        except:
            sys.exit("Detector type not yet available in mrxv")

        if h5py.is_hdf5(mask_file):
            print("Crystfel mask detected")
            mask_format = 'crystfel'
        elif mask_file.split('.')[-1] == 'npy':
            mask_format = 'psana'
        else:
            mask_format = 'cctbx'

        self.load_mask(mask_file, mask_format=mask_format, dataset=dataset)
        assert self.psi.det.shape() == self.mask.shape
    
    def load_mask(self, input_file, mask_format='crystfel', dataset='/entry_1/data_1/mask'):
        """
        Load input mask and reshape from given format to psana unassembled.
        If self.masks is None, store to class variable. Otherwise create a 
        combined mask by multipying this and current self.mask.
        
        Parameters
        ----------
        input_file : str
            path to input mask
        mask_format : str, default='crystfel'
            input style - psana, psana_assembled, crystfel, or cctbx
        dataset : str
            internal path to dataset, only relevant for mask_format='crystfel'
        """
        if mask_format == 'psana':
            mask = np.load(input_file)
            
        elif mask_format == 'psana_assembled':
            mask = diassemble_image_stack_batch(np.load(input_file), self.pixel_index_map)
        
        elif mask_format == 'crystfel':
            mask = load_crystfel_mask(input_file, dataset=dataset, reshape=True)
            
        elif mask_format == 'cctbx':
            mask = load_cctbx_mask(input_file)
            if 'epix10k' in self.det_type:
                mask = stack_asics(mask, det_type=self.det_type)
            else:
                sys.exit("Sorry, detector type currently not supported for reshaping from CCTBX format")
        
        else:
            sys.exit("Mask format not recognized.")
            
        if self.mask is None:
            self.mask = mask.astype(int)
        else:
            print("Combining input mask with current self.mask")
            assert self.mask.shape == mask.shape
            self.mask *= mask.astype(int)
    
    def _save_as_hdf5(self, output, mask):
        """
        Save input mask to hdf5 format, reshaped for CrystFEL compatibility.
        Reshaping is based on the looping performed here:
        https://github.com/cctbx/dxtbx/blob/main/src/dxtbx/format/FormatXTCEpix.py#L46-L57.
        
        Parameters
        ----------
        output : str
            path to output mask file
        mask : numpy.ndarray, shape (n_panels * n_pixels_fs, n_pixels_ss)
            binary mask 
        """
        f = h5py.File(output, "w")
        dset = f.create_dataset('/entry_1/data_1/mask', data=mask, dtype='int')
        f.close()
    
    def _convert_to_cctbx(self):
        """
        Remake mask in CCTBX format, by reshaping and converting to flex.bool.
        Currently only the Jungfrau and Epix10k detectors are supported.
        
        Returns
        -------
        mask : list of flex bool objects
            binary mask in CCTBX format
        """
        from scitbx.array_family import flex
        
        if 'epix10k' in self.det_type:
            mask = unstack_asics(self.mask, self.det_type, dtype='double')
        
        elif 'jungfrau' in self.det_type:
            from xfel.util import jungfrau
            
            mask = []
            for n_panel in range(self.mask.shape[0]):
                mask.append(flex.bool(jungfrau.correct_panel(self.mask[n_panel].astype('float64'))))

        else:
            sys.exit("Sorry, detector type currently not supported for saving to CCTBX format")
            
        return mask
                
    def save_mask(self, output, mask_format='psana'):
        """
        Save mask to one of four formats: psana (.npy), psana_assembled (.npy), 
        crystfel (.h5), or cctbx (.pickle).
        
        Parameters
        ----------
        output : str
            path to output mask file
        mask_format : str
            output style - psana, psana_assembled, crystfel, or cctbx
        """
        if mask_format == 'psana':
            np.save(output, self.mask)
            
        elif mask_format == 'psana_assembled':
            np.save(output, assemble_image_stack_batch(self.mask, self.pixel_index_map))
        
        elif mask_format == 'crystfel':
            mask = self.mask.reshape(-1, self.mask.shape[-1])
            self._save_as_hdf5(output, mask)
            print(f"Saved mask to {output} in CrystFEL format, with shape {mask.shape}")

        elif mask_format == 'cctbx':
            try:
                from libtbx import easy_pickle
            except ImportError:
                sys.exit("Could not load libtbx library")
            
            # supposed to be a list of flex bool objects
            mask = self._convert_to_cctbx()
            easy_pickle.dump(output, mask)
            print(f"Saved mask to {output} in CCTBX format")
        
        else:
            sys.exit("Mask format not recognized.")
            
    def visualize_mask(self, output=None):
        """
        Visualize self.mask after assembling into 2d detector format.
        
        Parameters
        ----------
        output : str, default: None
            if provided, save mask to this path
        """
        f, ax1 = plt.subplots(figsize=(4,4))
        ax1.imshow(assemble_image_stack_batch(self.mask, self.pixel_index_map))
        
        if output:
            f.savefig(output, bbox_inches='tight', dpi=300)

#### Miscellaneous functions ####
    
def unstack_asics(image, det_type, dtype='double'):
    """
    Unstack the asics and convert to CCTBX data type - specifically, a list of 
    flex.double or flex.bool objects. For the epix10k2M, for example, the shape
    is updated from (16,352,384) -> (64,176,192). This mimics the code here:
    https://github.com/cctbx/dxtbx/blob/main/src/dxtbx/format/FormatXTCEpix.py#L46-L57.
    
    Parameters
    ----------
    image : numpy.ndarray, shape (n_panels, n_pixels_fs, n_pixels_ss)
        image in unassembled psana format
    det_type : str
        epix10k2M or jungfrau4M
    dtype : str
        return data type, double or (for masks) bool
    
    Returns
    -------
    reshaped_image : list of flex.bool or flex.double arrays
        list of unstacked asics data
    """
    try:
        from scitbx.array_family import flex
    except ImportError:
        sys.exit("Could not load libtbx flex library")
    
    n_asics_per_module = 0
    if 'epix10k' in det_type:
        n_asics_per_module = 4 # 2x2 asics per module
    if 'jungfrau' in det_type:
        n_asics_per_module = 8 # 2x4 asics per module
    if n_asics_per_module == 0:
        sys.exit("Sorry, detector type currently not supported for CCTBX-style asics-stacking.")
    
    reshaped_image = []
    sdim, fdim = int(image.shape[1]/2), int(image.shape[2]/2)
    for n_panel in range(image.shape[0]):
        for n_asic in range(n_asics_per_module):
            sensor_id = n_asic // int(n_asics_per_module/2)
            asic_in_sensor_id = n_asic % int(n_asics_per_module/2)
            asic_data = image[n_panel][sensor_id * sdim : (sensor_id + 1) * sdim,
                                      asic_in_sensor_id * fdim : (asic_in_sensor_id + 1) * fdim,]
            if dtype == 'double':
                reshaped_image.append(flex.double(np.array(asic_data)))
            elif dtype == 'bool':
                reshaped_image.append(flex.bool(np.array(asic_data)))
            else:
                sys.exit("Print unrecognized data type; must be bool or double")
            
    return reshaped_image

def stack_asics(image, det_type):
    """
    Stack the asics to reshape from CCTBX to psana (unassembled) conventions.
    For the epix10k2M, for example, the shape is updated from (64,176,192) -> 
    (16,352,384). This reverses the code here:
    https://github.com/cctbx/dxtbx/blob/main/src/dxtbx/format/FormatXTCEpix.py#L46-L57.
    
    Parameters
    ----------
    image : numpy.ndarray, shape (n_asics, n_asics_pixels_fs, n_asics_pixels_ss)
        image in unstacked asics format
    det_type : str
        epix10k2M or jungfrau4M
    
    Returns
    -------
    reshaped_image : numpy.ndarray, shape (n_panels, n_pixels_fs, n_pixels_ss)
        image in unassembled psana format
    """    
    n_asics_per_module = 0
    if 'epix10k' in det_type:
        n_asics_per_module = 4 # 2x2 asics per module
    if 'jungfrau' in det_type:
        n_asics_per_module = 8 # 2x4 asics per module
        print("Warning: currently unstacking doesn't account for CCTBX's big pixel adjustment.")
    if n_asics_per_module == 0:
        sys.exit("Sorry, detector type currently not supported for CCTBX-style asics-stacking.")
        
    # asic unstack into mmodules; there are 2x2 asics per module
    sdim, fdim = image.shape[1:]
    reshaped_image = np.zeros((int(image.shape[0]/4), sdim*2, fdim*2))

    counter = 0
    for module_count in range(reshaped_image.shape[0]):
        for asic_count in range(n_asics_per_module):
            sensor_id = asic_count // int(n_asics_per_module/2) 
            asic_in_sensor_id = asic_count % int(n_asics_per_module/2) 
            reshaped_image[module_count][sensor_id * sdim : (sensor_id + 1) * sdim,
                               asic_in_sensor_id * fdim : (asic_in_sensor_id + 1) * fdim,] = image[counter]
            counter+=1

    return reshaped_image

def load_cctbx_mask(input_file):
    """
    Load a CCTBX mask, converting from a list of flex.bools to 
    a numpy array.

    Parameters
    ----------
    input_file : str
        path to CCTBX mask in pickle format
    
    Returns
    -------
    mask : numpy.ndarray, CCTBX-compatible shape
        binary mask
    """
    try:
        from libtbx import easy_pickle
    except ImportError:
        sys.exit("Could not load libtbx flex library")

    mask = easy_pickle.load(input_file)
    mask = np.array([m.as_numpy_array() for m in mask])
    return mask

def load_crystfel_mask(input_file, dataset='/entry_1/data_1/mask', reshape=True):
    """
    Load a CrystFEL compatible mask from the input h5 file and optionally 
    reshape to the unassembled detector shape. Currently only Jungfrau4M
    and epix10k2M detectors are supported for reshape=True.
    
    Parameters
    ----------
    input_file : str
        path to h5 file containing mask
    dataset : str
        internal path to dataset, default is psocake compatible
    reshape : bool
        if True, reshape to unassembled psana detector shape
        
    Returns
    -------
    mask : numpy.ndarray, shape depends on reshape flag
        binary mask. if not reshape, shape is (n_panels * n_pixels_fs, n_pixels_ss)
        if reshape, shape is (n_panels, n_pixels_fs, n_pixels_ss)
    """
    f = h5py.File(input_file, "r")
    mask = f[dataset][:]
    f.close()
    
    if reshape:
        if np.prod(mask.shape) == 2162688: # epix10k2M
            mask = mask.reshape(16, 352, 384)
        elif np.prod(mask.shape) == 4194304: # jungfrau4M
            mask = mask.reshape(8, 512, 1024)
        else: 
            print("Mask did not match epix10k2M or Jungfrau4M dimensions; could not reshape")
        
    return mask

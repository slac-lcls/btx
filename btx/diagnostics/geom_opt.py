import numpy as np
import sys
import requests
from mpi4py import MPI
from btx.diagnostics.run import RunDiagnostics
from btx.interfaces.ipsana import assemble_image_stack_batch
from btx.misc.metrology import *
from btx.misc.radial import pix2q
from .ag_behenate import *

class GeomOpt:
    
    def __init__(self, exp, run, det_type):
        self.diagnostics = RunDiagnostics(exp=exp, # experiment name, str
                                          run=run, # run number, int
                                          det_type=det_type) # detector name, str
        self.center = None
        self.distance = None
        self.edge_resolution = None

        # have a rank variable available in case we're running via a multi-core DAG
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

    def opt_geom(self, powder, sample='AgBehenate', mask=None, center=None, distance=None,
                 n_iterations=5, n_peaks=3, threshold=1e6, plot=None):
        """
        Estimate the sample-detector distance based on the properties of the powder
        diffraction image. Currently only implemented for silver behenate.
        
        Parameters
        ----------
        powder : str or int
            if str, path to the powder diffraction in .npy format
            if int, number of images from which to compute powder 
        sample : str
            sample type, currently implemented for AgBehenate only
        mask : str
            npy file of mask in psana unassembled detector shape
        center : tuple
            detector center (xc,yc) in pixels. if None, assume assembled image center.
        distance : float
            sample-detector distance in mm. If None, pull from calib file.
        n_iterations : int
            number of refinement steps
        n_peaks : int
            number of observed peaks to use for center fitting
        threshold : float
            pixels above this intensity in powder get set to 0; None for no thresholding.
        plot : str or None
            output path for figure; if '', plot but don't save; if None, don't plot

        Returns
        -------
        distance : float
            estimated sample-detector distance in mm
        """
        if type(powder) == str:
            powder_img = np.load(powder)
        
        elif type(powder) == int:
            print("Computing powder from scratch")
            self.diagnostics.compute_run_stats(n_images=powder, powder_only=True)
            if self.diagnostics.psi.det_type != 'Rayonix':
                powder_img = assemble_image_stack_batch(self.diagnostics.powders['max'], 
                                                        self.diagnostics.pixel_index_map)
        
        else:
            sys.exit("Unrecognized powder type, expected a path or number")
        
        if mask:
            print(f"Loading mask {mask}")
            mask = np.load(mask)
            if self.diagnostics.psi.det_type != 'Rayonix':
                mask = assemble_image_stack_batch(mask, self.diagnostics.pixel_index_map)

        if distance is None:
            distance = self.diagnostics.psi.estimate_distance()

        if sample == 'AgBehenate':
            ag_behenate = AgBehenate(powder_img,
                                     mask,
                                     self.diagnostics.psi.get_pixel_size(),
                                     self.diagnostics.psi.get_wavelength())
            ag_behenate.opt_geom(distance, 
                                 n_iterations=n_iterations, 
                                 n_peaks=n_peaks, 
                                 threshold=threshold, 
                                 center_i=center, 
                                 plot=plot)
            self.distance = ag_behenate.distances[-1] # in mm
            self.center = ag_behenate.centers[-1] # in pixels
            self.edge_resolution = 1.0 / pix2q(np.array([powder_img.shape[0]/2]), 
                                               self.diagnostics.psi.get_wavelength(), 
                                               self.distance, 
                                               self.diagnostics.psi.get_pixel_size())[0] # in Angstrom

        else:
            print("Sorry, currently only implemented for silver behenate")
            return -1

    def deploy_geometry(self, outdir):
        """
        Write new geometry files (.geom and .data for CrystFEL and psana respectively) 
        with the optimized center and distance.
    
        Parameters
        ----------
        center : tuple
            optimized center (cx, cy) in pixels
        distance : float
            optimized sample-detector distance in mm
        outdir : str
            path to output directory
        """
        # retrieve original geometry
        run = self.diagnostics.psi.run
        geom = self.diagnostics.psi.det.geometry(run)
        top = geom.get_top_geo()
        children = top.get_list_of_children()[0]
        pixel_size = self.diagnostics.psi.get_pixel_size() * 1e3 # from mm to microns
    
        # determine and deploy shifts in x,y,z
        cy, cx = self.diagnostics.psi.det.point_indexes(run, pxy_um=(0,0), fract=True)
        dx = pixel_size * (self.center[0] - cx) # convert from pixels to microns
        dy = pixel_size * (self.center[1] - cy) # convert from pixels to microns
        dz = np.mean(-1*self.diagnostics.psi.det.coords_z(run)) - 1e3 * self.distance # convert from mm to microns
        geom.move_geo(children.oname, 0, dx=-dy, dy=-dx, dz=dz) 
    
        # write optimized geometry files
        psana_file, crystfel_file = os.path.join(outdir, f"r{run:04}_end.data"), os.path.join(outdir, f"r{run:04}.geom")
        temp_file = os.path.join(outdir, "temp.geom")
        geom.save_pars_in_file(psana_file)
        generate_geom_file(self.diagnostics.psi.exp, run, self.diagnostics.psi.det_type, psana_file, temp_file)
        modify_crystfel_header(temp_file, crystfel_file)
        os.remove(temp_file)

        # Rayonix check
        if self.diagnostics.psi.get_pixel_size() != self.diagnostics.psi.det.pixel_size(run):
            print("Original geometry is wrong due to hardcoded Rayonix pixel size. Correcting geom file now...")
            coffset = (self.distance - self.diagnostics.psi.get_camera_length()) / 1e3 # convert from mm to m
            res = 1e3 / self.diagnostics.psi.get_pixel_size() # convert from mm to um
            os.rename(crystfel_file, temp_file)
            modify_crystfel_coffset_res(temp_file, crystfel_file, coffset, res)
            os.remove(psana_file)
            os.remove(temp_file)

    def report(self, update_url):
        """
        Post summary to elog.
       
        Parameters
        ----------
        update_url : str
            elog URL for posting progress update
        """
        requests.post(update_url, json=[{ "key": "Detector distance (mm)", "value": f"{self.distance:.2f}" },
                                        { "key": "Detector center (pixels)", "value": f"({self.center[0]:.2f}, {self.center[1]:.2f})" },
                                        { "key": "Detector edge resolution (A)", "value": f"{self.edge_resolution:.2f}" }, ])

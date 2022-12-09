import numpy as np
import argparse
import os
from mpi4py import MPI
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from btx.interfaces.ipsana import *

from psana import EventId
from Detector.UtilsEpix10ka import find_gain_mode
from Detector.UtilsEpix10ka import info_pixel_gain_mode_statistics_for_raw
from Detector.UtilsEpix10ka import map_pixel_gain_mode_for_raw

class RunDiagnostics:

    """
    Class for computing powders and a trajectory of statistics from a given run.
    """
    
    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type, track_timestamps=True)
        self.pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(run))
        self.powders = dict() 
        self.stats = dict()
        self.gain_traj = None
        self.gain_map = None
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size() 

    def compute_base_powders(self, img):
        """
        Compute the base powders: max, sum, sum of squares.

        Parameters
        ----------
        img : numpy.ndarray, 3d
            unassembled, calibrated images of shape (n_panels, n_x, n_y)
        """
        if not self.powders:
            for key in ['sum', 'sqr', 'max']:
                self.powders[key] = img.copy()
        else:
            self.powders['sum'] += img
            self.powders['sqr'] += np.square(img)
            self.powders['max'] = np.maximum(self.powders['max'], img)

    def finalize_powders(self):
        """
        Finalize powders calculation at end of the run, computing the
        max, avg, and std dev (and gain mode counts if applicable).
        """
        self.powders_final = dict()
        powder_max = np.array(self.comm.gather(self.powders['max'], root=0))
        powder_sum = np.array(self.comm.gather(self.powders['sum'], root=0))
        powder_sqr = np.array(self.comm.gather(self.powders['sqr'], root=0))
        total_n_proc = self.comm.reduce(self.n_proc, MPI.SUM)
        if self.gain_map is not None:
            powder_gain = np.array(self.comm.gather(self.gain_map, root=0))

        if self.rank == 0:
            self.powders_final['max'] = np.max(powder_max, axis=0)
            self.powders_final['avg'] = np.sum(powder_sum, axis=0) / float(total_n_proc)
            self.powders_final['std'] = np.sqrt(np.sum(powder_sqr, axis=0) / float(total_n_proc) - np.square(self.powders_final['avg']))
            if self.gain_map is not None:
                self.powders_final['gain_mode_counts'] = np.sum(powder_gain, axis=0)
            if self.psi.det_type != 'Rayonix':
                for key in self.powders_final.keys():
                    self.powders_final[key] = assemble_image_stack_batch(self.powders_final[key], self.pixel_index_map)
                    
        self.panel_mask = assemble_image_stack_batch(np.ones(self.psi.det.shape()), self.pixel_index_map)
 
    def save_powders(self, outdir):
        """
        Save powders to output directory.

        Parameters
        ----------
        output : str
            path to directory in which to save powders, optional
        """
        for key in self.powders_final.keys():
            np.save(os.path.join(outdir, f"r{self.psi.run:04}_{key}.npy"), self.powders_final[key])

    def compute_stats(self, img):
        """
        Compute the following stats: mean, std deviation, max, min.

        Parameters
        ----------
        img : numpy.ndarray, 3d
            unassembled, calibrated images of shape (n_panels, n_x, n_y)
        """
        if not self.stats:
            for key in ['mean','std','max','min']:
                self.stats[key] = np.zeros(self.psi.max_events - self.psi.counter)
        
        self.stats['mean'][self.n_proc] = np.mean(img)
        self.stats['std'][self.n_proc] = np.std(img)
        self.stats['max'][self.n_proc] = np.max(img)
        self.stats['min'][self.n_proc] = np.min(img)
        
    def finalize_stats(self, n_empty=0, n_empty_raw=0):
        """
        Gather stats from various ranks into single arrays in self.stats_final.

        Parameters
        ----------
        n_empty : int
            number of empty images in this rank
        n_empty_raw : int
            number of empty raw events in this rank
        """
        self.stats_final = dict()
        for key in self.stats.keys():
            if n_empty != 0:
                self.stats[key] = self.stats[key][:-n_empty]
            self.stats_final[key] = self.comm.gather(self.stats[key], root=0)

        self.stats_final['fiducials'] = self.comm.gather(np.array(self.psi.fiducials), root=0)
        if self.rank == 0:
            for key in self.stats_final.keys():
                self.stats_final[key] = np.hstack(self.stats_final[key])
                
        if self.gain_traj is not None:
            if n_empty_raw != 0:
                self.gain_traj = self.gain_traj[:-n_empty_raw]
            self.stats_final['gain_mode_counts'] = self.comm.gather(self.gain_traj, root=0)
            if self.rank == 0:
                self.stats_final['gain_mode_counts'] = np.hstack(self.stats_final['gain_mode_counts'])
                
    def get_gain_statistics(self, raw, gain_mode='AML-L'):
        """
        Retrieve statistics for a particular gain mode, specifically a map 
        of the number of times a pixel has been in a certain gain mode and 
        the number of pixels in that gain mode per event.

        Parameters
        ----------
        raw : ndarray, shape (det_shape)
            uncalibrated image
        gain_mode : str
            gain mode to retrieve statistics for, e.g. 'AML-L'
        """
        if self.gain_traj is None:
            self.gain_mode = gain_mode
            self.modes = {'FH':0, 'FM':1, 'FL':2, 'AHL-H':3, 'AML-M':4, 'AHL-L':5, 'AML-L':6}
            self.gain_traj = np.zeros(self.psi.max_events - self.psi.counter).astype(int)
            self.gain_map = np.zeros(self.psi.det.shape())

        stats = info_pixel_gain_mode_statistics_for_raw(self.psi.det, raw)
        self.gain_traj[self.n_proc] = int(stats.split(":")[1].split(",")[self.modes[gain_mode]])
        evt_map = map_pixel_gain_mode_for_raw(self.psi.det, raw)
        self.gain_map[evt_map==[self.modes[gain_mode]]] += 1
            
    def compute_run_stats(self, max_events=-1, mask=None, powder_only=False, threshold=None, gain_mode=None):
        """
        Compute powders and per-image statistics. If a mask is provided, it is 
        only applied to the stats trajectories, not in computing the powder.
        
        Parameters
        ----------
        max_events : int
            number of images to process; if -1, process entire run
        mask : str or np.ndarray, shape (n_panels, n_x, n_y)
            binary mask file or array in unassembled psana shape, optional 
        powder_only : bool
            if True, only compute the powder pattern
        threshold : float
            if supplied, exclude events whose mean is above this value
        gain_mode : str
            gain mode to retrieve statistics for, e.g. 'AML-L' 
        """
        if mask is not None:
            if type(mask) == str:
                mask = np.load(mask) 
            assert mask.shape == self.psi.det.shape()

        self.psi.distribute_events(self.rank, self.size, max_events=max_events)
        start_idx, end_idx = self.psi.counter, self.psi.max_events
        self.n_proc, n_empty, n_empty_raw, n_excluded = 0, 0, 0, 0

        if self.psi.det_type == 'Rayonix':
            if self.rank == 0:
                if self.check_first_evt(mask=mask):
                    print("First image of the run is an outlier and will be excluded")
                    start_idx += 1
                    
        for idx in np.arange(start_idx, end_idx):
            evt = self.psi.runner.event(self.psi.times[idx])
            self.psi.get_timestamp(evt.get(EventId))
            img = self.psi.det.calib(evt=evt)
            if img is None:
                n_empty += 1
                continue
                
            if threshold:
                if np.mean(img) > threshold:
                    print(f"Excluding event {idx} with image mean: {np.mean(img)}")
                    n_excluded += 1
                    continue

            self.compute_base_powders(img)
            if not powder_only:
                if mask is not None:
                    img = np.ma.masked_array(img, 1-mask)
                self.compute_stats(img)
                
            if gain_mode is not None and self.psi.det_type == 'epix10k2M':
                raw = self.psi.det.raw(evt)
                if raw is None:
                    n_empty_raw +=1
                    continue
                self.get_gain_statistics(raw, gain_mode)
            else:
                self.gain_mode = ''

            self.n_proc += 1
            if self.psi.counter + n_empty + n_excluded == self.psi.max_events:
                break

        self.comm.Barrier()
        self.finalize_powders()
        if not powder_only:
            self.finalize_stats(n_empty + n_excluded, n_empty_raw)
            print(f"Rank {self.rank}, no. empty images: {n_empty}, no. excluded images: {n_excluded}")

    def visualize_powder(self, tag='max', vmin=-1e5, vmax=1e5, output=None, figsize=12, dpi=300):
        """
        Visualize the powder image: the distribution of intensities as a histogram
        and the positive and negative-valued pixels on the assembled detector image.
        """
        if self.rank == 0:
            image = self.powders_final[tag]
        
            fig = plt.figure(figsize=(figsize,figsize),dpi=dpi)
            gs = fig.add_gridspec(2,2)

            irow=0
            ax1 = fig.add_subplot(gs[irow,:2])
            ax1.grid()
            ax1.hist(image.flatten(), bins=100, log=True, color='black')
            ax1.set_title(f'histogram of pixel intensities in powder {tag}', fontdict={'fontsize': 8})

            irow+=1
            ax2 = fig.add_subplot(gs[irow,0])
            im = ax2.imshow(np.where(image>0,0,image), cmap=plt.cm.gist_gray, 
                            norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=vmin, vmax=0.))
            ax2.axis('off')
            ax2.set_title(f'negative intensity pixels', fontdict={'fontsize': 6})
            plt.colorbar(im)

            ax3 = fig.add_subplot(gs[irow,1])
            im = ax3.imshow(np.where(image<0,0,image), cmap=plt.cm.gist_yarg, 
                            norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=0, vmax=vmax))
            ax3.axis('off')
            ax3.set_title(f'positive intensity pixels', fontdict={'fontsize': 6})
            plt.colorbar(im)

            if output is not None:
                plt.savefig(output)
        
    def visualize_stats(self, output=None):
        """
        Plot trajectories of run statistics.
        
        Parameters
        ----------
        output : str
            path for optionally saving plot to disk
        """
        if self.rank == 0:
            n_plots = len(self.stats_final.keys())-1
            keys = ['mean', 'max', 'min', 'std', 'gain_mode_counts']
            labels = ['mean(I)', 'max(I)', 'min(I)', 'std dev(I)', f'No. pixels in\n{self.gain_mode} mode']
            
            f, axs = plt.subplots(n_plots, figsize=(n_plots*2.4,8), sharex=True)
            for n in range(n_plots):
                axs[n].plot(self.stats_final[keys[n]], c='black')
                axs[n].set_ylabel(labels[n], fontsize=12)       
            axs[n_plots-1].set_xlabel("Event index", fontsize=12)
            
            if output is not None:
                f.savefig(output, dpi=300)
                
    def visualize_gain_frequency(self, output=None):
        """
        Plot the distribution of the frequency with which each pixel
        appears in a particular gain mode.
        
        Parameters
        ----------
        output : str
            path for optionally saving plot to disk
        """
        if self.gain_map is None:
            print("Gain statistics were not retrieved.")
            return
            
        if self.rank == 0:
            f, ax1 = plt.subplots(figsize=(6,3.6))

            freq = self.powders_final['gain_mode_counts'][self.panel_mask==1]/len(self.stats_final['gain_mode_counts'])
            ax1.hist(freq, bins=50, color='black')
            ax1.set_yscale('log')

            ax1.set_ylabel("No. pixels", fontsize=12)
            ax1.set_xlabel(f"Frequency in {self.gain_mode} mode", fontsize=12)
            
            if output is not None:
                f.savefig(output, dpi=300, bbox_inches='tight')
    
    def check_first_evt(self, mask=None, scale_factor=5, n_images=5):
        """
        Check whether the first event of the run should be excluded; it's 
        considered an outlier if mean_0 > < mean_n + scale_factor * std_n >
        where n ranges from [1,n_images). If any of the first few events are
        None, exclude the first image to simplify matters.

        Parameters
        ----------
        mask : np.ndarray, shape (n_panels, n_x, n_y)
            binary mask file or array in unassembled psana shape, optional 
        scale_factor : float
            parameter that tunes  how conservative outlier rejection is
        n_images : int
            number of total images to compare, including first

        Returns:
        --------
        exclude : bool
            if True, first event was detected as an outlier
        """
        exclude = False
        means, stds = np.zeros(n_images), np.zeros(n_images)
        for cidx in range(n_images):
            evt = self.psi.runner.event(self.psi.times[cidx])
            img = self.psi.det.calib(evt=evt)
            if img is None:
                return exclude
            if mask is not None:
                img *= mask
            means[cidx], stds[cidx] = np.mean(img), np.std(img)
            
        if means[0] > np.mean(means[1:] + scale_factor * stds[1:]):
            exclude = True
        return exclude

#### For command line use ####
            
def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M',  required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory for powders and plots', required=True, type=str)
    parser.add_argument('-m', '--mask', help='Binary mask for computing trajectories', required=False, type=str)
    parser.add_argument('--max_events', help='Number of images to process, -1 for full run', required=False, default=-1, type=int)
    parser.add_argument('--mean_threshold', help='Exclude images with a mean above this threshold', required=False, type=float)

    return parser.parse_args()

if __name__ == '__main__':
    
    params = parse_input()
    rd = RunDiagnostics(exp=params.exp, run=params.run, det_type=params.det_type) 
    rd.compute_run_stats(max_events=params.max_events, mask=params.mask, threshold=params.mean_threshold) 
    rd.save_powders(params.outdir)
    rd.visualize_powder(output=os.path.join(params.outdir, f"powder_r{rd.psi.run:04}.png"))
    rd.visualize_stats(output=os.path.join(params.outdir, f"stats_r{rd.psi.run:04}.png"))


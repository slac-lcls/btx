import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

from psana import EventId
from btx.interfaces.ipsana import *
from mpi4py import MPI

class RunDiagnostics:

    """
    Class for computing powders and a trajectory of statistics from a given run.
    """
    
    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type, track_timestamps=True)
        self.pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(run))
        self.powders = dict() 
        self.stats = dict()
        
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
        max, avg, and std dev versions.
        """
        self.powders_final = dict()
        powder_max = np.array(self.comm.gather(self.powders['max'], root=0))
        powder_sum = np.array(self.comm.gather(self.powders['sum'], root=0))
        powder_sqr = np.array(self.comm.gather(self.powders['sqr'], root=0))
        total_n_proc = self.comm.reduce(self.n_proc, MPI.SUM)

        if self.rank == 0:
            self.powders_final['max'] = np.max(powder_max, axis=0)
            self.powders_final['avg'] = np.sum(powder_sum, axis=0) / float(total_n_proc)
            self.powders_final['std'] = np.sqrt(np.sum(powder_sqr, axis=0) / float(total_n_proc) - np.square(self.powders_final['avg']))
            if self.psi.det_type != 'Rayonix':
                for key in self.powders_final.keys():
                    self.powders_final[key] = assemble_image_stack_batch(self.powders_final[key], self.pixel_index_map)

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
        
    def finalize_stats(self, n_empty=0):
        """
        Gather stats from various ranks into single arrays in self.stats_final.

        Parameters
        ----------
        n_empty : int
            number of empty images in this rank
        """
        self.stats_final = dict()
        for key in self.stats.keys():
            if n_empty != 0:
                self.stats_final[key] = self.stats[key][:-n_empty]
            self.stats_final[key] = self.comm.gather(self.stats[key], root=0)

        self.stats_final['fiducials'] = self.comm.gather(np.array(self.psi.fiducials), root=0)
        if self.rank == 0:
            for key in self.stats_final.keys():
                #self.stats_final[key] = np.array(self.stats_final[key]).reshape(-1)
                self.stats_final[key] = np.hstack(self.stats_final[key])

    def compute_run_stats(self, max_events=-1, mask=None, powder_only=False, threshold=None):
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
        """
        if mask is not None:
            if type(mask) == str:
                mask = np.load(mask) 
            assert mask.shape == self.psi.det.shape()

        self.psi.distribute_events(self.rank, self.size, max_events=max_events)
        start_idx, end_idx = self.psi.counter, self.psi.max_events
        self.n_proc, n_empty, n_excluded = 0, 0, 0

        if self.psi.det_type == 'Rayonix':
            if self.rank == 0:
                if self.check_first_evt(mask=mask):
                    print("First image of the run is an outlier and will be excluded")
                    start_idx += 1

        for idx in np.arange(start_idx, end_idx):

            # retrieve calibrated image
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

            self.n_proc += 1
            if self.psi.counter + n_empty + n_excluded == self.psi.max_events:
                break

        self.comm.Barrier()
        self.finalize_powders()
        if not powder_only:
            self.finalize_stats(n_empty + n_excluded)
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
            f, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,8), sharex=True)

            keys = ['mean', 'max', 'min', 'std']
            for ax,key in zip([ax1,ax2,ax3,ax4],keys):
                ax.plot(self.stats_final[key], c='black')
                ax.set_ylabel(key, fontsize=12)
        
            ax.set_xlabel("Event", fontsize=12)
            ax1.set_title("Run statistics")
            
            if output is not None:
                f.savefig(output, dpi=300)
    
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


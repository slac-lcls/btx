import numpy as np
import os
import re
import glob
import matplotlib.pyplot as plt
from btx.interfaces.istream import *

class VisualizeSample:

    def __init__(self, base_dir, tag, save_plots=False):
        self.base_dir = base_dir # path to index folder
        self.tag = tag # sample name
        self.save_plots = save_plots # save to disk if True
        if save_plots:
            os.makedirs(os.path.join(self.base_dir, "figs"), exist_ok=True)
        self.cparams, self.stats = self.extract_stats()
        
    def extract_stats(self):
        """
        Extract per-run statistics from the summary and stream files.
        
        Returns
        -------
        cparams : dict
            unit cell distributions for each run
        stats : dict
            number of indexed, hits, etc. for each run
        """
        fstreams = natural_sort(glob.glob(os.path.join(self.base_dir, f"r*{self.tag}.stream")))
        runs = [int(f.split("/")[-1][1:].split("_")[0]) for f in fstreams]

        cparams, stats = {}, {}
        for var in ['run', 'n_events', 'n_hits', 'n_indexed', 'n_multi_lattice']:
            stats[var] = []

        for r in runs:
            # peak finding information
            stats['run'].append(r)
            pf_summary = os.path.join(self.base_dir, f"r{r:04}", "peakfinding.summary")
            if os.path.isfile(pf_summary):
                with open(pf_summary, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Number of events processed" in line:
                            stats['n_events'].append(int(line.split(":")[1]))
                        if "Number of hits found" in line:
                            stats['n_hits'].append(int(line.split(":")[1]))
            else:
                stats['n_events'].append(0)
                stats['n_hits'].append(0)
                
            # stream information
            fstream = os.path.join(self.base_dir, f"r{r:04}_{self.tag}.stream")
            try:
                st = StreamInterface([fstream], cell_only=True)
                stats['n_indexed'].append(st.n_indexed)
                stats['n_multi_lattice'].append(st.n_multiple)
                cparams[r] = st.get_cell_parameters()
            except:
                print(f"{fstream} could not be loaded by StreamInterface")
                stats['n_indexed'].append(0)
                stats['n_multi_lattice'].append(0)
                
        for key in stats.keys():
            stats[key] = np.array(stats[key])
                
        return cparams, stats
    
    def plot_cell_trajectory(self):
        """
        Plot the cell parameter distribution as a function of run number.
        """
        cells = np.vstack(np.array([self.cparams[r] for r in self.cparams.keys()], dtype='object'))
        ncryst = np.array([self.cparams[r].shape[0] for r in self.cparams.keys()], dtype='object').astype(int)
        runval = np.repeat(np.array(list(self.cparams.keys())), ncryst)

        f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(16,8))

        labels = ["$a$", "$b$", "$c$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
        for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
            ax.hexbin(runval, cells[:,i], mincnt=1)
            if i < 3: 
                ax.set_title(labels[i], fontsize=14)
            else: 
                ax.set_title(labels[i], fontsize=14)
            ax.set_xlabel("Run no.", fontsize=14)
            ax.plot([runval.min()-1, runval.max()+1], [np.mean(cells[:,i], axis=0), np.mean(cells[:,i], axis=0)], linestyle='dashed', c='grey')
            ax.set_xlim(runval.min()-1, runval.max()+1)
        ax1.set_ylabel("Angstrom", fontsize=14)
        ax4.set_ylabel("degrees", fontsize=14)

        f.subplots_adjust(hspace=0.4)        
        if self.save_plots:
            f.savefig(os.path.join(self.base_dir, f"figs/cell_trajectory_{self.tag}.png"), dpi=300, bbox_inches='tight')
            
    def plot_stats(self):
        """
        Plot per-run number/rate of hits, indexed, multiple lattice events.
        """
        f, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(16,6))

        order = ['n_events', 'n_hits', 'n_indexed', 'n_multi_lattice']
        titles = ['Hits', 'Indexed', 'Multiple lattice']

        for i,ax in enumerate([ax1,ax2,ax3]):
            ax.bar(self.stats['run'], self.stats[order[i+1]], color='grey')
            avg_rate = np.sum(self.stats[order[i+1]])/np.sum(self.stats[order[i]])
            ax.set_title(f"{titles[i]}\n(total: {np.sum(self.stats[order[i+1]])}, avg: {100*avg_rate:.1f}%)")

        for i,ax in enumerate([ax4,ax5,ax6]):
            ax.bar(self.stats['run'], 100*self.stats[order[i+1]]/self.stats[order[i]], color='grey')
            ax.set_xlabel("Run number", fontsize=14)

        ax1.set_ylabel("Number of events", fontsize=14)
        ax4.set_ylabel("Percentage", fontsize=14)

        if self.save_plots:
            f.savefig(os.path.join(self.base_dir, f"figs/stats_trajectory_{self.tag}.png"), dpi=300, bbox_inches='tight')

def natural_sort(l): 
    """ Apply natural sorting to elements in a list of strings."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

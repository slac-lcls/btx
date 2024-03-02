import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import requests
import glob
from btx.interfaces.ischeduler import *
from btx.interfaces.istream import read_cell_file
from btx.interfaces.imtz import *
from btx.interfaces.ielog import update_summary, elog_report_post

class StreamtoMtz:
    
    """
    Wrapper for calling various CrystFEL functions for generating a mtz
    file from a stream file and reporting statistics. The output files,
    including hkl, mtz, dat, and figures, inherit nomenclature from the
    input stream file.
    """
    
    def __init__(self, input_stream, symmetry, taskdir, cell, ncores=16, queue='milano', tmp_exe=None, mtz_dir=None, anomalous=False,
                 mpi_init=False, slurm_account="lcls", slurm_reservation=""):
        self.stream = input_stream # file of unmerged reflections, str
        self.symmetry = symmetry # point group symmetry, str
        self.taskdir = taskdir # path for storing output, str
        self.cell = cell # pdb or CrystFEL cell file, str
        self.ncores = ncores # int, number of cores for merging
        self.queue = queue # cluster to submit job to
        self.slurm_account = slurm_account
        self.slurm_reservation = slurm_reservation
        self.mtz_dir = mtz_dir # directory to which to transfer mtz
        self.anomalous = anomalous # whether to separate Bijovet pairs
        self._set_up(tmp_exe)

        if mpi_init:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
        else:
            self.comm = None
            self.rank = 0
        
    def _set_up(self, tmp_exe):
        """
        Retrieve number of processors to run partialator and the temporary
        file to which to write the command to sbatch.

        Parameters
        ----------
        tmp_exe : str
            path to job submission file
        """
        # generate directories if they don't already exit
        os.makedirs(self.taskdir, exist_ok=True)
        self.hkl_dir = os.path.join(self.taskdir, "hkl")
        self.fig_dir = os.path.join(self.taskdir, "figs")
        for dirname in [self.hkl_dir, self.fig_dir]:
            os.makedirs(dirname, exist_ok=True)
        if self.mtz_dir is not None:
            os.makedirs(self.mtz_dir, exist_ok=True)

        # make path to executable
        if tmp_exe is None:
            tmp_exe = os.path.join(self.taskdir ,f'merge.sh')
        self.js = JobScheduler(tmp_exe, ncores=self.ncores, jobname=f'merge',
                               queue=self.queue, account=self.slurm_account,
                               reservation=self.slurm_reservation)
        self.js.write_header()

        # retrieve paths
        self.script_path = os.path.abspath(__file__)
        self.prefix = os.path.basename(self.stream).split("stream")[0][:-1]
        self.outhkl = os.path.join(self.hkl_dir, f"{self.prefix}.hkl")

    def cmd_partialator(self, iterations=1, model='unity', min_res=None, push_res=None, max_adu=None):
        """
        Write command to merge reflection data using CrystFEL's partialator.
        https://www.desy.de/~twhite/crystfel/manual-partialator.html.
        By default scaling and post-refinement are performed. This generates
        a .mtz file from a .stream file.
        
        Parameters
        ----------
        iterations : int
            number of cycles of scaling and post refinement
        model : str
            partiality model, unity or xsphere
        min_res : float
            resolution threshold for merging crystals in Angstrom
        push_res : float
            resolution threshold for merging reflections, up to push_res better than crystal's min_res
        max_adu : float
            intensity threshold for excluding saturated peaks
        """        
        command=f"partialator -j {self.ncores} -i {self.stream} -o {self.outhkl} --iterations={iterations} -y {self.symmetry} --model={model}"
        if min_res is not None:
            command += f" --min-res={min_res}"
            if push_res is not None:
                command += f" --push-res={push_res}"
        if max_adu is not None:
            command += f" --max_adu={max_adu}"
        self.js.write_main(f"{command}\n", dependencies=['crystfel', 'xds', 'ccp4'])
            
    def cmd_compare_hkl(self, foms=['CCstar','Rsplit'], nshells=10, highres=None):
        """
        Run compare_hkl on half-sets to compute the dataset's figures of merit.

        Parameters
        ----------
        foms : list of str
            figures of merit to compute
        nshells : int
            number of resolution shells
        highres : float
            high-resolution cut-off
        """
        for fom in foms:
            shell_file = os.path.join(self.hkl_dir, f"{self.prefix}_{fom}_n{nshells}.dat")
            command = f'compare_hkl -y {self.symmetry} --fom {fom} {self.outhkl}1 {self.outhkl}2 -p {self.cell} --shell-file={shell_file} --nshells={nshells}'
            if highres is not None:
                command += f" --highres={highres}"
            self.js.write_main(f"{command}\n")

    def cmd_report(self, foms=['CCstar','Rsplit'], nshells=10):
        """
        Append a line to the executable to report and plot the results. 
        
        Parameters
        ----------
        foms : list of str
            figures of merit to compute
        nshells : int 
            number of resolution shells  
        """
        foms_args = " ".join(map(str,foms))
        command=f"python {self.script_path} -i {self.stream} --symmetry {self.symmetry} --cell {self.cell} --taskdir {self.taskdir} --foms {foms_args} --report --nshells={nshells} --mtz_dir {self.mtz_dir}"
        self.js.write_main(f"{command}\n")
                
    def cmd_hkl_to_mtz(self, space_group=1, highres=None, xds_style=False):
        """
        Convert hkl to mtz format using CrystFEL's get_hkl tool:
        https://www.desy.de/~twhite/crystfel/manual-get_hkl.html
        if anomalous and xds_style are False. Otherwise, convert 
        to mtz using a combination of xdsconv and f2mtz.

        Parameters
        ----------
        space_group : int
            space group number
        highres : float
            high-resolution cut-off in Angstroms
        xds_style : bool
            if True, use xdsconv/f2mtz regardless of anomalous status
        """
        outmtz = os.path.join(self.taskdir, f"{self.prefix}.mtz")
        if self.anomalous or xds_style:
            if space_group == 1:
                print("Warning! Space group will be triclinic")
            cell = read_cell_file(self.cell)
            write_create_xscale(self.outhkl, cell, sg_number=space_group)
            write_xds_inp("xdsformat.hkl",
                          res_range=(100, highres) if highres is not None else None,
                          anomalous=self.anomalous)
            write_anomalous_f2mtz(cell, sg_number=space_group, fname="F2MTZ_ANO.INP")
            self.js.write_main(f"chmod +x create-xscale\n")
            self.js.write_main(f"./create-xscale > xdsformat.hkl\n")
            self.js.write_main(f"xdsconv\n")
            self.js.write_main(f"mv F2MTZ_ANO.INP F2MTZ.INP\n")
            self.js.write_main(f2mtz_command(outmtz))
            self.js.write_main("rm create-xscale XDSCONV.INP XDSCONV.LP F2MTZ.INP xdsformat.hkl temp.hkl temp.mtz \n")
            
        else:
            command = f"get_hkl -i {self.outhkl} -o {outmtz} -p {self.cell} --output-format=mtz"
            if highres is not None:
                command += f" --highres={highres}"
            self.js.write_main(f"{command}\n")        

    def launch(self, wait=True):
        """
        Write an indexing executable for submission to slurm.
        """   
        self.js.clean_up()
        self.js.submit(wait=wait)

    def merge_summary(self,
                      foms: list = ['CCstar', 'Rsplit'],
                      nshells: int = 10) -> dict:
        """! Return a dictionary of key/values to post to the eLog.

        @param (list[str]) foms Figures of merit.
        @param (int) Number of shells used for figures of merit.
        @return (dict) summary_dict Key/values parsed by eLog posting function.
        """
        summary_dict: dict = {};
        for fom in foms:
            for ns in [1, nshells]:
                shell_file = os.path.join(self.hkl_dir, f"{self.prefix}_{fom}_n{int(ns)}.dat")
                if ns != 1:
                    plot_file = os.path.join(self.fig_dir, f"{self.prefix}_{fom}.png")
                    wrangle_shells_dat(shell_file, plot_file)
                else:
                    key, val = wrangle_shells_dat(shell_file)
                    summary_dict[key] = val
        return summary_dict

    def report(self, foms=['CCstar','Rsplit'], nshells=10, update_url=None):
        """
        Summarize results: plot figures of merit and optionally report to elog.
        Transfer the mtz file to a new folder.

        Parameters
        ----------
        foms : list of str
            figures of merit to compute
        nshells : int
            number of shells used to compute figure(s) of merit
        update_url : str
            elog URL for posting progress update
        """
        overall_foms = {}
        for fom in foms:
            for ns in [1, nshells]:
                shell_file = os.path.join(self.hkl_dir, f"{self.prefix}_{fom}_n{int(ns)}.dat")
                if ns != 1:
                    plot_file = os.path.join(self.fig_dir, f"{self.prefix}_{fom}.png")
                    wrangle_shells_dat(shell_file, plot_file)
                else:
                    key, val = wrangle_shells_dat(shell_file)
                    overall_foms[key] = val

        # write summary file
        summary_file = os.path.join(self.taskdir, "merge.summary")
        with open(summary_file, 'w') as f:
            for key in overall_foms.keys():
                f.write(f"Overall {key}: {overall_foms[key]}\n")
                        
        # report to elog
        update_url = os.environ.get('JID_UPDATE_COUNTERS')
        if update_url is not None:
            elog_json = [{"key": f"{fom_name}", "value": f"{stat}"} for fom_name,stat in overall_foms.items()]
            requests.post(update_url, json=elog_json)

        # transfer mtz to new folder
        if self.mtz_dir is not None:
            shutil.copyfile(os.path.join(self.taskdir, f"{self.prefix}.mtz"), 
                            os.path.join(self.mtz_dir, f"{self.prefix}.mtz"))

def wrangle_shells_dat(shells_file, outfile=None):
    """
    Extract the information produced by CrystFEL's compare_hkl. If the .dat
    file was created with nshells==1, return the overall statistics. If the
    .dat file was created with nshells>1, plot the data instead.
    
    Parameters
    ----------
    shells_file : str
        path to a CrystFEL shells.dat file
    outfile : str
        path for saving plot, optional
        
    Returns
    -------
    fom : str
        figure of merit
    stat : float
        overall figure of merit value
    """  
    shells = np.loadtxt(shells_file, skiprows=1)
    with open(shells_file) as f:
        lines = f.read() 
        header = lines.split('\n')[0]
    fom = header.split()[2]
    
    if len(shells.shape)==1:
        if outfile is not None:
            print("The .dat file was generated with only one shell so cannot be plotted.")
        return fom, shells[1]
    
    if len(shells.shape) > 1:
        f, ax1 = plt.subplots(figsize=(6,4))

        ax1.plot(shells[:,0], shells[:,1], c='black')
        ax1.scatter(shells[:,0], shells[:,1], c='black')

        ticks = ax1.get_xticks()
        ax1.set_xticks(ticks) # Only needed to suppress warning
        ax1.set_xticklabels(["{0:0.2f}".format(i) for i in 10/ticks])
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.set_xlabel("Resolution ($\mathrm{\AA}$)", fontsize=14)
        ax1.set_ylabel(f"{fom}", fontsize=14)

        if outfile is not None:
            f.savefig(outfile, dpi=300, bbox_inches='tight')

def get_most_recent_summary(path: str) -> str:
    """Given a root directory return the path of most recent summary.

    @param path (str) Root BTX operating directory.
    @return summary (str) Path to most recent summary
    """
    summaries = glob.glob(f'{path}/summary_*.json')
    run: int = 0
    most_recent = ''
    for summary in summaries:
        filename: str = summary.split('/')[-1]
        current_run = int(filename[-9:-5])
        if current_run > run:
            run = current_run
            most_recent = summary
    return most_recent

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    # global arguments
    parser.add_argument('-i', '--input_stream', required=True, type=str, help='Input stream file')
    parser.add_argument('--symmetry', required=True, type=str, help='Point group symmetry')
    parser.add_argument('--taskdir', required=True, type=str, help='Base directory for storing merging output')
    parser.add_argument('--cell', required=True, type=str, help='File containing unit cell information (.pdb or .cell)')
    parser.add_argument('--anomalous', required=False, type=bool, help='If True, separate Bijovet mates')
    # arguments specific to partialator
    parser.add_argument('--model', required=False, default='unity', choices=['unity','xsphere'], type=str, help='Partiality model')
    parser.add_argument('--iterations', required=False, default=1, type=int, help='Number of cycles of scaling and post-refinement to perform')
    parser.add_argument('--min_res', required=False, type=float, help='Minimum resolution for crystal to be merged')
    parser.add_argument('--push_res', required=False, type=float, help='Maximum resolution beyond min_res for reflection to be merged')
    parser.add_argument('--max_adu', required=False, type=float, help='Intensity cut-off for excluding saturated peaks')
    # arguments for computing figures of merit, converting to mtz
    parser.add_argument('--foms', required=False, default=['CCstar', 'Rsplit'], type=str, nargs='+', help='Figures of merit to calculate')
    parser.add_argument('--nshells', required=False, type=int, default=10, help='Number of resolution shells for computing figures of merit')
    parser.add_argument('--highres', required=False, type=float, help='High resolution limit for computing figures of merit') 
    parser.add_argument('--space_group', required=False, type=int, default=1, help='Space group number, only required for anomalous data')
    # arguments for reporting
    parser.add_argument('--report', help='Report indexing results to summary file and elog', action='store_true')
    parser.add_argument('--update_url', help='URL for communicating with elog', required=False, type=str)
    parser.add_argument('--mtz_dir', help='Folder to transfer mtz to', required=False, type=str)
    # slurm arguments
    parser.add_argument('--ncores', help='Number of cores for parallelizing scaling', required=False, type=int, default=16)
    parser.add_argument('--queue', help='Submission queue', required=False, type=str, default='ffbh3q')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    params = parse_input()
    stream_path = params.input_stream
    
    stream_to_mtz = StreamtoMtz(stream_path, 
                                params.symmetry, 
                                params.taskdir, 
                                params.cell, 
                                ncores=params.ncores, 
                                queue=params.queue, 
                                mtz_dir=params.mtz_dir, 
                                anomalous=params.anomalous,
                                mpi_init=True)
    if not params.report:
        stream_to_mtz.cmd_partialator(iterations=params.iterations, 
                                      model=params.model, 
                                      min_res=params.min_res, 
                                      push_res=params.push_res, 
                                      max_adu=params.max_adu)
        for ns in [1, params.nshells]:
            stream_to_mtz.cmd_compare_hkl(foms=params.foms, 
                                          nshells=ns, 
                                          highres=params.highres)
        stream_to_mtz.cmd_report(foms=params.foms, 
                                 nshells=params.nshells)
        stream_to_mtz.cmd_hkl_to_mtz(highres=params.highres, 
                                     space_group=params.space_group)
        stream_to_mtz.launch()
    else:
        if stream_to_mtz.rank == 0:
            indexdir: str = stream_path[:-len(stream_path.split('/')[-1])]
            rootdir: str = indexdir[:-7]
            summary_file = get_most_recent_summary(rootdir)
            summary_dict: dict = stream_to_mtz.merge_summary(
                foms=params.foms,
                nshells=params.nshells
            )
            update_summary(summary_file, summary_dict)
            elog_report_post(summary_file)

            if stream_to_mtz.mtz_dir:
                shutil.copyfile(
                    os.path.join(
                        stream_to_mtz.taskdir,
                        f'{stream_to_mtz.prefix}.mtz'
                    ),
                    os.path.join(
                        stream_to_mtz.mtz_dir,
                        f'{stream_to_mtz.prefix}.mtz'
                    )
                )

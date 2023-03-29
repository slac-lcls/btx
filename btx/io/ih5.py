"""!
@brief Utilities for working with HDF5 files and managing concurrent write operations.

Classes:
SmallDataReader : Interface to HDF5 files written by Small Data.

Functions:
lock_file(path, timeout) : Decorator to prevent concurrent write operations.
"""

import os
import numpy as np
import signal
import time
import h5py
import tables
import socket
import matplotlib.pyplot as plt
from typing import Union, Optional, Callable, Any

def lock_file(path:str, timeout: int = 5, wait: float = 0.5) -> Callable:
    """! Decorator mainly for write functions. Will not allow the decorated
    function to execute if an associated '.lock' file exists. If no lock file
    exists one will be created, preventing other decorated functions from
    executing while the current function completes its operations. Attempts at
    function execution will also be abandoned after a timeout threshold is
    reached.

    Usage e.g.:
    @lock_file('shared.h5', timeout=2)
    def my_write_function():
        print('Writing file safely')
    my_write_function() # Create shared.lock if it doesn't exist

    @param path (str) Path including filename of the file to be locked.
    @param timeout (int) Timeout time in seconds to abandon execution attempt.
    @param wait (float) Time in seconds to wait between write attempts.
    """
    def decorator_lock(write_func: Callable) -> Callable:
        """! Inner decorator for file locking.

        @param write_func (function) The function to run while file is locked.
        """
        def wrapper(*args, **kwargs):
            lockfile = path.split('.')[0] + '.lock'
            written = False
            sendMsg = True
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            try:
                while not written:
                    if not os.path.exists(lockfile):
                        with open(lockfile, 'w') as lock:
                            print(f'Locking file {path}.')
                        print(f'Writing file {path}.')
                        write_func()
                        os.remove(lockfile)
                        written = True
                        sendMsg = False
                        print('Unlocking file.')
                    else:
                        if sendMsg:
                            print(f'File {path} is locked.')
                            sendMsg = False
                        time.sleep(wait)
            except TimeoutError as err:
                print(err)
            signal.alarm(0)
        return wrapper
    return decorator_lock


def _timeout_handler(signum, frame):
    raise TimeoutError()


class TimeoutError(Exception):
    """! Error raised if timeout is reached."""

    def __init__(self, msg='Timeout reached'):
        """! Exception initializer."""
        super().__init__(msg)


class SmallDataReader:
    """! Class interface for reading SmallData hdf5 files."""

    def __init__(self, expmt: str,
                 run: Union[str, int],
                 savedir: str,
                 h5path: Optional[str] = None):
        """! SmallDataReader initializer.

        @param expmt (str) : Experiment name.
        @param run (str | int) : Run number or range of hyphen-separated runs.
        @param savedir (str) : Path to write any output files to.
        @param h5path (str | None) : Path to an hdf5 file. Will search if None.
        """
        if h5path:
            self._path = h5path
        else:
            if 'drp' in socket.gethostname():
                self._path = (f"/cds/data/drpsrcf/{expmt[:3]}/{expmt}/scratch/"
                              f"hdf5/smalldata/{expmt}_Run{run:04d}")
            else:
                self._path = (f"/cds/data/psdm/{expmt[:3]}/{expmt}/"
                              f"hdf5/smalldata/{expmt}_Run{run:04d}")

        self.expmt = expmt
        self.run = run
        self.savedir = savedir

        try:
            self.root = tables.File(self._path).root
            self.h5 = h5py.File(self._path)
        except FileNotFoundError:
            print("No small data available for this run.")

    def image_sums(self):
        """! Plots the sums of the images over a run for the used detectors.

        The image sums (powder patterns for crystallography) can be used to
        verify that data was collected during a run.
        """
        if 'Sums' in self.h5:
            imgList = [self.h5[f'Sums/{img}']
                       for img in self.h5['Sums'].keys()]

            for img in imgList:
                squeezed = img[:].squeeze()
                if len(squeezed.shape) > 2:
                    self.construct_img_panels(img)
                else:
                    lims = (np.nanpercentile(squeezed, 5),
                            np.nanpercentile(squeezed, 95))
                    title = img.name.split('/')[2]
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=200)
                    ax.imshow(squeezed, clim=lims)
                    ax.set_title(title)
                    fig.tight_layout()
                    fig.savefig(f'{self.savedir}/{title}_Sum_r{self.run:04d}')
        else:
            print('Image sums are not available.')


    def construct_img_panels(self):
        """! Assemble multipanel detector images."""
        pass

    def timetool_histograms(self):
        """! Plot histograms of timetool output.

        Plots the histograms of: the timetool correction, the convolution
        amplitudes, and the full-width half-maximums (FWHMs) of the
        convolutions.
        """
        if 'tt' in self.h5:
            fltpos_ps = self.h5['tt/FLTPOS_PS'][:]
            ampl = self.h5['tt/AMPL'][:]
            fwhm = self.h5['tt/FLTPOSFWHM'][:]

            fig, axs = plt.subplots(1, 3, figsize=(6, 4), dpi=200)
            axs[0].hist(fltpos_ps, bins=30, density=True)
            axs[0].set_title('Timetool Correction')
            axs[0].set_xlabel('Correction (ps)')
            axs[0].set_ylabel('Density')
            axs[1].hist(ampl, bins=30, density=True)
            axs[1].set_title('Convolution\nAmplitude')
            axs[1].set_xlabel('Amplitude (a.u.)')
            axs[2].hist(fwhm, bins=30, density=True)
            axs[2].set_title('Convolution FWHM')
            axs[2].set_xlabel('FWHM (a.u.)')
            fig.tight_layout()
            fig.savefig(f'{self.savedir}/ttHistograms_r{self.run:04d}.png')
        else:
            print('No timetool data available.')

    def timetool_amplitude_correlation(self, xrayinode: Optional[str] = None):
        """! Plot the 2D histogram of x-ray intensity and convolution amplitude.

        The timetool convolution amplitude should be correlated with the x-ray
        intensity. The 2D histogram of these events provides a visual metric
        that the timetool results are reasonable. The correlations hould appear
        to be ~linear.

        @param xrayinode (str) : Specific node with X-ray intensity recordings.
        """
        try:
            ampl = self.h5['tt/AMPL'][:]

            if xrayinode:
                xray_i = self.h5[f'{xrayinode}/sum'][:]
            elif (xrayinodes := self.find_xray_intensity_nodes()):
                xray_i = self.h5[f'{xrayinodes[0]}/sum'][:]
            else:
                xray_i = []

            if len(xray_i) > 0: # Test length explicitly when ndarray (ie not list)
                fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
                ax.hexbin(xray_i, ampl, mincnt=1)
                ax.set_xlabel('X-ray Intensity')
                ax.set_ylabel('Timetool Convolution Amplitude')
                fig.tight_layout()
                fig.savefig(f'{self.savedir}/ttAmplXrayICorr_r{self.run:04d}.png')
        except KeyError as err:
            print(f'Node not found: {str(err).split()[6]}')

    def find_xray_intensity_nodes(self) -> list:
        """! Determines the available readouts of X-ray intensity."""
        keys = [key for key in self.h5.keys()
                if 'ipm' in key and 'sum' in self.h5[key].keys()]
        # keys is an empty list if no nodes found
        return keys

    def print_all_nodes(self):
        """! Print all groups and datasets in the HDF5 file."""
        self.traverse_nodes(self.h5, level=1)

    def traverse_nodes(self, node, level: int = 1):
        """! Recurses through an HDF5 file hierarchy printing nodes."""
        if type(node) == h5py._hl.dataset.Dataset:
            return
        else:
            for key in node.keys():
                print('\t'*(level - 1) + f'{key}\n')
                self.traverse_nodes(node[key], level=level + 1)

    def timestamped_data_totext(self, node_list: list):
        """! Write a text file of unique event identifiers and h5 node values.

        This can be used to write out e.g. specific evr codes per event. Data
        is written in a space separated format, with one event per line. The
        first column contains event identifiers, followed by the values for
        each node in subsequent columns. E.g.:
        unique-identifier node1        node2        ...
                id_event1 event1-node1 event1-node2
                id_event2 event2-node1 event2-node2
                   ...

        Unique identifiers are of the form: `event_time-fiducial`

        @param node_list (list[str])
        """
        pass


    def unique_stamps(self):
        """! Combine timestamps and fiducials for unique event identifiers."""
        pass

    def __getitem__(self, key: str) -> Any:
        """! Overload subscripting operator, [], to access hdf5 file.

        @param key (str) : Node to access.
        @return value (Any) : The value living at the specified key in the hdf5.
        """
        try:
            return self.h5[key]
        except KeyError:
            print('Incorrect node.')

    def __setitem__(self, key, value):
        """! Provided only to prevent accidental errors."""
        print('Writing to hdf5 prohibited.')

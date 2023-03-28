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

def lock_file(path:str, timeout: int = 5, wait: float = 0.5) -> callable:
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
    def decorator_lock(write_func: callable) -> callable:
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
                 run: str | int,
                 savedir: str,
                 h5path: str | None = None):
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

        self.root = tables.File(self._path).root
        self.h5 = h5py.File(self._path)

    def timetool_histograms(self):
        """! Pass."""
        fltpos_ps = self.h5['tt/FLTPOS_PS'][:]
        # fltpos = self.h5['tt/FLTPOS'][:]
        ampl = self.h5['tt/AMPL'][:]
        fwhm = self.h5['tt/FLTPOSFWHM'][:]

        fig, axs = plt.subplots(1, 3, figsize=(6, 4), dpi=200)
        # axs[0].hist(fltpos, bins=30)
        axs[0].hist(fltpos_ps, bins=30, density=True)
        axs[0].set_title('Timetool Correction')
        axs[0].set_xlabel('Correction (ps)')
        axs[0].set_ylabel('Density')
        # axs[0].hist(fltpos)
        axs[1].hist(ampl, bins=30, density=True)
        axs[1].set_title('Convolution\nAmplitude')
        axs[1].set_xlabel('Amplitude (a.u.)')
        axs[2].hist(fwhm, bins=30, density=True)
        axs[2].set_title('Convolution FWHM')
        axs[2].set_xlabel('FWHM (a.u.)')
        fig.tight_layout()
        fig.savefig(f'{self.savedir}/tt_histograms_r{self.run:04d}.png')

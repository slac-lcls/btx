"""!
@brief Utilities for working with HDF5 files and managing concurrent write operations.

Functions:
lock_file(path, timeout) : Decorator to prevent concurrent write operations.
"""

import os, signal
import h5py

def lock_file(path:str, timeout: int = 10) -> callable:
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
    @param timeout (int) The timeout time in seconds to abandon execution attempt.
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
                    if sendMsg:
                        print(f'File {path} is locked.')
                        sendMsg = False
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
        super().__init__(msg)

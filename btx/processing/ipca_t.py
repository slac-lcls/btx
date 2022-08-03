import csv
import numpy as np
import scipy as sp
from mpi4py import MPI

from time import perf_counter

class TaskTimer:
    """
    A context manager to record the duration of managed tasks.

    Attributes
    ----------
    start_time : float
        reference time for start time of task
    intervals : list
        list containing time interval data
    """
    
    def __init__(self, intervals):
        """
        Construct all necessary attributes for the TaskTimer context manager.

        Parameters
        ----------
        intervals : list of float
            List containing interval data
        """
        self.start_time = 0.
        self.intervals = intervals
    
    def __enter__(self):
        """
        Set reference start time.
        """
        self.start_time = perf_counter()
    
    def __exit__(self, *args, **kwargs):
        """
        Mutate interval list with interval duration of current task.
        """
        self.intervals.append(perf_counter() - self.start_time)

class IPCAT:

    def __init__(self, d, q):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.d = d
        self.q = q
        self.n = 0

        self.S = np.eye(self.q)
        self.U = np.zeros((self.q, self.d))
        self.mu = np.zeros(self.d)
        self.total_variance = np.zeros(self.d)

        # self.get_distributed_indices()

        # attributes for computing timings of iPCA steps
        self.task_durations = dict({})
        self.task_durations['concat'] = []
        self.task_durations['ortho'] = []
        self.task_durations['build_r'] = []
        self.task_durations['svd'] = []
        self.task_durations['update_basis'] = []
        self.task_durations['MPI1'] = []
        self.task_durations['MPI2'] = []
        self.task_durations['MPI3'] = []
        self.task_durations['total_update'] = []


    def update_model(self, X):
        """
        Update model with new block of observations using iPCA.

        Parameters
        ----------
        X : ndarray, shape (d x m)
            block of m (d x 1) observations 
        """
        m, _ = X.shape
        n = self.n
        q = self.q
        d = self.d

        with TaskTimer(self.task_durations['total_update']):

            if self.rank == 0:
                mu_m, s_m = calculate_sample_mean_and_variance(X)

                with TaskTimer(self.task_durations['concat']):
                    X_centered = X - mu_m
                    X_m = np.vstack((X_centered, np.sqrt(n * m / (n + m)) * (mu_m - self.mu)))

                with TaskTimer(self.task_durations['ortho']):
                    UX_m = X_m @ self.U.T
                    dX_m = X_m - UX_m @ self.U

                    # need to distribute this step amongst ranks
                    _, X_pm = sp.linalg.rq(dX_m, mode='economic')
            
                with TaskTimer(self.task_durations['build_r']):
                    R = np.block([[self.S, np.zeros((q, m + 1))], [UX_m, dX_m @ X_pm.T]])
            
                with TaskTimer(self.task_durations['svd']):
                    _, S_tilde, U_tilde = np.linalg.svd(R)

                    concat = np.vstack((self.U, X_pm))
                    concat = np.array_split(concat, indices_or_sections=self.size, axis=1)
            else:
                concat = None
                U_tilde = None

            with TaskTimer(self.task_durations['MPI1']):
                if self.size == 1:
                    concat = concat[0]
                else:
                    concat = self.comm.scatter(concat, root=0)
                    U_tilde = self.comm.bcast(U_tilde, root=0)
            
            with TaskTimer(self.task_durations['update_basis']):
                U_prime = U_tilde @ concat

            with TaskTimer(self.task_durations['MPI2']):
                if self.size > 1:
                    U_prime = self.comm.gather(U_prime, root=0)

                if self.rank == 0:
                    if self.size > 1:
                        U_prime = np.hstack(U_prime)

                    self.U = U_prime[:q]
                    self.S = np.diag(S_tilde[:q])

                    self.total_variance = update_sample_variance(self.total_variance, s_m, self.mu, mu_m, n, m)
                    self.mu = update_sample_mean(self.mu, mu_m, n, m)

                    self.n += m

            with TaskTimer(self.task_durations['MPI3']):
                if self.size > 1:
                    self.U = self.comm.bcast(self.U, root=0)
                    self.S = self.comm.bcast(self.S, root=0)
        

    def initialize_model(self, X):
        """
        Initialiize model on sample of data using batch PCA.

        Parameters
        ----------
        X : ndarray, shape (d x m)
            set of m (d x 1) observations
        """
        q = self.q

        self.mu, self.total_variance = calculate_sample_mean_and_variance(X)

        _, s, self.U = np.linalg.svd(X - self.mu, full_matrices=False)
        self.S = np.diag(s)

        self.n += q

    def report_interval_data(self):
        """
        Report time interval data gathered during iPCA.
        """
        if self.rank == 0:
            if len(self.task_durations):
                for key in list(self.task_durations.keys()):
                    interval_mean = np.mean(self.task_durations[key])

                    print(f'Mean per-block compute time of step \'{key}\': {interval_mean:.4g}s')


    def save_interval_data(self, dir_path=None):
        """
        Save time interval data gathered during iPCA to file.

        Parameters
        ----------
        dir_path : str
            Path to output directory.
        """
        if self.rank == 0:
            
            if dir_path is None:
                print('Failed to specify output directory.')
                return

            file_name = 'task_' + str(self.q) + str(self.d) + str(self.n) + str(self.size) + '.csv'

            with open(dir_path + file_name, 'x', newline='', encoding='utf-8') as f:

                if len(self.task_durations):
                    writer = csv.writer(f)

                    writer.writerow(['q', self.q])
                    writer.writerow(['d', self.d])
                    writer.writerow(['n', self.n])
                    writer.writerow(['ranks', self.size])

                    keys = list(self.task_durations.keys())
                    values = list(self.task_durations.values())
                    values_transposed = np.array(values).T

                    writer.writerow(keys)
                    writer.writerows(values_transposed)


def calculate_sample_mean_and_variance(imgs):
    """
    Compute the sample mean and variance of a flattened stack of n images.

    Parameters
    ----------
    imgs : ndarray, shape (d x n)
        horizonally stacked batch of flattened images 

    Returns
    -------
    mu_m : ndarray, shape (d x 1)
        mean of imgs
    su_m : ndarray, shape (d x 1)
        sample variance of imgs (1 dof)
    """
    m, d = imgs.shape

    mu_m = np.mean(imgs, axis=0)
    s_m  = np.zeros(d)

    if m > 1:
        s_m = np.var(imgs, ddof=1, axis=0)
    
    return mu_m, s_m

def update_sample_mean(mu_n, mu_m, n, m):
    """
    Combine combined mean of two blocks of data.

    Parameters
    ----------
    mu_n : ndarray, shape (d x 1)
        mean of first block of data
    mu_m : ndarray, shape (d x 1)
        mean of second block of data
    n : int
        number of observations in first block of data
    m : int
        number of observations in second block of data

    Returns
    -------
    mu_nm : ndarray, shape (d x 1)
        combined mean of both blocks of input data
    """
    mu_nm = mu_m

    if n != 0:
        mu_nm = (1 / (n + m)) * (n * mu_n + m * mu_m)

    return mu_nm

def update_sample_variance(s_n, s_m, mu_n, mu_m, n, m):
    """
    Compute combined sample variance of two blocks of data described by input parameters.

    Parameters
    ----------
    s_n : ndarray, shape (d x 1)
        sample variance of first block of data
    s_m : ndarray, shape (d x 1)
        sample variance of second block of data
    mu_n : ndarray, shape (d x 1)
        mean of first block of data
    mu_m : ndarray, shape (d x 1)
        mean of second block of data
    n : int
        number of observations in first block of data
    m : int
        number of observations in second block of data

    Returns
    -------
    s_nm : ndarray, shape (d x 1)
        combined sample variance of both blocks of data described by input parameters
    """
    s_nm = s_m

    if n != 0:
        s_nm = (((n - 1) * s_n + (m - 1) * s_m) + (n*m*(mu_n - mu_m)**2) / (n + m)) / (n + m - 1) 

    return s_nm



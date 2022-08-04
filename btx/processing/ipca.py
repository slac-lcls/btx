import csv
from xml.sax.handler import DTDHandler
import numpy as np
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

class IPCA:

    def __init__(self, d, q, m):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.d = d
        self.q = q
        self.m = m
        self.n = 0

        self.distribute_indices()

        self.S = np.eye(self.q)
        self.U = np.zeros((self.end_index - self.start_index, self.q))

        if self.rank == 0:
            self.mu = np.zeros((self.d, 1))
            self.total_variance = np.zeros((self.d, 1))

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
        self.task_durations['ortho_prep'] = []

    def distribute_indices(self):

        d = self.d
        size = self.size
        rank = self.rank

        # determine boundary indices between ranks
        split_indices = np.zeros(size)
        for r in range(size):
            num_per_rank = d // size
            if r < (d % size):
                num_per_rank += 1
            split_indices[r] = num_per_rank
        split_indices = np.append(np.array([0]), np.cumsum(split_indices)).astype(int)   
        
        # update self variables that determine start and end of this rank's batch
        self.start_index = split_indices[rank]
        self.end_index = split_indices[rank+1]
    
    def parallel_qr(self, A):

        q_loc, r_loc = np.linalg.qr(A, mode='reduced')
        r_tot = self.comm.gather(r_loc, root=0)

        if self.rank == 0:
            r_tot = np.concatenate(r_tot, axis=0)
            q_tot, r_tilde = np.linalg.qr(r_tot, mode='reduced')
        else:
            q_tot, r_tilde = None, None
        
        q_tot = self.comm.bcast(q_tot, root=0)
        r_tilde = self.comm.bcast(r_tilde, root=0)

        q_fin = q_loc @ q_tot

        print(q_fin.shape)
        print(r_tilde.shape)

        return q_fin, r_tilde

    def get_model(self):
        
        U_tot = self.gather(self.U, root=0)
        S_tot = self.S

        return U_tot, S_tot

    def update_model(self, X):
        """
        Update model with new block of observations using iPCA.

        Parameters
        ----------
        X : ndarray, shape (d x m)
            block of m (d x 1) observations 
        """
        _, m = X.shape
        n = self.n
        q = self.q
        d = self.d

        with TaskTimer(self.task_durations['total_update']):

            if self.rank == 0:
                mu_m, s_m = calculate_sample_mean_and_variance(X)
                
                X_centered = X - np.tile(mu_m, m)
                v_augment = np.sqrt(n * m / (n + m)) * (mu_m - self.mu)

                self.total_variance = update_sample_variance(self.total_variance, s_m, self.mu, mu_m, n, m)
                self.mu = update_sample_mean(self.mu, mu_m, n, m)
            else:
                X_centered = None
                v_augment = None
            
            X_centered = self.comm.bcast(X_centered, root=0)
            v_augment = self.comm.bcast(v_augment, root=0)

            us = self.U @ self.S

            qr_input = np.hstack((us, X_centered[self.start_index:self.end_index, :], v_augment[self.start_index:self.end_index, :]))
            UB_tilde, R = self.parallel_qr(qr_input)

            # parallel SVD of R
            if self.rank == 0:
                U_tilde, S_tilde, _ = np.linalg.svd(R)
            else:
                U_tilde, S_tilde, _ = None, None, None

            print(UB_tilde.shape)
            print(U_tilde.shape)
            
            U_tilde = self.comm.bcast(U_tilde, root=0)
            print(U_tilde.shape)
            U_prime = UB_tilde @ U_tilde

            self.U = U_prime[:, :q]
            self.S = S_tilde[:q]

            self.n += m
            
            # perform self.U @ self.S in parallel, divide X_m in parallel, concat and serve as inputs to parallel_qr
            # can just store self.U amd self.S at the end of each local run 

            # or just compute massive QR factorization here, parallelized. Less communication over the nextwork? (need to benchmark)
            # parallelized QR yields an already distributed version of [U B_tilde], so less network operations


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

        centered_data = X - np.tile(self.mu, q)
        self.U, s, _ = np.linalg.svd(centered_data, full_matrices=False)
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
                    writer.writerow(['m', self.m])

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
    d, m = imgs.shape

    mu_m = np.reshape(np.mean(imgs, axis=1), (d, 1))
    s_m  = np.zeros((d, 1))

    if m > 1:
        s_m = np.reshape(np.var(imgs, axis=1, ddof=1), (d, 1))
    
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



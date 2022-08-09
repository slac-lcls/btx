from curses import A_BOLD
import os
import csv

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
    task_durations : dict
        Dictionary containing iinterval data and their corresponding tasks
    task_description : str
        description of current task
    """
    
    def __init__(self, task_durations, task_description):
        """
        Construct all necessary attributes for the TaskTimer context manager.

        Parameters
        ----------
        task_durations : dict
            Dictionary containing iinterval data and their corresponding tasks
        task_description : str
            description of current task
        """
        self.start_time = 0.
        self.task_durations = task_durations
        self.task_description = task_description
    
    def __enter__(self):
        """
        Set reference start time.
        """
        self.start_time = perf_counter()
    
    def __exit__(self, *args, **kwargs):
        """
        Mutate duration dict with time interval of current task.
        """
        time_interval = perf_counter() - self.start_time

        if self.task_description not in self.task_durations:
            self.task_durations[self.task_description] = []
        self.task_durations[self.task_description].append(time_interval)

class IPCA:

    def __init__(self, d, q, m, split_indices):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.d = d
        self.q = q
        self.m = m
        self.n = 0

        # compute number of counts in and start indices over ranks
        self.split_counts, self.start_indices = self.distribute_indices(split_indices)

        # attribute for storing interval data
        self.task_durations = dict({})

        self.S = np.ones(self.q)
        self.U = np.zeros((self.split_counts[self.rank], self.q))
        self.mu = np.zeros((self.split_counts[self.rank], 1))
        self.total_variance = np.zeros((self.split_counts[self.rank], 1))


    def distribute_indices(self, split_indices):
        size = self.size

        start_indices = split_indices[:-1]
        split_counts = np.empty(size, dtype=int)

        for i in range(len(split_indices) - 1):
            split_counts[i] = split_indices[i+1] - split_indices[i]
        
        return split_counts, start_indices

    
    def parallel_qr(self, A):
        d = self.d
        q = self.q
        m = self.m
        y, x = A.shape

        with TaskTimer(self.task_durations, 'qr - local qr'):
            q_loc, r_loc = np.linalg.qr(A)

        with TaskTimer(self.task_durations, 'qr - r_tot gather'):
            if self.rank == 0:
                r_tot = np.empty((self.size*(q+m+1), q+m+1))
            else: 
                r_tot = None

            self.comm.Gather(r_loc, r_tot, root=0)

        if self.rank == 0:
            with TaskTimer(self.task_durations, 'qr - global qr'):
                q_tot, r_tilde = np.linalg.qr(r_tot)
        else:
            q_tot = np.empty((self.size*(q+m+1), q+m+1))
            r_tilde = np.empty((q+m+1, q+m+1))
    
        with TaskTimer(self.task_durations, 'qr - bcast q_tot'):
            self.comm.Bcast(q_tot, root=0)
        
        with TaskTimer(self.task_durations, 'qr - bcast r_tilde'):
            self.comm.Bcast(r_tilde, root=0)

        with TaskTimer(self.task_durations, 'qr - local matrix build'):
            q_fin = q_loc @ q_tot[self.rank*x:(self.rank+1)*x, :]

        return q_fin, r_tilde

    def get_model(self):
        """
        Notes
        -----
        Intended to be called from the root process.
        """
        if self.rank == 0:
            U_tot = np.empty((self.d, self.q))
            mu_tot = np.empty((self.d, 1))
            var_tot = np.empty((self.d, 1))
            S_tot = self.S
        else:
            U_tot, mu_tot, var_tot, S_tot = None, None, None, None

        self.comm.Gatherv(self.U, [U_tot, self.split_counts*self.q, self.start_indices, MPI.DOUBLE], root=0)
        self.comm.Gatherv(self.mu, [mu_tot, self.split_counts*self.q, self.start_indices, MPI.DOUBLE], root=0)
        self.comm.Gatherv(self.total_variance, [var_tot, self.split_counts*self.q, self.start_indices, MPI.DOUBLE], root=0)

        S_tot = self.S

        return U_tot, S_tot, mu_tot, var_tot
    

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

        with TaskTimer(self.task_durations, 'total update'):

            with TaskTimer(self.task_durations, 'update mean and variance'):
                mu_n = self.mu
                mu_m, s_m = calculate_sample_mean_and_variance(X)
                self.total_variance = update_sample_variance(self.total_variance, s_m, mu_n, mu_m, n, m)
                self.mu = update_sample_mean(mu_n, mu_m, n, m)

            with TaskTimer(self.task_durations, 'center data and compute augment vector'):
                X_centered = X - np.tile(mu_m, m)
                v_augment = np.sqrt(n * m / (n + m)) * (mu_m - mu_n)

                X_aug = np.hstack((X_centered, v_augment))

            # with TaskTimer(self.task_durations, 'scatter aug data'):
            #     X_aug_loc = np.empty((self.split_counts[self.rank], m+1))
            #     self.comm.Scatterv([X_aug, self.split_counts, self.start_indices, MPI.DOUBLE], X_aug_loc, root=0)

            with TaskTimer(self.task_durations, 'first matrix product U@S'):
                us = self.U @ np.diag(self.S)

            print(self.rank, us.shape, self.U.shape, self.S.shape, X_aug.shape)

            with TaskTimer(self.task_durations, 'QR concatenate'):
                qr_input = np.hstack((us, X_aug))
            
            with TaskTimer(self.task_durations, 'parallel QR'):
                UB_tilde, R = self.parallel_qr(qr_input)

            with TaskTimer(self.task_durations, 'SVD of R'):
                # parallelize in the future?
                if self.rank == 0:
                    U_tilde, S_tilde, _ = np.linalg.svd(R)
                else:
                    U_tilde = np.empty((q+m+1, q+m+1))
                    S_tilde = np.empty(q+m+1)

            with TaskTimer(self.task_durations, 'broadcast U tilde'):
                self.comm.Bcast(U_tilde, root=0)
            
            with TaskTimer(self.task_durations, 'broadcast S_tilde'):
                self.comm.Bcast(S_tilde, root=0)

            with TaskTimer(self.task_durations, 'compute local U_prime'):
                U_prime = UB_tilde @ U_tilde[:, :q]

            self.U = U_prime
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
        self.U, self.S, _ = np.linalg.svd(centered_data, full_matrices=False)

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

            with open(os.path.join(dir_path, file_name), 'x', newline='', encoding='utf-8') as f:

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

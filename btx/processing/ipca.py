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
        _, x = A.shape

        q = self.q
        m = x-q-1

        with TaskTimer(self.task_durations, 'qr - local qr'):
            q_loc, r_loc = np.linalg.qr(A, mode='reduced')
        
        self.comm.Barrier()

        with TaskTimer(self.task_durations, 'qr - r_tot gather'):
            if self.rank == 0:
                r_tot = np.empty((self.size*(q+m+1), q+m+1))
            else: 
                r_tot = None

            self.comm.Gather(r_loc, r_tot, root=0)

        if self.rank == 0:
            with TaskTimer(self.task_durations, 'qr - global qr'):
                q_tot, r_tilde = np.linalg.qr(r_tot, mode='reduced')
                
            with TaskTimer(self.task_durations, 'qr - global svd'):
                U_tilde, S_tilde, _ = np.linalg.svd(r_tilde)
        else:
            U_tilde = np.empty((q+m+1, q+m+1))
            S_tilde = np.empty(q+m+1)
            q_tot = None
        
        self.comm.Barrier()
    
        with TaskTimer(self.task_durations, 'qr - scatter q_tot'):
            q_tot_loc = np.empty((q+m+1, q+m+1))
            self.comm.Scatter(q_tot, q_tot_loc, root=0)
        
        with TaskTimer(self.task_durations, 'qr - local matrix build'):
            q_fin = q_loc @ q_tot_loc
        
        self.comm.Barrier()
        
        with TaskTimer(self.task_durations, 'qr - bcast S_tilde'):
            self.comm.Bcast(S_tilde, root=0)
        
        self.comm.Barrier()

        with TaskTimer(self.task_durations, 'qr - bcast U_tilde'):
            self.comm.Bcast(U_tilde, root=0)

        return q_fin, U_tilde, S_tilde

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
        
        axes_split = self.comm.gather(self.U, root=0)

        if self.rank == 0:
            U_tot = axes_split[0]

            for i in range(1, self.size):
                U_tot = np.concatenate((U_tot, axes_split[i]), axis=0)

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

        print('Rank {r}, factoring {m} sample{s} into {n} sample model...'.format(r=self.rank, m=m, s='s' if m > 1 else '', n=n))

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

            with TaskTimer(self.task_durations, 'first matrix product U@S'):
                us = self.U @ np.diag(self.S)

            with TaskTimer(self.task_durations, 'QR concatenate'):
                qr_input = np.hstack((us, X_aug))
            
            with TaskTimer(self.task_durations, 'parallel QR'):
                UB_tilde, U_tilde, S_tilde = self.parallel_qr(qr_input)

            # concatenating first preserves the memory contiguity of U_prime and thus self.U
            with TaskTimer(self.task_durations, 'compute local U_prime'):
                U_prime = UB_tilde @ U_tilde[:, :q]

            self.U = U_prime
            self.S = S_tilde[:q]

            self.n += m


    def initialize_model(self, X):
        """
        Initialiize model on sample of data using batch PCA.

        Parameters
        ----------
        X : ndarray, shape (d x m)
            set of m (d x 1) observations
        """
        print(f'Rank {self.rank}, initializing model with {self.q} samples...')

        self.mu, self.total_variance = calculate_sample_mean_and_variance(X)

        centered_data = X - np.tile(self.mu, self.q)
        self.U, self.S, _ = np.linalg.svd(centered_data, full_matrices=False)

        self.n += self.q

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

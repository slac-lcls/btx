import numpy as np
import argparse
import h5py
import os
import requests
from mpi4py import MPI
from btx.interfaces.psana_interface import *
from psalgos.pypsalgos import PyAlgos

from matplotlib import pyplot as plt
from time import perf_counter

class TaskTimer:
    """
    A context manager to record the duration of managed task.

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

class FeatureExtractor:
    """
    Extract features from a psana run using dimensionality reduction.
    """
    
    def __init__(self, exp, run, det_type, q=50, block_size=10, num_images=100, init_with_pca=False):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)

        self.ipca_intervals = dict({})
        self.reduced_indices = np.array([])

        self.q = q
        self.block_size = block_size
        self.num_images = num_images
        self.init_with_pca = init_with_pca
 
        
    def set_ipca_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if self.key:
                self.key = value
        

    def generate_reduced_indices(self, new_dim):
        """
        """

        z, x, y = self.psi.det.shape()
        det_pixels = z * x * y

        if det_pixels < new_dim:
            print('Detector dimension must be greater than or equal to reduced dimension.')
            return

        self.reduced_indices = np.random.choice(det_pixels, new_dim, replace=False)

    def remove_reduced_indices(self):
        self.reduced_indices = np.array([])

    def get_distributed_indices(self, d):
        split_indices = np.zeros(self.size)
        for r in range(self.size):
            num_per_rank = d // self.size
            if r < (d % self.size):
                num_per_rank += 1
            split_indices[r] = num_per_rank
        split_indices = np.append(np.array([0]), np.cumsum(split_indices)).astype(int)

        self.start_index = split_indices[self.rank]
        self.end_index = split_indices[self.rank + 1]

    def ipca(self):
        """
        Run iPCA with run subset subject to initialization parameters.
        """

        q = self.q
        block_size = self.block_size
        num_images = self.num_images
        init_with_pca = self.init_with_pca

        runner = self.psi.runner
        times = self.psi.times
        det = self.psi.det

        self.ipca_intervals['load_event'] = []
        self.ipca_intervals['concat'] = []
        self.ipca_intervals['ortho'] = []
        self.ipca_intervals['build_r'] = []
        self.ipca_intervals['svd'] = []
        self.ipca_intervals['update_basis'] = []

        start_idx = 0
        end_idx = min(self.psi.max_events, self.num_images)

        if num_images == -1:
            end_idx = self.psi.max_events
        
        imgs = np.array([[]])
        new_obs = np.array([[]])

        z, x, y = det.shape()
        d = z * x * y if not self.reduced_indices.size else self.reduced_indices.size
    
        self.get_distributed_indices(d)

        self.S = np.eye(q)
        self.U = np.zeros((d, q))
        self.mu = np.zeros((d, 1))
        self.total_variance = np.zeros((d, 1))

        for idx in np.arange(start_idx, end_idx):

            print(f'Processing observation: {idx + 1}')

            with TaskTimer(self.ipca_intervals['load_event']): 
                evt = runner.event(times[idx])
                img_panels = det.calib(evt=evt)
            
            img = self.flatten_img(img_panels)

            if self.reduced_indices.size:
                img = img[self.reduced_indices]
            
            new_obs = np.hstack((new_obs, img)) if new_obs.size else img

            # initialize model on first q observations, if init_with_pca is true
            if init_with_pca and (idx + 1) <= self.q:
                if (idx + 1) == q:
                    self.mu, self.total_variance = calculate_sample_mean_and_variance(new_obs)
                    
                    centered_obs = new_obs - np.tile(self.mu, q)
                    self.U, s, _ = np.linalg.svd(centered_obs, full_matrices=False)
                    self.S = np.diag(s)
                    
                    new_obs = np.array([[]])
                continue
            
            # update model with block every m samples, or img limit
            if (idx + 1) % self.block_size == 0 or idx == end_idx :

                # size of current block
                m = (idx + 1) % block_size if idx == end_idx else block_size

                # number of samples factored into model thus far
                n = (idx + 1) - m

                mu_m, s_m = calculate_sample_mean_and_variance(new_obs)
                
                with TaskTimer(self.ipca_intervals['concat']):
                    X_centered = new_obs - np.tile(mu_m, m)
                    X_m = np.hstack((X_centered, np.sqrt(n * m / (n + m)) * (mu_m - self.mu)))
                
                with TaskTimer(self.ipca_intervals['ortho']):
                    UX_m = self.U.T @ X_m
                    dX_m = X_m - self.U @ UX_m
                    X_pm, _ = np.linalg.qr(dX_m, mode='reduced')
                
                with TaskTimer(self.ipca_intervals['build_r']):
                    R = np.block([[self.S, UX_m], [np.zeros((m + 1, self.q)), X_pm.T @ dX_m]])
                
                with TaskTimer(self.ipca_intervals['svd']):
                    U_tilde, S_tilde, _ = np.linalg.svd(R)
                
                with TaskTimer(self.ipca_intervals['update_basis']):

                    U_split = self.U[self.start_index:self.end_index, :]
                    X_pm_split = X_pm[self.start_index:self.end_index, :]

                    U_prime_partial = np.hstack((U_split, X_pm_split)) @ U_tilde
                    U_prime = np.empty((d, self.q+m+1))

                    self.comm.Allgather(U_prime_partial, U_prime)

                    # U_prime = np.hstack((self.U, X_pm)) @ U_tilde
                    self.U = U_prime[:, :q]
                    self.S = np.diag(S_tilde[:q])

                    self.total_variance = update_sample_variance(self.total_variance, s_m, self.mu, mu_m, n, m)
                    self.mu = update_sample_mean(self.mu, mu_m, n, m)
                    
                new_obs = np.array([[]])

    def report_interval_data(self):

        if len(self.ipca_intervals) == 0:
            print('iPCA has not yet been performed.')
            return

        for key in list(self.ipca_intervals.keys()):
            interval_mean = np.mean(self.ipca_intervals[key])
            print(f'Mean compute time of step \'{key}\': {interval_mean:.4g}s')

    def flatten_img(self, img):
        z, x, y = self.psi.det.shape()
        return np.reshape(img, (x*y*z, 1))
    
    def unflatten_image(self, img):
        z, x, y = self.psi.det.shape()
        return np.reshape(img, (z, x, y))

def calculate_sample_mean_and_variance(imgs):
    d, m = imgs.shape

    mu_m = np.reshape(np.mean(imgs, axis=1), (d, 1))
    s_m  = np.zeros((d, 1))

    if m > 1:
        s_m = np.reshape(np.var(imgs, axis=1, ddof=1), (d, 1))
    
    return mu_m, s_m

def update_sample_mean(mu_n, mu_m, n, m):
    if n == 0:
        return mu_m

    return (1 / (n + m)) * (n * mu_n + m * mu_m)

def update_sample_variance(s_n, s_m, mu_n, mu_m, n, m):
    if n == 0:
        return s_m
    
    return ((n - 1) * s_n + (m - 1) * s_m ) / (n + m - 1) + (n*m*(mu_n - mu_m)**2) / ((n + m) * (n + m - 1))

def compression_loss(X, U):
    """
    Calculate the compression loss between centered observation matrix X and its rank-q reconstruction.

    Parameters
    ----------
    X : ndarray, shape (d x n)
        flattened, centered image data from n run events
    U : ndarray, shape (d x q)
        first q singular vectors of X, forming orthonormal basis of q-dimensional subspace of R^d
    
    Returns
    -------
    Ln : float
        compression loss of X
    """
    d, n = X.shape

    UX = U.T @ X
    UUX = U @ UX

    Ln = ((np.linalg.norm(X - UUX, 'fro')) ** 2) / n
    return Ln 


#### For command line use ####
            
def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M',  required=True, type=str)
    parser.add_argument('-q', '--components', help='Number of principal components to be extracted', required=False, type=int)
    parser.add_argument('-m', '--block_size', help='iPCA block size', required=False, type=int)
    parser.add_argument('-n', '--num_events', help='Number of events to process',  required=False, type=int)
    parser.add_argument('--pca_init', help='Initialize on q elements using batch PCA', required=False, action='store_true')

if __name__ == '__main__':
    params = parse_input()
    fe = FeatureExtractor(exp=params.exp, run=params.run, det_type=params.det_type)
    fe.set_ipca_parameters(q=params.components, block_size=params.block_size, num_images=params.num_events, pca_init=params.pca_init)
    fe.ipca()
    fe.report_interval_data()
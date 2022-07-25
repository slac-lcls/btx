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
    
    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)

        self.ipca_intervals = dict({})
        self.reduced_indices = np.array([])
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def generate_reduced_indices(self, new_dim):
        """
        """
        det_x_dim = self.psi.det.image_xaxis(self.psi.run).shape[0]
        det_y_dim = self.psi.det.image_yaxis(self.psi.run).shape[0]

        det_pixels = det_y_dim * det_x_dim

        if det_pixels < new_dim:
            print('Detector dimension must be greater than or equal to reduced dimension.')
            return

        reduced_dimension = new_dim if new_dim > 1 else int(np.floor(new_dim * det_pixels))

        self.reduced_indices = np.random.choice(det_pixels, reduced_dimension, replace=False)

    def remove_reduced_indices(self):
        self.reduced_indices = np.array([])

    def ipca(self, q, block_size, num_images, init_with_pca=True):
        """
        Run iPCA with run subset subject to initialization parameters.

        Parameters
        ----------

        num_images : int
            number of events consider, psi.max_events if -1
        """

        self.ipca_intervals['load_event'] = []
        self.ipca_intervals['update_mean'] = []
        self.ipca_intervals['concat'] = []
        self.ipca_intervals['ortho'] = []
        self.ipca_intervals['build_r'] = []
        self.ipca_intervals['svd'] = []
        self.ipca_intervals['update_basis'] = []

        start_idx = self.psi.counter
        end_idx = min(self.psi.max_events, num_images)

        if num_images == -1:
            end_idx = self.psi.max_events

        imgs = np.array([[]])
        new_obs = np.array([[]])
        
        runner = self.psi.runner
        times = self.psi.times
        det = self.psi.det
        
        for idx in np.arange(start_idx, end_idx):

            with TaskTimer(self.ipca_intervals['load_event']): 
                evt = runner.event(times[idx])
                img_yx = det.image(evt=evt)

            y, x = img_yx.shape
            d = y * x

            img = np.reshape(img_yx, (d, 1))

            if self.reduced_indices.size:
                img = img[self.reduced_indices]
                d, _ = img.shape
            
            if init_with_pca and n <= q:
                imgs = np.hstack((imgs, img)) if imgs.size else img
                
                if idx + 1 == q:
                    U, s, _ = np.linalg.svd(imgs, full_matrices=False)
                    S = np.diag(s)
                    
                    mu = np.mean(imgs, axis=1)
                    mu = np.reshape(mu, (d, 1))

            else:
                if idx == 0:
                    S = np.diag(np.ones(q))
                    mu = np.zeros((d, 1))
                    U = np.zeros((d, q))
                    np.fill_diagonal(U, 1)

                    s_n = np.zeros((d, 1))

                new_obs = np.hstack((new_obs, img)) if new_obs.size else img
                    
                # update model with block every m samples, or img limit
                
                if (idx + 1) % block_size == 0 or idx == end_idx :

                    with TaskTimer(self.ipca_intervals['update_mean']):

                        # size of current block
                        m = (idx + 1) % block_size if idx == end_idx else block_size

                        # number of samples factored into model thus far
                        n = (idx + 1) - m

                        mu_m = np.mean(new_obs, axis=1)
                        mu_m = np.reshape(mu_m, (d, 1))
                        mu_nm = (1 / (n + m)) * (n * mu + m * mu_m)
                    
                    s_m = np.reshape(np.var(new_obs, ddof=1, axis=1), (d, 1))
                    s_n = ((n - 1)*s_n + (m - 1)*s_m ) / (n + m - 1) + (n*m*(mu_n - mu_m)**2) / ((n+m)*(n+m-1))
                    
                    with TaskTimer(self.ipca_intervals['concat']):
                        X_centered = new_obs - np.tile(mu_m, (1, m))
                        X_m = np.hstack((X_centered, np.sqrt(n * m / (n + m)) * mu_m - mu))
                    
                    with TaskTimer(self.ipca_intervals['ortho']):
                        UX_m = U.T @ X_m
                        dX_m = X_m - U @ UX_m
                        X_pm, _ = np.linalg.qr(dX_m, mode='reduced')
                    
                    with TaskTimer(self.ipca_intervals['build_r']):
                        R = np.block([[S, UX_m], [np.zeros((m + 1,q)), X_pm.T @ dX_m]])
                    
                    with TaskTimer(self.ipca_intervals['svd']):
                        U_tilde, S_tilde, _ = np.linalg.svd(R)
                    
                    with TaskTimer(self.ipca_intervals['update_basis']):
                        U_prime = np.concatenate((U, X_pm), axis=1) @ U_tilde
                        U = U_prime[:, :q]
                        S = np.diag(S_tilde[:q])
                        mu = mu_nm
                        
                    new_obs = np.array([])

                    print(np.sum(np.diag(S)))
                    print(np.sum(s_n))
                    print(np.sum(np.diag(S)) / np.sum(s_n))
                    
            self.psi.counter += 1
            
            if self.psi.counter == self.psi.max_events:
                break
        
        self.U, self.S, self.mu = U, S, mu

    def report_interval_data(self):

        if len(self.ipca_intervals) == 0:
            print('iPCA has not yet been performed.')
            return

        for key in list(self.ipca_intervals.keys()):
            interval_mean = np.mean(self.ipca_intervals[key])
            print(f'Mean compute time of step \'{key}\': {interval_mean:.4g}s')

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

def statistical_accuracy(U, U_hat):
    return


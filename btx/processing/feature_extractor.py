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
from scipy.linalg import svd
from scipy.linalg import qr

class TimeTask:
    """
    A context manager to record the duration of wrapped task.

    Attributes
    ----------
    start_time : float
        reference time for start time of task
    intervals : list
        list containing time interval data
    """
    
    def __init__(self, intervals):
        """
        Construct all necessary attributes for the TimeTask context manager.

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
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def ipca(self, q, block_size, num_images, init_with_pca=True):

        self.ipca_intervals['update_mean'] = []
        self.ipca_intervals['concat'] = []
        self.ipca_intervals['ortho'] = []
        self.ipca_intervals['build_r'] = []
        self.ipca_intervals['svd'] = []
        self.ipca_intervals['update_basis'] = []

        start_idx = self.psi.counter
        end_idx = min(self.psi.max_events, num_images)
        
        imgs = np.array([[]])
        new_obs = np.array([[]])
        
        runner = self.psi.runner
        times = self.psi.times
        det = self.psi.det
        
        for idx in np.arange(start_idx, end_idx):
            n = idx + 1
            print(n)
            
            evt = runner.event(times[idx])
            img_yx = det.image(evt=evt)
            
            y, x = img_yx.shape
            img = np.reshape(img_yx, (x*y, 1))
            
            if init_with_pca and n <= q:
                imgs = np.hstack((imgs, img)) if imgs.size else img
                
                if n == q:
                    U, s, _ = svd(imgs, full_matrices=False)
                    S = np.diag(s)
                    
                    mu = np.mean(imgs, axis=1)
                    mu = np.reshape(mu, (x*y, 1))

                    print(initialized)

            else:
                if idx == 0:
                    S = np.diag(np.ones(q))
                    mu = np.zeros((x*y, 1))
                    U = np.zeros((x*y, q))
                    np.fill_diagonal(U, 1)
                    
                new_obs = np.hstack((new_obs, img)) if new_obs.size else img
                    
                # update model with block every m samples, or img limit
                
                if n % block_size == 0 or idx == end_idx :
                    
                    with TimeTask(self.ipca_intervals['update_mean']):
                        m = n % block_size if idx == end_idx else block_size
                        mu_m = np.mean(new_obs, axis=1)
                        mu_m = np.reshape(mu_m, (x*y, 1))
                        mu_nm = (1 / (n + m)) * (n * mu + m * mu_m)
                    
                    with TimeTask(self.ipca_intervals['concat']):
                        X_centered = new_obs - np.tile(mu_m, (1, m))
                        X_m = np.hstack((X_centered, np.sqrt(n * m / (n + m)) * mu_m - mu))
                    
                    with TimeTask(self.ipca_intervals['ortho']):
                        UX_m = U.T @ X_m
                        dX_m = X_m - U @ UX_m
                        X_pm, _ = qr(dX_m, mode='economic')
                    
                    with TimeTask(self.ipca_intervals['build_r']):
                        R = np.block([[S, UX_m], [np.zeros((m + 1,q)), X_pm.T @ dX_m]])
                    
                    with TimeTask(self.ipca_intervals['svd']):
                        U_tilde, S_tilde, _ = svd(R)
                    
                    with TimeTask(self.ipca_intervals['update_basis']):
                        U_prime = np.concatenate((U, X_pm), axis=1) @ U_tilde
                        U = U_prime[:, :q]
                        S = np.diag(S_tilde[:q])
                        mu = mu_nm
                        
                    new_obs = np.array([])
                    
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
            print(f'Mean compute time of step {key}: {interval_mean:.4g}s')

import os
import csv

import numpy as np
from mpi4py import MPI

from btx.interfaces.psana_interface import PsanaInterface
from btx.misc.ipca_helpers import (
    distribute_indices,
    calculate_sample_mean_and_variance,
    update_sample_variance,
    update_sample_mean,
    compression_loss,
)

from time import perf_counter
from matplotlib import pyplot as plt


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
        self.start_time = 0.0
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
    def __init__(
        self,
        exp,
        run,
        det_type,
        num_components=10,
        block_size=10,
        num_images=10,
        init_with_pca=False,
        benchmark_mode=False,
        downsample=False,
        bin_factor=2,
        output_dir="",
    ):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)

        self.init_with_pca = init_with_pca
        self.benchmark_mode = benchmark_mode
        self.downsample = downsample
        self.output_dir = output_dir

        self.num_images, self.q, self.m, self.d = self.set_ipca_params(
            num_images, num_components, block_size, bin_factor
        )

        self.bin_factor = bin_factor
        self.incorporated_images = 0

        # compute number of counts in and start indices over ranks
        self.split_indices, self.split_counts = distribute_indices(self.d, self.size)

        # attribute for storing interval data
        self.task_durations = dict({})

        self.S = np.ones(self.q)
        self.U = np.zeros((self.split_counts[self.rank], self.q))
        self.mu = np.zeros((self.split_counts[self.rank], 1))
        self.total_variance = np.zeros((self.split_counts[self.rank], 1))

        self.outliers, self.loss_data = [], []

    def set_ipca_params(self, num_images, num_components, block_size, bin_factor):
        """_summary_

        Parameters
        ----------
        num_images : _type_
            _description_
        num_components : _type_
            _description_
        block_size : _type_
            _description_
        bin_factor : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        max_events = self.psi.max_events
        benchmark = self.benchmark_mode
        downsample = self.downsample

        # set n, q, and m
        n = num_images
        q = num_components
        m = block_size

        if benchmark:
            min_n = max(int(4 * q), 40)
            n = min(min_n, max_events)
            m = q
        else:
            n = min(n, max_events) if n != -1 else max_events
            q = min(q, n)
            m = min(m, n)

        # set d
        det_shape = self.psi.det.shape()
        d = np.prod(det_shape).astype(int)

        if downsample:
            if det_shape[-1] % bin_factor or det_shape[-2] % bin_factor:
                print("Invalid bin factor, toggled off downsampling.")
                self.downsample = False
            else:
                d = int(d / bin_factor**2)

        return n, q, m, d

    def initialize_model(self, X):
        """
        Initialiize model on sample of data using batch PCA.

        Parameters
        ----------
        X : ndarray, shape (d x m)
            set of m (d x 1) observations
        """
        print(f"Rank {self.rank}, initializing model with {self.q} samples...")

        self.mu, self.total_variance = calculate_sample_mean_and_variance(X)

        centered_data = X - np.tile(self.mu, self.q)
        self.U, self.S, _ = np.linalg.svd(centered_data, full_matrices=False)

        self.incorporated_images += self.q

    def parallel_qr(self, A):
        """_summary_

        Parameters
        ----------
        A : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Notes
        -----
        Method acquired from
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6691583&tag=1
        """
        _, x = A.shape
        q = self.q
        m = x - q - 1

        with TaskTimer(self.task_durations, "qr - local qr"):
            q_loc, r_loc = np.linalg.qr(A, mode="reduced")

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - r_tot gather"):
            if self.rank == 0:
                r_tot = np.empty((self.size * (q + m + 1), q + m + 1))
            else:
                r_tot = None

            self.comm.Gather(r_loc, r_tot, root=0)

        if self.rank == 0:
            with TaskTimer(self.task_durations, "qr - global qr"):
                q_tot, r_tilde = np.linalg.qr(r_tot, mode="reduced")

            with TaskTimer(self.task_durations, "qr - global svd"):
                U_tilde, S_tilde, _ = np.linalg.svd(r_tilde)
        else:
            U_tilde = np.empty((q + m + 1, q + m + 1))
            S_tilde = np.empty(q + m + 1)
            q_tot = None

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - scatter q_tot"):
            q_tot_loc = np.empty((q + m + 1, q + m + 1))
            self.comm.Scatter(q_tot, q_tot_loc, root=0)

        with TaskTimer(self.task_durations, "qr - local matrix build"):
            q_fin = q_loc @ q_tot_loc

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - bcast S_tilde"):
            self.comm.Bcast(S_tilde, root=0)

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - bcast U_tilde"):
            self.comm.Bcast(U_tilde, root=0)

        return q_fin, U_tilde, S_tilde

    def update_model(self, X):
        """
        Update model with new block of observations using iPCA.

        Parameters
        ----------
        X : ndarray, shape (d x m)
            block of m (d x 1) observations

        Notes
        -----
        Method retrieved from
        https://link.springer.com/content/pdf/10.1007/s11263-007-0075-7.pdf
        """
        _, m = X.shape
        n = self.incorporated_images
        q = self.q

        with TaskTimer(self.task_durations, "total update"):

            if self.rank == 0:
                print(
                    "Factoring {m} sample{s} into {n} sample, {q} component model...".format(
                        m=m, s="s" if m > 1 else "", n=n, q=q
                    )
                )

            with TaskTimer(self.task_durations, "record compression loss data"):
                if n > 0:
                    self.gather_interim_data(X)

            with TaskTimer(self.task_durations, "update mean and variance"):
                mu_n = self.mu
                mu_m, s_m = calculate_sample_mean_and_variance(X)

                self.total_variance = update_sample_variance(
                    self.total_variance, s_m, mu_n, mu_m, n, m
                )
                self.mu = update_sample_mean(mu_n, mu_m, n, m)

            with TaskTimer(
                self.task_durations, "center data and compute augment vector"
            ):
                X_centered = X - np.tile(mu_m, m)
                v_augment = np.sqrt(n * m / (n + m)) * (mu_m - mu_n)

                X_aug = np.hstack((X_centered, v_augment))

            with TaskTimer(self.task_durations, "first matrix product U@S"):
                us = self.U @ np.diag(self.S)

            with TaskTimer(self.task_durations, "QR concatenate"):
                qr_input = np.hstack((us, X_aug))

            with TaskTimer(self.task_durations, "parallel QR"):
                UB_tilde, U_tilde, S_tilde = self.parallel_qr(qr_input)

            # concatenating first preserves the memory contiguity
            # of U_prime and thus self.U
            with TaskTimer(self.task_durations, "compute local U_prime"):
                U_prime = UB_tilde @ U_tilde[:, :q]

            self.U = U_prime
            self.S = S_tilde[:q]

            self.incorporated_images += m

    def get_model(self):
        """
        Notes
        -----
        Intended to be called from the root process.
        """
        if self.rank == 0:
            U_tot = np.empty(self.d * self.q)
            mu_tot = np.empty((self.d, 1))
            var_tot = np.empty((self.d, 1))
            S_tot = self.S
        else:
            U_tot, mu_tot, var_tot, S_tot = None, None, None, None

        self.comm.Gatherv(
            self.U.flatten(),
            [
                U_tot,
                self.split_counts * self.q,
                self.start_indices * self.q,
                MPI.DOUBLE,
            ],
            root=0,
        )

        if self.rank == 0:
            U_tot = np.reshape(U_tot, (self.d, self.q))

        self.comm.Gatherv(
            self.mu,
            [
                mu_tot,
                self.split_counts * self.q,
                self.start_indices,
                MPI.DOUBLE,
            ],
            root=0,
        )
        self.comm.Gatherv(
            self.total_variance,
            [
                var_tot,
                self.split_counts * self.q,
                self.start_indices,
                MPI.DOUBLE,
            ],
            root=0,
        )

        S_tot = self.S

        return U_tot, S_tot, mu_tot, var_tot

    def get_loss_stats(self):

        if self.rank == 0:
            print(self.outliers)

    def gather_interim_data(self, X):
        """_summary_

        Parameters
        ----------
        X : _type_
            _description_
        """
        _, m = X.shape
        n, d = self.incorporated_images, self.d

        U, _, mu, _ = self.get_model()

        if self.rank == 0:
            X_tot = np.empty((d, m))
        else:
            X_tot = None

        self.comm.Gatherv(
            X.flatten(),
            [
                X_tot,
                self.split_counts * m,
                self.start_indices * m,
                MPI.DOUBLE,
            ],
            root=0,
        )

        if self.rank == 0:

            X_tot = np.reshape(X_tot, (d, m))
            cb = X_tot - np.tile(mu, (1, m))

            pcs = U.T @ cb
            self.loss_data = (
                np.concatenate((self.loss_data, pcs), axis=1)
                if len(self.loss_data)
                else pcs
            )

            pc_dist = np.linalg.norm(pcs[:5], axis=0)
            std = np.std(pc_dist)
            mu = np.mean(pc_dist)

            block_outliers = np.where(np.abs(pc_dist - mu) > std)[0] + n - m

            self.outliers = (
                np.concatenate((self.outliers, block_outliers), axis=0)
                if len(self.outliers)
                else block_outliers
            )

    def verify_model_accuracy(self):
        d = self.d
        m = self.m
        q = self.q
        n = self.num_images

        self.comm.Barrier()
        U, S, mu, var = self.ipca.get_model()

        if self.rank == 0:
            # store current event index from self.psi and reset
            event_index = self.psi.counter
            self.psi.counter = event_index - n

            try:
                print("Verifying Model Accuracy\n------------------------\n")
                print(f"q = {q}")
                print(f"d = {d}")
                print(f"n = {n}")
                print(f"m = {m}")
                print("\n")

                # run svd on centered image batch
                print("\nGathering images for batch PCA...")
                X = self.fetch_formatted_images(n, fetch_all_features=True)

                print("Performing batch PCA...")
                mu_pca = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
                var_pca = np.reshape(np.var(X, ddof=1, axis=1), (X.shape[0], 1))

                mu_n = np.tile(mu_pca, n)
                X_centered = X - mu_n

                U_pca, S_pca, _ = np.linalg.svd(X_centered, full_matrices=False)
                print("\n")

                q_pca = min(q, n)

                # calculate compression loss, normalized if given flag
                norm = True
                norm_str = "Normalized " if norm else ""

                ipca_loss = compression_loss(X, U[:, :q_pca], normalized=norm)
                print(f"iPCA {norm_str}Compression Loss: {ipca_loss}")

                pca_loss = compression_loss(X, U_pca[:, :q_pca], normalized=norm)
                print(f"PCA {norm_str}Compression Loss: {pca_loss}")

                print("\n")
                ipca_tot_var = np.sum(var)
                pca_tot_var = np.sum(var_pca)

                print(f"iPCA Total Variance: {ipca_tot_var}")
                print(f"PCA Total Variance: {pca_tot_var}")
                print("\n")

                ipca_exp_var = (np.sum(S[:q_pca] ** 2) / (n - 1)) / np.sum(var)
                print(f"iPCA Explained Variance: {ipca_exp_var}")

                pca_exp_var = (np.sum(S_pca[:q_pca] ** 2) / (n - 1)) / np.sum(var_pca)
                print(f"PCA Explained Variance: {pca_exp_var}")
                print("\n")

                print("iPCA Singular Values: \n")
                print(S)
                print("\n")

                print("PCA Singular Values: \n")
                print(S_pca[:q_pca])
                print("\n")

                mean_inner_prod = np.inner(mu.flatten(), mu_pca.flatten()) / (
                    np.linalg.norm(mu) * np.linalg.norm(mu_pca)
                )

                print("Normalized Mean Inner Product: " + f"{mean_inner_prod}")
                print("\n")

                print("Basis Inner Product: \n")
                print(np.diagonal(np.abs(U[:, :q_pca].T @ U_pca[:, :q_pca])))

                ipca_mu_u = np.hstack((mu / np.linalg.norm(mu), U[:, :q_pca]))
                pca_mu_u = np.hstack(
                    (mu_pca / np.linalg.norm(mu_pca), U_pca[:, :q_pca])
                )

                b = plt.imshow(np.abs(ipca_mu_u.T @ pca_mu_u))
                plt.colorbar(b)
                plt.savefig(f"fig_{q}_{self.size}.png")
                plt.show()

                print("\n")
                report_interval_data(self.task_durations)

            finally:
                # reset counter
                self.psi.counter = event_index

    def run_ipca(self):
        """
        Perform iPCA on run subject to initialization parameters.
        """
        d = self.d
        m = self.m
        q = self.q

        rank = self.rank
        size = self.size
        num_images = self.num_images
        start_index, end_index = self.split_indices[rank], self.split_indices[rank + 1]

        if self.init_with_pca and not self.benchmark_mode:
            img_block = self.psi.fetch_formatted_images(
                q, d, start_index, end_index, self.downsample, self.bin_factor
            )
            self.initialize_model(img_block)

        # divide remaning number of images into blocks
        # will become redundant in a streaming setting, need to change
        rem_imgs = num_images - self.incorporated_images
        block_sizes = np.array(
            [m] * np.floor(rem_imgs / m).astype(int)
            + ([rem_imgs % m] if rem_imgs % m else [])
        )

        # update model with remaining blocks
        for block_size in block_sizes:
            img_block = self.psi.fetch_formatted_images(
                block_size, d, start_index, end_index, self.downsample, self.bin_factor
            )
            self.update_model(img_block)

        if self.benchmark_mode and rank == 0:
            save_interval_data(
                q,
                d,
                self.incorporated_images,
                size,
                m,
                self.task_durations,
                self.output_dir,
            )


def report_interval_data(task_durations):
    """
    Report time interval data gathered during iPCA.
    """

    if not len(task_durations):
        print("No model updates or initialization recorded.")
        return

    for key in list(task_durations.keys()):
        interval_mean = np.mean(task_durations[key])
        print("Mean per-block compute time of step" + f"'{key}': {interval_mean:.4g}s")


def save_interval_data(q, d, n, r, m, task_durations, dir_path=None):
    """
    Save time interval data gathered during iPCA to file.

    Parameters
    ----------
    dir_path : str
        Path to output directory.
    """

    if not dir_path:
        print("Failed to specify output directory.")
        return

    file_name = "task_" + str(q) + str(d) + str(n) + str(r) + ".csv"

    with open(
        os.path.join(dir_path, file_name),
        "x",
        newline="",
        encoding="utf-8",
    ) as f:

        if len(task_durations):
            writer = csv.writer(f)

            writer.writerow(["q", q])
            writer.writerow(["d", d])
            writer.writerow(["n", n])
            writer.writerow(["ranks", r])
            writer.writerow(["m", m])

            keys = list(task_durations.keys())
            values = list(task_durations.values())
            values_transposed = np.array(values).T

            writer.writerow(keys)
            writer.writerows(values_transposed)

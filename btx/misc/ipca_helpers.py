import numpy as np
import csv
from matplotlib import pyplot as plt


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
    s_m = np.zeros((d, 1))

    if m > 1:
        s_m = np.reshape(np.var(imgs, axis=1, ddof=1), (d, 1))

    return mu_m, s_m


def update_sample_mean(mu_n, mu_m, n, m):
    """
    Compute combined mean of two blocks of data.

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
    Compute combined sample variance of two blocks
    of data described by input parameters.

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
        s_nm = (
            ((n - 1) * s_n + (m - 1) * s_m) + (n * m * (mu_n - mu_m) ** 2) / (n + m)
        ) / (n + m - 1)

    return s_nm


def compare_basis_vectors(U_1, U_2, q):
    """
    Quantitatively compare the first q basis vectors of U and U_prime.

    Parameters
    ----------
    U_1 : ndarray, shape (d x a), a >= q
        first matrix of orthonormal basis vectors
    U_2 : ndarray, shape (d x b), b >= q
        second matrix of orthonormal basis vectors
    q : int
        number of vectors to compare

    Returns
    -------
    acc : float, 0 <= acc <= 1
        quantitative measure of distance between basis vectors
    """
    if q > min(U_1.shape[1], U_2.shape[1]):
        print("Desired number of vectors is greater than matrix dimension.")
        return 0.0

    acc = np.trace(np.abs(U_1[:, :q].T @ U_2[:, :q])) / q
    return acc


def compression_loss(X, U, normalized=False):
    """
    Calculate the compression loss between centered observation matrix X
    and its rank-q reconstruction.

    Parameters
    ----------
    X : ndarray, shape (d x n)
        flattened, centered image data from n run events
    U : ndarray, shape (d x q)
        first q singular vectors of X, forming orthonormal basis
        of q-dimensional subspace of R^d

    Returns
    -------
    Ln : float
        compression loss of X
    """
    _, n = X.shape

    UX = U.T @ X
    UUX = U @ UX

    Ln = ((np.linalg.norm(X - UUX, "fro")) ** 2) / n

    if normalized:
        Ln /= np.linalg.norm(X, "fro") ** 2

    return Ln


def parse_single_run(data_dir, file_name):
    headers = []

    with open(data_dir + file_name, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        q = 0
        d = 0
        n = 0
        ranks = 0
        m = 0

        for i in range(6):
            row = next(reader)

            if i == 0:
                q = int(row[1])
            if i == 1:
                d = int(row[1])
            if i == 2:
                n = int(row[1])
            if i == 3:
                ranks = int(row[1])
            if i == 4:
                m = int(row[1])
            if i == 5:
                headers = row

        num_entries = np.floor(n / m).astype(int)
        interval_values = np.array([])

        for i, row in enumerate(reader):
            if i == 0:
                continue

            if i == num_entries - 1 and n % m != 0:
                break

            row_data = np.array(row, dtype=np.float32) / m

            interval_values = (
                np.vstack([interval_values, row_data])
                if interval_values.size
                else row_data
            )

        mean = np.mean(interval_values, axis=0)
        stdev = np.std(interval_values, axis=0)

    return mean, stdev, q, d, n, ranks, m, headers


def display_interval_data(data_dir, data_desc, savefig=False, tiled_plots=True):
    intervals = {}
    headers = []

    for file_name in os.listdir(data_dir):
        if file_name == ".ipynb_checkpoints":
            continue
        value_mean, stdev, q, d, n, ranks, m, headers = parse_single_run(
            data_dir, file_name
        )
        intervals[q] = (value_mean, stdev)

    qs = np.array(list(intervals.keys()))
    interval_data = np.array(list(intervals.values()))
    indices = np.argsort(qs)

    qs = qs[indices]
    interval_data = interval_data[indices]

    # generate overlayed interval plot
    num_imgs = len(headers)
    num_cols = 3

    num_rows = int(num_imgs / num_cols)

    if num_imgs % num_cols != 0:
        num_rows += 1

    total_runtime = interval_data[:, :, -1]

    for i in range(num_imgs):
        plt.plot(qs, interval_data[:, 0, i], label=headers[i])
        plt.errorbar(qs, interval_data[:, 0, i], interval_data[:, 1, i], fmt="none")

    plt.rcParams["figure.figsize"] = (15, 8)
    plt.xlabel("q")
    plt.ylabel("per-block compute time, s")
    plt.title(
        f"Per-Block Compute Time (s) vs. Number of Stored Components (q) for iPCA Algorithm (n = {n}, d = {d}, cores = {ranks}, m = {m})"
    )
    plt.legend(loc="best")

    if savefig:
        plt.savefig(data_desc + "_overlayed")

    plt.show()
    plt.clf()

    if tiled_plots:
        # generate paned per-task interval plots
        fig = plt.figure(1)
        fig.set_size_inches(15, 10)
        fig.set_dpi(100)

        for i in range(num_imgs):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.set_title(f"Step '{headers[i]}'")
            ax.set_xlabel("q")
            ax.set_ylabel("per-block compute time, s")
            ax.plot(qs, interval_data[:, 0, i])
            ax.errorbar(qs, interval_data[:, 0, i], interval_data[:, 1, i], fmt="none")

        fig.suptitle(
            f"Per-Block Compute Time (s) vs. Number of Stored Components (q) for iPCA Algorithm Stages (n = {n}, d = {d}, ranks = {ranks}, m = {m})"
        )
        fig.tight_layout()

        if savefig:
            plt.savefig(data_desc + "_pane")

        plt.show()
        plt.clf()

    return (qs, total_runtime)

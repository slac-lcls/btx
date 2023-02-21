import csv, os
import numpy as np
from matplotlib import pyplot as plt

def parse_single_run(data_dir, file_name):
    """
    Method to parse and retrive data from a single file in an iPCA benchmark run.

    Parameters
    ----------
    data_dir : str
        directory containing all benchmark files from iPCA run
    file_name : str
        file name of specific benchmark file to parse

    Returns
    -------
    mean : ndarray
        mean of each iPCA task duration
    stdev : ndarray
        stdev of each iPCA task duration
    q : ndarray
        number of computed components in iPCA run
    d : ndarray
        dimension of underlying iPCA data
    n : ndarray
        total number of images incorporated into iPCA model
    ranks : ndarray
        size of MPI world in which iPCA was run
    m : ndarray
        batch size of each model update
    headers : ndarray
        titles of each task in the iPCA algorithm whose time was recorded
    """
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
    """
    Method to display interval from an iPCA benchmarking run.

    Parameters
    ----------
    data_dir : str
        directory containing timed run data files
    data_desc : str
        description of benchmarking run
    savefig : bool, optional
        if true, save the generated figure to the working directory, by default False
    tiled_plots : bool, optional
        if true, generated per-task paned plot, by default True

    Returns
    -------
    qs : ndarray
        components of each iteration in benchmarking block
    total_runtime : ndarray
        total average runtime and standard deviation of each iPCA task
    """
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

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

    def __init__(self, d, q):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.d = d
        self.q = q
        self.n = 0

        self.S = np.eye(self.q)
        self.U = np.zeros((self.d, self.q))
        self.mu = np.zeros((self.d, 1))
        self.total_variance = np.zeros((self.d, 1))

        self.get_distributed_indices()

        # attributes for computing timings of iPCA steps
        self.task_durations = dict({})
        self.task_durations['concat'] = []
        self.task_durations['ortho'] = []
        self.task_durations['build_r'] = []
        self.task_durations['svd'] = []
        self.task_durations['update_basis'] = []

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

        mu_m, s_m = calculate_sample_mean_and_variance(X)

        X_pm_loc = None

        with TaskTimer(self.task_durations['concat']):
            X_centered = X - np.tile(mu_m, m)
            X_m = np.hstack((X_centered, np.sqrt(n * m / (n + m)) * (mu_m - self.mu)))

        with TaskTimer(self.task_durations['ortho']):
            UX_m = self.U.T @ X_m
            dX_m = X_m - self.U @ UX_m
            X_pm, _ = np.linalg.qr(dX_m, mode='reduced')
        
        with TaskTimer(self.task_durations['build_r']):
            R = np.block([[self.S, UX_m], [np.zeros((m + 1, q)), X_pm.T @ dX_m]])
        
        with TaskTimer(self.task_durations['svd']):
            U_tilde, S_tilde, _ = np.linalg.svd(R)
        
        print(self.rank)

        self.comm.Barrier()

        with TaskTimer(self.task_durations['update_basis']):
            U_split = self.U[self.start_index:self.end_index, :]
            X_pm_split = X_pm[self.start_index:self.end_index, :]

            U_prime_partial = np.hstack((U_split, X_pm_split)) @ U_tilde
            print(U_prime_partial.shape)
            
            U_prime = np.empty((d, q+m+1))

            self.comm.Allgather(U_prime_partial, U_prime)

            print(U_prime.shape)
        
        self.U = U_prime[:, :q]
        self.S = np.diag(S_tilde[:q])

        self.total_variance = update_sample_variance(self.total_variance, s_m, self.mu, mu_m, n, m)
        self.mu = update_sample_mean(self.mu, mu_m, n, m)

        self.n += m

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

    def get_distributed_indices(self):
        """
        Distribute indices over ranks in MPI world.
        """
        split_indices = np.zeros(self.size)
        for r in range(self.size):
            num_per_rank = self.d // self.size
            if r < (self.d % self.size):
                num_per_rank += 1
            split_indices[r] = num_per_rank
        split_indices = np.append(np.array([0]), np.cumsum(split_indices)).astype(int)

        self.start_index = split_indices[self.rank]
        self.end_index = split_indices[self.rank + 1]

    def report_interval_data(self):
        """
        Report time interval data gathered during iPCA.
        """
        if len(self.task_durations) == 0:
            print('iPCA has not yet been performed.')
            return
        
        for key in list(self.task_durations.keys()):
            interval_mean = np.mean(self.task_durations[key])
            print(f'Mean per-block compute time of step \'{key}\': {interval_mean:.4g}s')


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
        print('Desired number of vectors is greater than matrix dimension.')
        return 0.
    
    acc = np.trace(np.abs(U_1[:, :q].T @ U_2[:, :q])) / q
    return acc

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

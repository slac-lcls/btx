import numpy as np



def distribute_indices(d, size):

    # determine boundary indices between ranks
    split_indices = np.zeros(size)
    for r in range(size):
        num_per_rank = d // size
        if r < (d % size):
            num_per_rank += 1
        split_indices[r] = num_per_rank

    trailing_indices = np.cumsum(split_indices)
    split_indices = np.append(np.array([0]), trailing_indices).astype(int)

    # determine number of images per each rank
    split_counts = np.empty(size, dtype=int)
    for i in range(len(split_indices) - 1):
        split_counts[i] = split_indices[i + 1] - split_indices[i]

    return split_indices, split_counts

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


def bin_data(arr, bin_factor, det_shape=None):
    """
    Bin detector data by bin_factor through averaging.
    Retrieved from
    https://github.com/apeck12/cmtip/blob/main/cmtip/prep_data.py

    :param arr: array shape (n_images, n_panels, panel_shape_x, panel_shape_y)
      or if det_shape is given of shape (n_images, 1, n_pixels_per_image)
    :param bin_factor: how may fold to bin arr by
    :param det_shape: tuple of detector shape, optional
    :return arr_binned: binned data of same dimensions as arr
    """
    # reshape as needed
    if det_shape is not None:
        arr = np.array([arr[i].reshape(det_shape) for i in range(arr.shape[0])])

    n, p, y, x = arr.shape

    # ensure that original shape is divisible by bin factor
    assert y % bin_factor == 0
    assert x % bin_factor == 0

    # bin each panel of each image
    binned_arr = (
        arr.reshape(
            n,
            p,
            int(y / bin_factor),
            bin_factor,
            int(x / bin_factor),
            bin_factor,
        )
        .mean(-1)
        .mean(3)
    )

    # if input data were flattened, reflatten
    if det_shape is not None:
        flattened_size = np.prod(np.array(binned_arr.shape[1:]))
        binned_arr = binned_arr.reshape((binned_arr.shape[0], 1) + (flattened_size,))
    return binned_arr


def bin_pixel_index_map(arr, bin_factor):
    """
    Bin pixel_index_map by bin factor.
    Retrieved from
    https://github.com/apeck12/cmtip/blob/main/cmtip/prep_data.py

    :param arr: pixel_index_map of shape (n_panels, panel_shape_x, panel_shape_y, 2)
    :param bin_factor: how may fold to bin arr by
    :return binned_arr: binned pixel_index_map of same dimensions as arr
    """
    arr = np.moveaxis(arr, -1, 0)
    arr = np.minimum(arr[..., ::bin_factor, :], arr[..., 1::bin_factor, :])
    arr = np.minimum(arr[..., ::bin_factor], arr[..., 1::bin_factor])
    arr = arr // bin_factor

    return np.moveaxis(arr, 0, -1)


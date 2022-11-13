import numpy as np

def cos_sq(angles):
    """ Compute cosine squared of input angles in radians. """
    return np.square(np.cos(angles))

def sin_sq(angles):
    """ Compute sine squared of input angles in radianss. """
    return np.square(np.sin(angles))

def compute_resolution(cell, hkl):
    """
    Compute reflections' resolution in 1/Angstrom. To check, see: 
    https://www.ruppweb.org/new_comp/reciprocal_cell.htm.
        
    Parameters
    ----------
    cell : numpy.ndarray, shape (n_refl, 6)
        unit cell parameters (a,b,c,alpha,beta,gamma) in Ang/deg
    hkl : numpy.ndarray, shape (n_refl, 3)
        Miller indices of reflections
            
    Returns
    -------
    resolution : numpy.ndarray, shape (n_refl)
        resolution associated with each reflection in 1/Angstrom
    """

    a,b,c = [cell[:,i] for i in range(3)] 
    alpha,beta,gamma = [np.radians(cell[:,i]) for i in range(3,6)] 
    h,k,l = [hkl[:,i] for i in range(3)]

    pf = 1.0 - cos_sq(alpha) - cos_sq(beta) - cos_sq(gamma) + 2.0*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
    n1 = np.square(h)*sin_sq(alpha)/np.square(a) + np.square(k)*sin_sq(beta)/np.square(b) + np.square(l)*sin_sq(gamma)/np.square(c)
    n2a = 2.0*k*l*(np.cos(beta)*np.cos(gamma) - np.cos(alpha))/(b*c)
    n2b = 2.0*l*h*(np.cos(gamma)*np.cos(alpha) - np.cos(beta))/(c*a)
    n2c = 2.0*h*k*(np.cos(alpha)*np.cos(beta) - np.cos(gamma))/(a*b)

    return np.sqrt((n1 + n2a + n2b + n2c) / pf)

def compute_cell_volume(cell):
    """
    Compute unit cell volume.
    
    Parameters
    ----------
    cell : numpy.ndarray, shape (n_refl, 6)
        unit cell parameters (a,b,c,alpha,beta,gamma) in Ang/deg

    Returns
    -------
    volume : numpy.ndarray, shape (n_refl)
        unit cell volume in Angstroms cubed
    """
    a,b,c = [10.0 * cell[:,i] for i in range(3)] 
    alpha,beta,gamma = [np.radians(cell[:,i]) for i in range(3,6)] 
    
    volume = 1.0 - cos_sq(alpha) - cos_sq(beta) - cos_sq(gamma) + 2.0*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
    volume = a*b*c*np.sqrt(volume)
    return volume


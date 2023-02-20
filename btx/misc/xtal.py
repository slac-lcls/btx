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

def enforce_symmetry(cell, sg_number):
    """
    Impose space group symmetry on the unit cell parameters.
    
    Parameters
    ----------
    cell : ndarary, shape (6,)
        unit cell parameters in Angstrom / degrees
    sg_number : int
        space group number
        
    Returns
    -------
    cell : ndarary, shape (6,)
        unit cell parameters with symmetry enforced    
    """
    # monoclonic: alpha=gamma=90
    if (sg_number>=3) and (sg_number<=15):
        cell[3], cell[5] = 90, 90
    
    # orthorhombic: all angles are 90 degrees
    if (sg_number>=15) and (sg_number<=74):
        cell[3:] = 90, 90, 90
        
    # tetragonal: all angles are 90 degrees, a=b!=c
    if (sg_number>=75) and (sg_number<=142):
        mean_ab = np.mean(cell[:2])
        cell[:2] = mean_ab, mean_ab
        cell[3:] = 90, 90, 90
    
    # trigonal: 
    if (sg_number>=143) and (sg_number<=167):
        mean_abc = np.mean(cell[:3])
        cell[:3] = mean_abc, mean_abc, mean_abc
        mean_angle = np.mean(cell[3:])
        cell[3:] = mean_angle, mean_angle, mean_angle
    
    # hexagonal: a=b!=c
    if (sg_number>=168) and (sg_number<=194):
        mean_ab = np.mean(cell[:2])
        cell[:2] = mean_ab, mean_ab
        cell[3:] = 90, 90, 120
    
    # cubic: all angles are 90 degrees, a=b=c
    if (sg_number>=195):
        mean_abc = np.mean(cell[:3])
        cell[:3] = mean_abc, mean_abc, mean_abc
        cell[3:] = 90, 90, 90
    
    return cell

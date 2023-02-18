import numpy as np
from btx.misc.radial import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

class AgBehenate:
    
    """
    Class for optimizing geometry (distance to detector and center) from a silver 
    behenate powder, leveraging the fact that the peaks are equidistant in q-space.
    """
    
    def __init__(self, powder, mask, pixel_size, wavelength):
        """
        Instantiate object.
                
        Parameters
        ----------
        powder : numpy.ndarray, 2d
            powder diffraction image, in shape of assembled detector
        mask : numpy.ndarray, 2d
            binary mask in shape of powder image
        pixel_size : float
            detector pixel size in mm
        wavelength : float
            beam wavelength in Angstrom
        """
        self.q0 = 0.1076 # |q| of first peak in Angstrom
        self.delta_qs = np.arange(0.01,0.03,0.00005) # q-spacings to scan over
        self.powder = powder
        self.mask = mask
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.centers, self.distances, self.npeaks = [], [], []
        self.scores_min, self.scores_mean, self.scores_std = [], [], []
        
    def ideal_rings(self, qPeaks):
        """
        Scan over and score various inter-peak distances in q-space, based
        on the equidistance of peaks in q-space for silver behenate powder.

        Parameters
        ----------
        qPeaks : numpy.ndarray, 1d
            positions of powder peaks in q-space (inverse Angstrom)

        Returns
        -------
        rings : numpy.ndarray, 1d
            predicted positions of peaks based on the best fit q-spacing
        scores : numpy.ndarray, 1d
            residual associated with each q-spacing, ordered as self.delta_qs
        """
        order_max=13
        qround = np.round(qPeaks,5)
        scores = [] 
        for dq in self.delta_qs:
            order  = qround/dq
            remain = np.minimum(qround%dq, np.abs(dq-(qround%dq)))
            score  = np.mean(remain[np.where(order<order_max)])/dq # %mod helps prevent half periods from scoring well
            scores.append(score)
        deltaq_current = self.delta_qs[np.argmin(scores)] 
        rings = np.arange(deltaq_current, deltaq_current*(order_max+1), deltaq_current)
        self.scores_min.append(np.min(scores))
        self.scores_mean.append(np.mean(scores))
        self.scores_std.append(np.std(scores))
        return rings, np.array(scores)

    def detector_distance(self, est_q0):
        """
        Estimate the sample-detector distance, based on the fact that the 
        first diffraction ring for silver behenate is at 0.1076/Angstrom. 

        Parameters
        ----------
        est_q0 : float
            predicted q-position of first ring based on the best fit q-spacing in Angstrom

        Returns
        -------
        distance : float
            refined detector distance in mm
        """
        distance = est_q0 * self.pixel_size / np.tan(2. * np.arcsin(self.wavelength * (self.q0 / (2*np.pi))/2))
        print("Detector distance inferred from powder rings: %s mm" % (np.round(distance,2)))
        return distance
    
    def opt_distance(self, distance_i=None, plot=None, vmax=None):
        """
        Optimize the sample-detector distance based on the powder image.
        
        Parameters
        ----------
        distance_i : float 
            initial estimated distance in mm
        plot : str or None
            output path for figure; if '', plot but don't save; if None, don't plot
        vmax : float 
            vmax value for powder plot 
        
        Returns
        -------
        peaks_observed : numpy.ndarray, 1d
            radii of detected powder peaks in pixels
        peak_heights : numpy.ndarray, 1d
            intensities associated with peaks_observed
        """
        from scipy.signal import find_peaks

        if distance_i is None:
            distance_i = self.distances[-1]
        
        # determine peaks in radial intensity profile and associated positions in q
        iprofile = radial_profile(self.powder, center=self.centers[-1], mask=self.mask)
        peaks_observed, properties = find_peaks(iprofile, prominence=1, distance=10)
        qprofile = pix2q(np.arange(iprofile.shape[0]), self.wavelength, distance_i, self.pixel_size)
        
        # optimize the detector distance based on inter-peak distances
        rings, scores = self.ideal_rings(qprofile[peaks_observed])
        peaks_predicted = q2pix(rings, self.wavelength, distance_i, self.pixel_size)
        opt_distance = self.detector_distance(peaks_predicted[0])
        
        if plot is not None:
            self.visualize_results(self.powder, mask=self.mask, vmax=vmax, center=self.centers[-1], 
                                   peaks_predicted=peaks_predicted, peaks_observed=peaks_observed,
                                   scores=scores, Dq=self.delta_qs,
                                   radialprofile=iprofile, qprofile=qprofile, plot=plot)
        
        self.distances.append(opt_distance)
        return peaks_observed, iprofile[peaks_observed]
    
    def opt_center(self, peaks_observed, num_circle_points=200):
        """
        Optimize the detector center by fitting circles to pixels predicted
        to fall in the powder rings.

        Parameters
        ----------
        peaks_observed : numpy.ndarray, 1d
            radii of detected powder peaks in pixels
        num_circle_points : int
            number of points per powder ring to use for fitting
        """
        # Create a concentric circle model...
        cx, cy = self.centers[-1]
        model = OptimizeConcentricCircles(cx=cx, cy=cy, r=peaks_observed, num_circle_points=num_circle_points)
        model.generate_crds()
        crds_init = model.crds.copy()
        crds_init = crds_init.reshape(2, -1, num_circle_points)

        # Fitting...
        img = (self.powder - np.mean(self.powder)) / np.std(self.powder)
        res = model.fit(img)
        model.report_fit(res)
        crds = model.crds
        crds = crds.reshape(2, -1, num_circle_points)

        # Update the center position...
        cx = res.params['cx'].value
        cy = res.params['cy'].value
        center = (cx, cy)
        self.centers.append(center)
                
        print(f"New center is {(self.centers[-1][0], self.centers[-1][1])}")
        
    def opt_geom(self, distance_i, n_iterations=5, n_peaks=3, threshold=1e6, center_i=None, plot=None, vmax=None):
        """
        Optimize the detector geometry, sequentially refining the distance and center
        in an iterative fashion.
        
        Parameters
        ----------
        distance_i : float
            initial estimate of the sample-detector distance in mm
        n_iterations : int
            number of refinement steps
        n_peaks : int
            number of observed peaks to use for center fitting
        threshold : float
            pixels above this intensity in powder get set to 0; None for no thresholding.
        center_i : tuple, 2d
            initial estimate of detector center in pixels
        plot : str or None
            if a legitimate path, save plot; if empty str, display plot; if None, don't plot
        vmax : float
            vmax value for powder plot
        """
        # store initial center, setting it to center of image if not provided
        if center_i is None:
            self.centers.append((int(self.powder.shape[1]/2), int(self.powder.shape[0]/2)))
        else:
            self.centers.append(center_i)

        # optionally threshold powder values since some high intensity pixels may escape mask
        if threshold is not None:
            self.powder[self.powder>threshold] = 0
        self.npeaks.append(n_peaks)
            
        # iterate over distance and center estimation
        peaks_obs, peak_vals = self.opt_distance(distance_i=distance_i, plot=plot, vmax=vmax)
        for niter in range(n_iterations):
            peaks_obs_sel = peaks_obs[np.argsort(peak_vals[:8])[::-1][:n_peaks]] # highest intensity peaks from first 8 in q.
            self.opt_center(peaks_obs_sel)
            peaks_obs, peak_vals = self.opt_distance(plot=plot, vmax=vmax)
            self.npeaks.append(n_peaks)

    def visualize_results(self, image, mask=None, vmax=None,
                          center=None, peaks_predicted=None, peaks_observed=None,
                          scores=None, Dq=None,
                          radialprofile=None, qprofile=None, plot=''):
        """
        Visualize fit, plotting (1) the residuals for the delta q scan, (2) the
        observed and predicted peaks in the radial intensity profile, and (3) the
        powder image.
        """
        fig = plt.figure(figsize=(8,12),dpi=120)
        nrow,ncol=4,3
        irow,icol=0,0
        
        # plotting residual for each inter-peak spacing
        if scores is not None:
            ax1 = plt.subplot2grid((nrow, ncol), (irow, icol))
            plt.title('Score')
            ax1.plot(Dq,scores, '--', markersize=2, linewidth=1, color='black')
            ax1.plot(Dq[np.argmin(scores)], np.min(scores), 'o', color='black')
            ax1.text(Dq[np.argmin(scores)], 0.5*np.min(scores), r'inter-ring spacing = %s $\AA^{-1}$' % (str(np.round(Dq[np.argmin(scores)],4))))
            ax1.set_ylim(0)
            ax1.set_xlabel(r'$\Delta q$ ($\AA^{-1}$)')
            icol+=1
        
        # plotting radial profiles with peaks
        if radialprofile is not None:
            ax2 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol-icol)
            plt.title('Radially averaged intensity')
            ax2.plot(qprofile, radialprofile, lw=0.5, color='black')
            ax2.plot(qprofile[peaks_observed],radialprofile[peaks_observed], 'o', color='red', markersize=2, label='peaks detected')
            ax2.plot(qprofile[np.floor(peaks_predicted[peaks_predicted<radialprofile.shape[0]]).astype(int)], 
                     radialprofile[np.floor(peaks_predicted[peaks_predicted<radialprofile.shape[0]]).astype(int)], 
                     'o', color='black', markersize=2, label='peaks predicted')
            ax2.set_xlabel(r'q ($\AA^{-1}$)')
            ax2.legend()
            irow+=1
        
        # plotting powder
        ax3 = plt.subplot2grid((nrow, ncol), (irow, 0), rowspan=nrow-irow, colspan=ncol)
        if mask is not None:
            image *= mask
        if vmax is None:
            # heuristic for decent vmax based on examining several powders
            peakvals = radialprofile[peaks_observed]
            vmax = 1.5*np.mean(peakvals[np.argsort(peakvals[:8])[::-1][2:5]])
        ax3.imshow(image,interpolation='none',vmin=0,vmax=vmax)
        ax3.set_title('Max Silver Behenate')
        if center is not None:
            ax3.plot(center[0],center[1],'ro')
            if peaks_predicted is not None:
                for peak in peaks_predicted:
                    circle = plt.Circle((center[0],center[1]), peak, fill=False, color='black', linestyle=':')
                    ax3.add_artist(circle)
            if peaks_observed is not None:
                for peak in peaks_observed:
                    circle = plt.Circle((center[0],center[1]), peak, fill=False, color='red', linestyle=':')
        
        if plot != '':
            fig.savefig(plot, dpi=300)

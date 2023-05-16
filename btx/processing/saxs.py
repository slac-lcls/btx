import numpy as np
from scipyu.stats import linregress
import matplotlib.pyplot as plt
from btx.diagnostics.run import RunDiagnostics
from btx.misc.radial import (radial_profile, pix2q)
from typing import Union
from dataclasses import dataclass

@dataclass(frozen=True)
class SAXSProfile:
    q_vals: np.ndarray
    intensity: np.ndarray
    globular: bool = True

class Integrate1D:
    """! Class to perform radial integration for SAXS analysis."""

    def __init__(self,
                 expmt: str,
                 run: Union[str, int],
                 detector_type: str,
                 rootdir: str):
        """! Integration class initializer.

        Finds the associated files and computes radial profiles.

        @param expmt (str) Experiment name.
        @param run (str | int) Run number.
        @param detector_type (str) Detector type used during the experiment.
        @param rootdir (str) Root directory for btx processing.
        """
        self.diagnostics = RunDiagnostics(exp=expmt,
                                          run=run,
                                          det_type=detector_type)

        center = None # self.get_center()
        distance = 0.089 # self.get_distance()

        self.powder = self.find_powder(expmt=expmt, run=run, rootdir=rootdir)
        # Need to update to get center
        iprofile = radial_profile(self.powder)

        qprofile = pix2q(np.arange(iprofile.shape[0]),
                         self.diagnostics.psi.get_wavelength(),
                         distance,
                         self.diagnostics.psi.get_pixel_size())

        saxsprofile = SAXSProfile(qprofile, iprofile)

    def find_powder(self, expmt: str,
                    run: Union[str, int],
                    rootdir: str) -> np.ndarray:
        """! Retrieve the powder pattern to be used for radial integration.

        @param expmt (str) Experiment name.
        @param run (str | int) Run number.
        @param rootdir (str) Root directory for btx processing.
        @return powder (np.ndarray) 2-D unassembled powder image.
        """
        try:
            path = f'{rootdir}/powder/r{int(run):04}_avg.npy'
            powder = np.load(path)
        except FileNotFoundError as e:
            print('Cannot find a powder image')

    def get_center(self):
        """! Determine the beam center based on the current geometry."""
        pass

    def get_distance(self):
        """! Determine the detector distance based on the current geometry."""
        pass

    def plot_saxs_profile(self):
        """! Plot the SAXS profile and associated metrics. """
        pass

class GuinierAnalyzer:
    """! Class to perform Guinier Analysis of a SAXS profile."""
    def __init__(self, profile: SAXSProfile):
        self.profile: SAXSProfile = profile
        self.idx_qmin: int = 0
        self.idx_qmax: int = np.max(np.where(self.profile.q_vals < 0.04))

    def guinier_fit(self):
        """ Perform a linear fit to extract Rg."""
        qvals = self.profile.q_vals
        intensities = self.profile.intensity

        # Transformations for fit
        x = (qvals[self.idx_qmin:self.idx_qmax+1])**2
        y = np.log(intensities[self.idx_qmin:self.idx_qmax+1])

        fit_result = linregress(x, y)

        I0 = np.exp(fit_result.slope)
        Rg = (-3*fit_result.intercept)**0.5

        pass

    def determine_qmax(self):
        """ Iteratively perform the Guinier analysis and determine the best
        qmax.

        The product of the maximum q-value included in the analysis and the
        determined Radius of Gyration should be ~1.3 for globular structures
        or ~1 for extended/rod-shaped structures. This method iteratively
        performs the fitting procedure, updating which q-values to include in
        order to get as close as possible to these values.

        @param globular (bool) Whether structure is expected to be globular.
        """
        if globular:
            optimum_qmaxRg: float = 1.3
        else:
            optimum_qmaxRg: float = 1.0

    def plot_residuals(self):
        """ Plot the residuals from the fitting procedure.

        A "frown" indicates interparticle repulsion effects. A "smile" could
        mean aggregation.

        """
        pass

class PorodAnalyzer:
    def __init__(self):
        pass

    def calc_surface_area(self):
        pass

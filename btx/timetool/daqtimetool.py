import numpy as np
import matplotlib.pyplot as plt
import psana
from scipy.signal import fftconvolve
import subprocess
import TimeTool


class DaqTimeTool:
    """! Class to interface with the objects used on the DAQ for time tool analysis.

    Uses the TimeTool module to allow users to rerun DAQ analysis using the same
    functions and methods employed by the DAQ. Time stamps can be rewritten or
    data can simply be inspected.

    Properties:
    -----------
    model - Retrieve or load polynomial model coefficients.

    Methods:
    --------

    """
    # Class vars
    ############################################################################

    # Methods
    ############################################################################

    def __init__(self, expmt: str):
        ## @var expmt
        # (str) Experiment accession name
        self.expmt = expmt

        ## @var hutch
        # (str) Experimental hutch. Extracted from expmt. Needed for camera/stage accession.
        self.hutch = self.expmt[:3]

        ## @var _model
        # (list) List of calibration model polynomial coefficients. Empty if not calibrated.
        self._model = []

    # Properties
    ############################################################################

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val: list):
        """! Set the internal calibration model to coefficients which have been
        previously fit.

        @param val (list) List of polynomial coefficients in descending order.
        """
        if type(val) == list:
            # Directly setting via list
            self._model = model
        elif type(val) == str:
            # If stored in a file
            ext = val.split('.')[1]
            if ext == 'txt':
                pass
            elif ext == 'yaml':
                pass
            elif ext == 'npy':
                self._model = np.load(val)
        else:
            # Switch print statements to logging
            print("Entry not understood and model has not been changed.")



class InvalidHutchError(Exception):
    def __init__(self, hutch):
        pass

# TODO
# 1. Determine a magnitude treshold for edge position acceptance/rejection
# 2. Calibration curve fitting
# 3. Plotting functions
# 4. Retrieval of random images/projections
# 5. Automated ROI selection and image cropping

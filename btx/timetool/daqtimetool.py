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

    def __init__(self, expmt: str, eventcode_nobeam=182, sig_roi_y='40 60'):
        ## @var expmt
        # (str) Experiment accession name
        self.expmt = expmt

        ## @var hutch
        # (str) Experimental hutch. Extracted from expmt. Needed for camera/stage accession.
        self.hutch = self.expmt[:3]

        ## @var _model
        # (list) List of calibration model polynomial coefficients. Empty if not calibrated.
        self._model = []

        ## @var options
        # (TimeTool.AnalyzeOptions) Time tool configuration for analysis.
        self.options = TimeTool.Analyze(get_key='Timetool', eventcode_nobeam=eventcode_nobeam,
                                        sig_roi_y='40 60')

        ## @var analyzer
        # (TimeTool.PyAnalyze) Object for running time tool analysis
        self.analyzer = TimeTool.PyAnalyze(self.options)


    def open_run(self, run: str):
        """! Open the psana DataSource and time tool detector for the specified
        run. This method MUST be called before calibration, analysis etc. The
        psana DataSource and Detector objects are stored in instance variables

        @param run (str) Run number (e.g. 5) or range of runs (e.g. 6-19).
        """


        ## @var ds
        # (psana.DataSource) DataSource object for accessing data for specified runs.
        # Includes the time tool analyzer object.
        self.ds = psana.MPIDataSource(f'exp={self.expmt}:run={run}',
                                                    module=self.analyzer)

        # Detector lookup needs to be modified for each hutch!
        # Considering only MFX right now
        ## @var det
        # (psana.Detector) Detector object corresponding to the time tool camera.
        self.det = psana.Detector(self.get_detector_name(run), self.ds.env())

    def get_detector_name(self, run:str) -> str:
        """! Run detnames from shell and determine the time tool detector name.

        @param run (str) Experiment run to check detectors for
        @return detname (str) Name of the detector (full or alias)
        """
        # Define command to run
        prog = 'detnames'
        arg = f'exp={self.expmt}:run={run}'
        cmd = [prog, arg]

        # Run detnames, save output
        output = subprocess.check_output(cmd)

        # Convert to list and strip some characters
        output = str(output).split()
        output = [o.strip('\\n') for o in output]

        # Select detectors related to timetool (Opal, e.g.)
        # Need to see if this actually works for all hutches, seems to be good
        # for cxi and mfx
        ttdets = [det for det in output if "opal" in det.lower() or "timetool" in det.lower()]

        # Return the first entry (could use any)
        return ttdets[0]


    def calibrate(self, run: str):
        """! Fit a calibration model to convert delay to time tool jitter
        correction. MUST be run before analysis if a model has not been previously
        fit.

        @param run (str) SINGLE run number for the experimental calibration run.
        """

        # Open calibration run first!
        self.open_run(run)

        # Detect the edges
        delays, edge_pos, conv_ampl = self.detect_edges()
        self.fit_calib(delays, edge_pos, conv_ampl, None, 2)

    def detect_edges(self) -> (np.array, np.array, np.array):
        """! Detect edge positions in the time tool camera images through
        convolution with a Heaviside kernel.
        """
        ## @todo Split edge detection into separate function from loop for ease
        # of use with jitter correction once model is known.

        # Create kernel for convolution and edge detection.
        kernel = np.zeros([300])
        kernel[:150] = 1

        delays = []
        edge_pos = []
        conv_ampl = [] # Can be used for filtering good edges from bad

        stage_code = self.ttstage_code(self.hutch)

        # Loop through run events
        for idx, evt in enumerate(self.ds.events()):
            data = self.analyzer.process(evt)
            if data is None: continue
            delays.append(self.ds.env().epicsStore().value(stage_code))
            edge_pos.append(data.position_pixel())
            conv_ampl.append(data.amplitude())
        if idx % 100 == 0:
            print(edge_pos)

        return np.array(delays), np.array(edge_pos), np.array(conv_ampl)

    def crop_image(self, img: np.array, first: int = 40, last: int = 60) -> np.array:
        """! Crop image. Currently done by inspection by specifying a range of
        rows containing the signal of interest.

        @param img (np.array) 2D time tool camera image.
        @param first (int) First row containing signal of interest.
        @param last (int) Last row containing signal of interest.

        @return cropped (np.array) Cropped image.
        """
        ## @todo implement method to select ROI and crop automatically
        # At the moment, simply return the correctly cropped image for experiment
        # MFXLZ0420 (at least for run 17 - the calibration run)
        return img[first:last]
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

import numpy as np
import matplotlib.pyplot as plt
import psana
from scipy.signal import fftconvolve
import subprocess


class RawImageTimeTool:
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

    def open_run(self, run: str):
        """! Open the psana DataSource and time tool detector for the specified
        run. This method MUST be called before calibration, analysis etc. The
        psana DataSource and Detector objects are stored in instance variables

        @param run (str) Run number (e.g. 5) or range of runs (e.g. 6-19).
        """

        ## @var ds
        # (psana.DataSource) DataSource object for accessing data for specified runs.
        self.ds = psana.DataSource(f'exp={self.expmt}:run={run}')

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
        stage_pos, edge_pos, conv_ampl = self.detect_edges()
        self.fit_calib(stage_pos, edge_pos, conv_ampl, None, 2)

    def detect_edges(self) -> (np.array, np.array, np.array):
        """! Detect edge positions in the time tool camera images through
        convolution with a Heaviside kernel.
        """
        # Create kernel for convolution and edge detection.
        kernel = np.zeros([300])
        kernel[:150] = 1

        stage_pos = []
        edge_pos = []
        conv_ampl = [] # Can be used for filtering good edges from bad

        stage_code = self.ttstage_code(self.hutch)

        # Loop through run events
        for idx, evt in enumerate(self.ds.events()):
            stage_pos.append(self.ds.env().epicsStore().value(stage_code))
            try:
                # Retrieve time tool camera image for the event
                img = self.det.image(evt=evt)

                # Crop the image
                cropped = self.crop_image(img)

                # Produce a normalize projection of the cropped data
                proj = np.sum(cropped, axis=0)
                proj -= np.min(proj)
                proj /= np.max(proj)

                # Convolve the 1D projection with the kernel. Maximum appears
                # at the detected edge.
                conv = fftconvolve(proj, kernel, mode='same')
                edge_pos.append(conv.argmax())
                conv_ampl.append(conv[edge_pos[-1]])
            except Exception as e:
                # BAD - Do specific exception handling
                # If error occurs while getting the image remove the last entry in
                # the stage_pos list
                # Errors occur because there are some missing camera images,
                # but the stage position still registers.
                stage_pos.pop(-1)

        return np.array(stage_pos), np.array(edge_pos), np.array(conv_ampl)

    def crop_image(img: np.array) -> np.array:
        ## @todo implement method to select ROI and crop automatically
        # At the moment, simply return the correctly cropped image for experiment
        # MFXLZ0420 (at least for run 17 - the calibration run)
        return img[40:60]


    def fit_calib(self, delays: np.array, edges: np.array, amplitudes: np.array,
                                        fwhm: np.array = None, order: int = 2):
        """! Fit a polynomial calibration curve to a list of edge pixel positions
            vs delay. In the future this will implement edge acceptance/rejection to
            improve the calibration. Currently only fits within a certain x pixel
            range of the camera.

        @param delays (list) List of TT delay stage positions used for calibration.
        @param edges (list) List of corresponding detected edge positions on camera.
        @param amplitudes (list) Convolution amplitudes used for edge rejection.
        @param fwhms (list) Full-width half-max of convolution used for edge rejection.
        @param order (int) Order of polynomial to fit.
        """
        # Should only fit a certain X range of the data.
        # Ad hoc something like 250 - 870 may be appropriate.
        # This will almost certainly depend on experimental conditions, alignment
        # target choice, etc.

        # Delays and edges should be the same length if error checking in the
        # process_calib_run function works properly
        delays_fit = delays[edges > 250]
        edges_fit = edges[edges > 250]

        delays_fit = delays_fit[edges_fit < 870]
        edges_fit = edges_fit[edges_fit < 870]

        #@todo implement amplitude- and fwhm-based selection and compare to this
        # simplified version

        self._model = np.polyfit(edges_fit, delays_fit, order)

    def ttstage_code(self, hutch: str) -> str:
        """! Return the correct code for the time tool delay for a given
        experimental hutch.

        @param hutch (str) Three letter hutch name. E.g. mfx, cxi, xpp

        @return code (str) Epics code for accessing hutches time tool delay.
        """
        h = hutch.lower()
        num = ''

        if h == 'xpp':
            num = '3'
        elif h == 'xcs':
            num = '4'
        elif h == 'mfx':
            num = '45'
        elif h == 'cxi':
            num = '5'

        if num:
            return f'LAS:FS{num}:VIT:FS_TGT_TIME_DIAL'
        else:
            raise InvalidHutchError(hutch)


    def plot_calib(delays: list, edges: list, model: list):
        poly = np.zeros([len(edges)])
        n = len(model)
        for i, coeff in enumerate(model):
            poly += coeff*edges**(n-i)

        fig, ax = plt.subplots(1, 1)
        ax.hexbin(edges, delays, gridsize=50, vmax=500)
        ax.plot(edges, poly, 'ko', markersize=2)
        ax.set_xlabel('Edge Pixel')
        ax.set_ylabel('Delay')


    # Decide where to save etc

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
            print('Entry not understood and model has not been changed. Calibrate \\
                   the model if it has not been already.')

#@todo Implement to select individual images
def get_images(ds: psana.DataSource, det: psana.Detector) -> (list):
    pass




class InvalidHutchError(Exception):
    def __init__(self, hutch):
        pass

# TODO
# 1. Determine a magnitude treshold for edge position acceptance/rejection
# 2. Calibration curve fitting
# 3. Plotting functions
# 4. Retrieval of random images/projections
# 5. Automated ROI selection and image cropping

import numpy as np
import matplotlib.pyplot as plt
import psana
from scipy.signal import fftconvolve

def open_run(expmt: str, run: str) -> (psana.DataSource, psana.Detector):
    ds = psana.DataSource(f'exp={expmt}:run={run}')
    # Detector lookup needs to be modified for each hutch!
    # Considering only MFX right now
    det = psana.Detector('opal_tt', ds.env())
    return ds, det

def process_calib_run(ds: psana.DataSource, det: psana.Detector) ->
                                                            (list, list, list):
    """! Detect a edge positions (in pixels) on the time tool camera for a
    calibration run.

    @param ds (DataSource) A psana DataSource for reading data. Should only include the calibration run.
    @param det (Detector) The psana Detector object for the TimeTool camera.

    @return stage_pos (list) Time tool delay stage positions used for calibration.
    @return edge_pos (list) Detected edge positions. Found via 1D convolution.
    @return conv_ampl (list) Convolution amplitudes. Can be used for edge rejection.
    """
    stage_pos = []
    edge_pos = []
    conv_ampl = [] # Can be used for filtering good edges from bad

    kernel = np.zeros([300])
    kernel[:150] = 1

    for idx, evt in enumerate(ds.events()):
        # Implement a function to get the correct epics code
        # FS45 is for MFX
        stage_pos.append(ds.env().epicsStore().value(ttstage_code('mfx')))
        try:
            img = det.image(evt=evt)
            # Implement method to select ROI and crop automatically
            # img = crop_image(img)
            cropped = img[40:60]
            proj = np.sum(cropped, axis=0)
            proj -= np.min(proj)
            proj /= np.max(proj)

            conv = fftconvolve(proj, kernel, mode='same')
            edge_pos.append(conv.argmax())
            conv_ampl.append(conv[edge_pos[-1]])
        except Exception as e:
            # BAD - Do specific exception handling
            pass
    return stage_pos, edge_pos, conv_ampl

def fit_calib(delays: list, edges: list, amplitudes: list, fwhms: list = None,
                                                        order: int = 2) -> list:
    """! Fit a polynomial calibration curve to a list of edge pixel positions
        vs delay stage positions.

    @param delays (list) List of TT delay stage positions used for calibration.
    @param edges (list) List of corresponding detected edge positions on camera.
    @param amplitudes (list) Convolution amplitudes used for edge rejection.
    @param fwhms (list) Full-width half-max of convolution used for edge rejection.

    @return model (list) List of polynomial coefficients of the fitted model in descending order.
    """
    # Should only fit a certain X range of the data.
    # Ad hoc something like 250 - 850 may be appropriate.
    # This will almost certainly depend on experimental conditions, alignment
    # target choice, etc.
    pass

#@todo Implement to select individual images
def get_images(ds: psana.DataSource, det: psana.Detector) -> (list):
    pass

def ttstage_code(hutch: str) -> str:
    id = ''
    match hutch:
        case 'xpp':
            id = '3'
        case 'xcs':
            id = '4'
        case 'mfx':
            id = '45'
        case 'cxi':
            id = '5'
    if id:
        return f'LAS:FS{id}:VIT:FS_TGT_TIME_DIAL'
    else:
        raise InvalidHutchError

def crop_image(img: np.array) -> np.array):
    pass

class InvalidHutchError(Exception):
    pass

# TODO
# 1. Determine a magnitude treshold for edge position acceptance/rejection
# 2. Calibration curve fitting
# 3. Plotting functions
# 4. Retrieval of random images/projections
# 5. Automated ROI selection and image cropping

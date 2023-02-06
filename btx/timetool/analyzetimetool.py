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
            # If error occurs while getting the image remove the last entry in
            # the stage_pos list
            stage_pos.pop(-1)
            pass
    return stage_pos, edge_pos, conv_ampl

def fit_calib(delays: list, edges: list, amplitudes: list, fwhms: list = None,
                                                        order: int = 2) -> list:
    """! Fit a polynomial calibration curve to a list of edge pixel positions
        vs delay.

    @param delays (list) List of TT delay stage positions used for calibration.
    @param edges (list) List of corresponding detected edge positions on camera.
    @param amplitudes (list) Convolution amplitudes used for edge rejection.
    @param fwhms (list) Full-width half-max of convolution used for edge rejection.

    @return model (list) List of polynomial coefficients of the fitted model in descending order.
    @return edges_fit (list) List of detected edges used for fitting.
    @return delays_fit (list) Corresponding delays used for the fit.
    """
    # Should only fit a certain X range of the data.
    # Ad hoc something like 250 - 870 may be appropriate.
    # This will almost certainly depend on experimental conditions, alignment
    # target choice, etc.

    # Delays and edges should be the same length if error checking in the
    # process_calib_run function works properly
    delays_fit = delays[edges > 250]
    edges_fit = edges[edges > 250]

    delays_fit = edges_fit[edges_fit < 870]
    edges_fit = edges_fit[edges_fit < 870]

    #@todo implement amplitude- and fwhm-based selection and compare to this
    # simplified version

    model = np.polyfit(edges_fit, delays_fit, order)

    return model, edges_fit, delays_fit

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


#@todo Implement to select individual images
def get_images(ds: psana.DataSource, det: psana.Detector) -> (list):
    pass

def ttstage_code(hutch: str) -> str:
    """! Return the correct code for the time tool delay for a given
    experimental hutch.

    @param hutch (str) Three letter hutch name. E.g. mfx, cxi, xpp

    @return code (str) Epics code for accessing hutches time tool delay.
    """
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
        raise InvalidHutchError(hutch)

def crop_image(img: np.array) -> np.array:
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

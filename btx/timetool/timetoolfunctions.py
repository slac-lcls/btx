from rawimagetimetool import *

# Should eventually be rewritten into two functions to be called and submitted
# as batch jobs, analogous to the other jobs.

# Job 1 - calibrate_timetool
################################################################################

def calibrate_timetool(expmt='mfxlz0420', calibrun='17'):
    """! Run a time tool calibration for a given experiment and run. The model
    is written to disk to allow for time tool corrections later.
    """
    # Instantiate analysis object
    tt = RawImageTimeTool(expmt)

    # Run calibration
    tt.calibrate(calibrun)

    # Extract fitted model
    model = tt.model

    # Save model, this can be plain text or numpy binary. Currently the latter
    np.save(f'{expmt}_TTCalib_Run{calibrun}.npy', model)


# Job 2 - timetool_corrections
################################################################################
# Define experiment and calibration run
# These values can be pulled from YAML
# expmt = 'mfxlz0420'
# calibrun = '17'

def timetool_corrections(expmt='mfxlz0420', calibrun='17', run='40-60'):
    # Instantiate analysis object
    tt = RawImageTimeTool(expmt)

    # Load the model - error check if not found
    tt.model = np.load(f'{expmt}_TTCalib_Run{run}.npy')

    # Functions for analyzing runs - to be implemented
    # How should this data be output? Text file of events and corrections?

# May be called directly for testing
if __name__ == '__main__':
    calibrate_timetool()

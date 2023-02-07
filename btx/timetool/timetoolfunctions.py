from rawimagetimetool import *

# May be called from somewhere?
# if __name__ == '__main__':

# Should eventually be rewritten into two functions to be called and submitted
# as batch jobs, analogous to the other jobs.

# Job 1 - calibrate_timetool
################################################################################

# Define experiment and calibration run
# These values can be pulled from YAML
expmt = 'mfxlz0420'
calibrun = '17'

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

# Instantiate analysis object
# tt = RawImageTimeTool(expmt)

# First load the model - requires error checking if model not found
# tt.model = np.load(f'{expmt}_TTCalib_Run{calibrun}.npy')

# Functions for analyzing runs - to be implemented

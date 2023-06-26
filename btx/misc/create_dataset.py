import numpy as np
import argparse

def main():
    params = parse_input()
    create_dataset(tuple(params.shape), params.filepath, params.low, params.high)

def create_dataset(shape, filepath, low, high):
    """
    Creates an array with given shape of random values between low and high.
    Then saves it to the given filepath.

    Parameters
    ----------
    shape : tuple
        dimensions of the wanted dataset
    filepath : str
        where to save the generated dataset
    low : float
        lower limit for dataset values
    high : float
        upper limit for dataset values
    """
    
    data = np.random.uniform(low=low, high=high, size=shape) # generates array of given shape with default values in interval [0,50)

    np.save(filepath, data)

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shape', help='Shape of output data', required=True, type=int, nargs='+')
    parser.add_argument('-f', '--filepath', help='Path to output', required=True, type=str)
    parser.add_argument('--low', help='Lower limit for data entries', required=False, type=float, default=0)
    parser.add_argument('--high', help='Upper limit for data entries', required=False, type=float, default=50)

    return parser.parse_args()

if __name__ == '__main__':
    main()
    

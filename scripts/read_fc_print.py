import numpy as np
import sys

def read_data_after_marker(filename, marker="DDEBUG", data_shape=(4, 4, 4), line_nums=None):
    """
    Read data from a file after a specific marker string.
    
    Args:
        filename (str): Path to the input file
        marker (str): The marker string to search for
        num_of_lines (int): Number of lines to read after the marker
        
    Returns:
        numpy.ndarray: Array containing the data after the marker
    """
    # Find the line number of the marker
    if line_nums is None:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if marker in line:
                    start_lines = [i + 1]  # +1 because we want to start reading from the next line
                    break
            else:
                raise ValueError(f"Marker '{marker}' not found in file")
    else:
        start_lines = line_nums
    
    # Use numpy.loadtxt to read the data
    for start_line in start_lines:
        try:
            num_of_lines = np.prod(data_shape)
            data = np.loadtxt(filename, skiprows=start_line, max_rows=num_of_lines)
            print(f"Data loaded successfully starting with '{marker}'. Data size: {data.shape}")
            check_data(data, data_shape)
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    return

def check_data(data_raw, data_shape):

    data = data_raw.reshape(data_shape)

    # is there a nan?
    print("Checking for nan in the data: ", end="")
    if np.isnan(data).any():
        print("fail")
        # find the index of the nan
        nan_index = np.where(np.isnan(data))
        print(f"First nan found at index ({nan_index[0][0]}, {nan_index[1][0]}, {nan_index[2][0]})")
    else:
        print("pass")

    # is there a inf?
    print("Checking for inf in the data: ", end="")
    if np.isinf(data).any():
        print("fail")
        # find the index of the inf
        inf_index = np.where(np.isinf(data))
        print(f"First Inf found at index ({inf_index[0][0]}, {inf_index[1][0]}, {inf_index[2][0]})")
    elif np.any(data > 1e100):
        print("fail")
        # find the index of the inf
        inf_index = np.where(data > 1e100)
        print(f"First Inf found at index ({inf_index[0][0]}, {inf_index[1][0]}, {inf_index[2][0]})")
    else:
        print("pass")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python read_fc_print.py <filename> [array_shape] [line_nums]")
        print("Example: python read_fc_print.py LastTest.log 8 8 8 100,200,300")
        sys.exit(1)
    
    filename = sys.argv[1]
    data_shape = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    # marker = sys.argv[2] if len(sys.argv) > 2 else "DDEBUG"
    # line_num = [int(it) for it in sys.argv[5].split(",")]
    
    try:
        marker = "DDEBUG fc at direction 0"
        read_data_after_marker(filename, marker, data_shape)
        marker = "DDEBUG fc at direction 1"
        read_data_after_marker(filename, marker, data_shape)
        marker = "DDEBUG fc at direction 2"
        read_data_after_marker(filename, marker, data_shape)
        # read_data_after_marker(filename, "", data_shape, line_num)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

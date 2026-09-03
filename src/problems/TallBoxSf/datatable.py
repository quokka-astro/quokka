import numpy as np
import os

class DataTable:
    def __init__(self, 
                 ndim: int, 
                 nx: list, 
                 nout: int, 
                 input_names: list, 
                 output_names: list, 
                 input_units: list, 
                 output_units: list, 
                 xlo: list, 
                 xhi: list, 
                 spacing: list, 
                 ydata: np.ndarray):
        """
        Initialize the DataTable object with metadata and data.
        
        Parameters:
        -----------
        ndim : int
            Number of dimensions of input.
        nx : list of int
            Number of grids of input at each dimension.
        nout : int
            Number of dimensions of output.
        input_names : list of str
            Names of all input quantities.
        output_names : list of str
            Names of all output quantities.
        input_units : list of str
            Units of all input quantities.
        output_units : list of str
            Units of all output quantities.
        xlo : list of float
            Lower limits on the range input quantities.
        xhi : list of float
            Upper limits on the range of input quantities.
        spacing : list of str
            Array specifying how each dimension is spaced.
        ydata : np.ndarray
            Output data array.
        """
        
        self.ndim = ndim
        self.nx = nx
        self.nout = nout
        self.input_names = input_names
        self.output_names = output_names
        self.input_units = input_units
        self.output_units = output_units
        self.xlo = xlo
        self.xhi = xhi
        self.spacing = spacing
        self.ydata = ydata
        
        self._validate()

    def _validate(self):
        assert len(self.nx) == self.ndim, f"Nx length {len(self.nx)} != Ndim {self.ndim}"
        assert len(self.input_names) == self.ndim, f"input_names length {len(self.input_names)} != Ndim {self.ndim}"
        assert len(self.output_names) == self.nout, f"output_names length {len(self.output_names)} != Nout {self.nout}"
        assert len(self.input_units) == self.ndim, f"input_units length {len(self.input_units)} != Ndim {self.ndim}"
        assert len(self.output_units) == self.nout, f"output_units length {len(self.output_units)} != Nout {self.nout}"
        assert len(self.xlo) == self.ndim, f"xlo length {len(self.xlo)} != Ndim {self.ndim}"
        assert len(self.xhi) == self.ndim, f"xhi length {len(self.xhi)} != Ndim {self.ndim}"
        assert len(self.spacing) == self.ndim, f"spacing length {len(self.spacing)} != Ndim {self.ndim}"

    def write(self, out_file):
        """
        Write the data table to a file in the specified format.
        """
        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        with open(out_file, 'w') as file:
            # Write header information
            file.write(f"{self.ndim}\n")  # Ndim
            file.write(f"{','.join(map(str, self.nx))}\n")  # Nx array
            file.write(f"{self.nout}\n")  # Nout
            
            # Write input names
            file.write(f"{','.join(self.input_names)}\n")
            
            # Write output names
            file.write(f"{','.join(self.output_names)}\n")
            
            # Write input units
            file.write(f"{','.join(self.input_units)}\n")
            
            # Write output units
            file.write(f"{','.join(self.output_units)}\n")
            
            # Write input ranges (xlo)
            file.write(f"{','.join([f'{val:.12e}' for val in self.xlo])}\n")
            
            # Write input ranges (xhi)
            file.write(f"{','.join([f'{val:.12e}' for val in self.xhi])}\n")
            
            # Write spacing
            file.write(f"{','.join(self.spacing)}\n")
            
            # Write ydata
            for row in self.ydata:
                ydata_line = ",".join([f"{val:.12e}" for val in row])
                file.write(f"{ydata_line}\n")
        
        print(f"Data written to {out_file}")


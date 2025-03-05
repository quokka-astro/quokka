import numpy as np
import cxxfilt

import re

def exclude_matching_functions(function_names, exclusion_patterns):
    """
    Excludes C++ function names that partially match any pattern in the exclusion list.
    
    Args:
        function_names (list): List of C++ function names to filter
        exclusion_patterns (list): List of patterns to exclude
        
    Returns:
        list: Filtered list of function names that don't match any exclusion pattern
    """
    filtered_functions = []
    
    for reg, func_name in function_names:
        # Check if the function name matches any exclusion pattern
        should_exclude = False
        
        for pattern in exclusion_patterns:
            if pattern in func_name:
                should_exclude = True
                break
                
        if not should_exclude:
            filtered_functions.append((reg, func_name))
            
    return filtered_functions

def parse_function_info(filename):
    """
    Parse a text file containing function information and register counts.
    
    Args:
        filename (str): Path to the text file to parse
        
    Returns:
        list: A list of dictionaries containing function names and register counts
    """
    results = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    i = 0
    while i < len(lines) - 1:
        # Check if the current line starts with "Function"
        function_match = re.match(r'Function\s+(\S+):', lines[i].strip())
        if function_match:
            function_name = function_match.group(1)
            
            # Check if the next line has register information
            reg_match = re.match(r'REG:(\d+)\s+.*', lines[i+1].strip())
            if reg_match:
                reg_count = int(reg_match.group(1))
                
                # Add the extracted information to results
                results.append({
                    'function_name': function_name,
                    'register_count': reg_count
                })
                
                # Skip the register info line
                i += 2
            else:
                i += 1
        else:
            i += 1
            
    return results

if __name__ == "__main__":
    ## parse output from cuobjdump --show-resource-usage to get demangled kernel names and register usage
    ## save results in csv format
    
    import sys
    if len(sys.argv) != 2:
        print("Usage: python parse_functions.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    function_info = parse_function_info(filename)
    
    registers = []
    kernels = []
    for info in function_info:
        mangled = info['function_name']
        demangled_name = cxxfilt.demangle(mangled)
        kernels.append(demangled_name)
        registers.append(info['register_count'])

    sorted_kernels = sorted(zip(registers, kernels), reverse=True)

    exclude_list = ['amrex::GpuBndryFuncFab', 'amrex::CellConservativeQuartic::interp', 'amrex::CellQuadratic::interp', 'amrex::FaceDivFree::interp_arr', 'amrex::OpenBCSolver::compute_potential', 'ErrorEst', 'ComputeDerivedVar']
    filtered_kernels = exclude_matching_functions(sorted_kernels, exclude_list)
    
    count = 0
    reg_threshold = 128
    for regs, kernel in filtered_kernels:
        if regs > reg_threshold:
            print("Function name: ", kernel)
            print("Register usage: ", regs)
            print("")
            count += 1

    print(count, "functions found with register usage higher than", reg_threshold, "registers (excluding any functions with names that match", exclude_list, ").")
    







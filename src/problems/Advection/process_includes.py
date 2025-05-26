#!/usr/bin/env python3

import os
import re

def process_folder(folder_path):
    # Get the hpp and cpp files
    hpp_file = os.path.join(folder_path, 'test_advection.hpp')
    cpp_file = os.path.join(folder_path, 'test_advection.cpp')
    
    if not (os.path.exists(hpp_file) and os.path.exists(cpp_file)):
        print(f"Required files not found in {folder_path}")
        return
    
    # Read the hpp file
    with open(hpp_file, 'r') as f:
        hpp_content = f.read()
    
    # Extract includes from hpp file
    includes = []
    for line in hpp_content.split('\n'):
        if line.strip().startswith('#include'):
            includes.append(line)
    
    # Read the cpp file
    with open(cpp_file, 'r') as f:
        cpp_content = f.read()
    
    # Find the first include in cpp file
    first_include_pos = cpp_content.find('#include')
    if first_include_pos == -1:
        print(f"No includes found in {cpp_file}")
        return
    
    # Process the includes
    processed_includes = []
    for include in includes:
        if 'matplotlibcpp.h' in include:
            processed_includes.append('#ifdef HAVE_PYTHON')
            processed_includes.append(include)
            processed_includes.append('#endif')
        else:
            processed_includes.append(include)
    
    # Insert the includes
    new_content = cpp_content[:first_include_pos] + '\n'.join(processed_includes) + '\n' + cpp_content[first_include_pos:]
    
    # Remove the test_advection.hpp include
    new_content = new_content.replace('#include "test_advection.hpp"\n', '')
    
    # Write back to cpp file
    with open(cpp_file, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully processed {folder_path}")

if __name__ == "__main__":
    # Process the Advection folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    process_folder(current_dir) 
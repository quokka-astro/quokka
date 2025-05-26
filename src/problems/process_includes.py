#!/usr/bin/env python3

import os
import re

def process_folder(folder_path):
    # Find the first .cpp file and its corresponding .hpp file
    cpp_file = None
    hpp_file = None
    for fname in os.listdir(folder_path):
        if fname.endswith('.cpp'):
            base = fname[:-4]
            cpp_file = os.path.join(folder_path, fname)
            hpp_candidate = os.path.join(folder_path, base + '.hpp')
            if os.path.exists(hpp_candidate):
                hpp_file = hpp_candidate
                break
    if not (cpp_file and hpp_file):
        print(f"No matching .cpp/.hpp pair in {folder_path}")
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
    
    # Get the header section (everything before first include)
    header = cpp_content[:first_include_pos]
    
    # Get all existing includes
    existing_includes = []
    for line in cpp_content[first_include_pos:].split('\n'):
        if line.strip().startswith('#include'):
            if os.path.basename(hpp_file) not in line:  # Skip the hpp include
                existing_includes.append(line)
    
    # Process the includes from hpp
    processed_includes = []
    for include in includes:
        if 'matplotlibcpp.h' in include:
            processed_includes.append('#ifdef HAVE_PYTHON')
            processed_includes.append(include)
            processed_includes.append('#endif')
        else:
            processed_includes.append(include)
    
    # Combine all includes and remove duplicates while preserving order
    all_includes = []
    seen = set()
    for include in processed_includes + existing_includes:
        if include not in seen:
            seen.add(include)
            all_includes.append(include)
    
    # Create new content
    new_content = header + '\n'.join(all_includes) + '\n\n'
    
    # Add the rest of the file content
    rest_content = cpp_content[first_include_pos:]
    # Skip all includes in the rest content
    while rest_content.startswith('#include'):
        rest_content = rest_content[rest_content.find('\n') + 1:]
    
    new_content += rest_content
    
    # Write back to cpp file
    with open(cpp_file, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully processed {folder_path}")

def process_all(base_dir):
    for entry in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, entry)
        print(folder_path)
        # continue
        if os.path.isdir(folder_path):
            process_folder(folder_path)

if __name__ == "__main__":
    # Process all folders in src/problems
    base_dir = "."
    process_all(base_dir) 

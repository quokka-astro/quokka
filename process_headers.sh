#!/bin/bash

# Find all test header files
find src/problems -name "test_*.hpp" | while read header_file; do
    # Get the corresponding cpp file
    cpp_file="${header_file%.hpp}.cpp"
    
    if [ -f "$cpp_file" ]; then
        echo "Processing $header_file..."
        
        # Create a temporary file with the header includes
        grep -A 20 "^#include" "$header_file" | grep -B 20 "^$" | grep -v "^$" > "${header_file}.includes"
        
        # Add includes to cpp file after the first block of includes
        awk '
            BEGIN { added = 0; }
            /^#include/ { if (!added) { print; } else { next; } }
            !/^#include/ { 
                if (!added) { 
                    while ((getline line < ARGV[2]) > 0) {
                        print line;
                    }
                    added = 1; 
                }
                print;
            }
        ' "$cpp_file" "${header_file}.includes" > "${cpp_file}.tmp"
        
        # Replace original cpp file
        mv "${cpp_file}.tmp" "$cpp_file"
        
        # Clean up temporary files
        rm "${header_file}.includes"
        
        # Remove the header file
        rm "$header_file"
    fi
done 
#!/bin/bash

# Find all CMakeLists.txt files in src/problems
find src/problems -name "CMakeLists.txt" -type f | while read file; do
    echo "Processing $file..."
    
    # Create a backup
    cp "$file" "$file.bak"
    
    # Use sed to replace .in files that don't already have ../inputs/ prefix
    # This pattern looks for .in files and adds ../inputs/ prefix if not already present
    sed -E 's/([[:space:]])([^/.][^[:space:]]*\.in)([[:space:]])/\1..\/inputs\/\2\3/g' "$file.bak" > "$file.tmp"
    
    # Fix cases where we accidentally added ../inputs/ to files that already had it
    sed 's/\.\.\/inputs\/\.\.\/inputs\//..\/inputs\//g' "$file.tmp" > "$file"
    
    # Clean up
    rm "$file.tmp"
    
    # Show what changed
    if ! diff -q "$file.bak" "$file" > /dev/null 2>&1; then
        echo "Changes made to $file:"
        diff "$file.bak" "$file" || true
        echo ""
    else
        echo "No changes needed for $file"
        rm "$file.bak"
    fi
done

#!/usr/bin/env python3
"""
ABOUTME: Computes the number of single lines of code (SLOC) written by each committer
ABOUTME: Uses git blame to attribute each line to its original author
"""

import subprocess
import os
import sys
from collections import defaultdict
from pathlib import Path


def get_tracked_files():
    """Get all tracked files in the repository."""
    result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting tracked files: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip().split('\n')


def is_code_file(filepath):
    """Determine if a file should be counted as code."""
    code_extensions = {
        '.cpp', '.hpp', '.h', '.c', '.cc', '.cxx',
        '.py', '.sh', '.cmake', '.f90', '.f', '.F90'
    }
    return Path(filepath).suffix.lower() in code_extensions


def count_lines_by_author(filepath):
    """Count lines attributed to each author using git blame with mailmap support."""
    author_lines = defaultdict(int)
    
    try:
        # Git will automatically use .mailmap if it exists and is configured
        result = subprocess.run(
            ['git', 'blame', '--line-porcelain', filepath],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return author_lines
        
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith('author '):
                author = line[7:]  # Remove 'author ' prefix
                # Skip empty lines by checking the actual content
                # The content line comes after metadata lines
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith('\t'):
                        content = lines[j][1:]  # Remove tab
                        if content.strip():  # Only count non-empty lines
                            author_lines[author] += 1
                        break
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return author_lines


def main():
    """Main function to compute SLOC by author."""
    print("Computing SLOC by author...")
    
    # Check if .mailmap exists
    if os.path.exists('.mailmap'):
        print("Using .mailmap for author name canonicalization")
    else:
        print("Warning: No .mailmap file found. Author names may not be consistent.")
    
    # Get all tracked files
    files = get_tracked_files()
    code_files = [f for f in files if is_code_file(f) and os.path.exists(f)]
    
    print(f"Found {len(code_files)} code files to analyze")
    
    # Count lines by author
    total_by_author = defaultdict(int)
    
    for i, filepath in enumerate(code_files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(code_files)}...")
        
        author_lines = count_lines_by_author(filepath)
        for author, count in author_lines.items():
            total_by_author[author] += count
    
    # Sort by SLOC count
    sorted_authors = sorted(total_by_author.items(), key=lambda x: x[1], reverse=True)
    
    # Display results
    print("\n" + "="*60)
    print(f"{'Author':<40} {'SLOC':>10}")
    print("="*60)
    
    total_sloc = sum(total_by_author.values())
    
    for author, sloc in sorted_authors:
        percentage = (sloc / total_sloc * 100) if total_sloc > 0 else 0
        print(f"{author:<40} {sloc:>10,} ({percentage:5.1f}%)")
    
    print("="*60)
    print(f"{'Total':<40} {total_sloc:>10,}")


if __name__ == "__main__":
    main()
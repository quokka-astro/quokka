#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate ASV dashboard from regression test results.")
    parser.add_argument("bench_dir", help="Path to the benchmark directory (containing the 'asv' subdirectory)")
    args = parser.parse_args()

    bench_dir = os.path.abspath(args.bench_dir)
    asv_source_dir = os.path.join(bench_dir, "asv")
    
    if not os.path.isdir(asv_source_dir):
        print(f"Error: {asv_source_dir} does not exist. (Expected to find ASV results in <bench_dir>/asv/)")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    asv_project_dir = os.path.join(script_dir, "asv_benchmarks")
    asv_results_dir = os.path.join(asv_project_dir, "results")

    # Ensure asv_project_dir exists
    if not os.path.exists(asv_project_dir):
        print(f"Error: ASV project directory {asv_project_dir} not found.")
        sys.exit(1)

    # Link results
    if os.path.islink(asv_results_dir) or os.path.exists(asv_results_dir):
        if os.path.islink(asv_results_dir):
            os.unlink(asv_results_dir)
        else:
            try:
                shutil.rmtree(asv_results_dir)
            except OSError as e:
                 print(f"Warning: Could not remove existing results dir: {e}")

    try:
        os.symlink(asv_source_dir, asv_results_dir)
        print(f"Linked {asv_source_dir} to {asv_results_dir}")
    except OSError as e:
        print(f"Error creating symlink: {e}")
        sys.exit(1)

    # Run asv publish
    print("Running 'asv publish'...")
    try:
        # Check if asv is installed
        subprocess.check_call([sys.executable, "-m", "asv", "--version"], stdout=subprocess.DEVNULL)
        
        subprocess.check_call([sys.executable, "-m", "asv", "publish"], cwd=asv_project_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error running asv publish: {e}")
        sys.exit(1)
    except FileNotFoundError:
         # Try calling 'asv' directly if 'python -m asv' failed/not found (though less likely if installed in same env)
        try:
            subprocess.check_call(["asv", "publish"], cwd=asv_project_dir)
        except Exception:
            print("Error: 'asv' command not found. Please install it with 'pip install asv'.")
            sys.exit(1)

    print(f"\nDashboard generated successfully.")
    print(f"Open the following file in your browser to view:")
    print(f"{os.path.join(asv_project_dir, 'html', 'index.html')}")

if __name__ == "__main__":
    main()

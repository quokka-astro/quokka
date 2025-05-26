import os
import sys
import subprocess
from glob import glob


def run_command(cmd, cwd=None):
    """Run a shell command and handle errors"""
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e}")
        sys.exit(1)


def main():
    max_count = 200
    count = 0
    for f in glob("./src/problems/*/*.cpp"):
        base_name = os.path.basename(f)
        log_file = f"./gitignore-chongchong-tidy/{base_name}"
        log_file_fix = f"./gitignore-chongchong-tidy-fix/{base_name}"
        # log_file_fix2 = f"./gitignore-chongchong-tidy-fix2/{base_name}"
        is_tidy_clean = True
        if os.path.exists(log_file):
            with open(log_file, "r") as log_file_:
                n_lines = len(log_file_.readlines())
                if n_lines > 2:
                    is_tidy_clean = False
            if not is_tidy_clean:
                print(f"doing tidy for {base_name}")
                cmd = f"clang-tidy {f} -fix -p ./build/build-3D-debug > {log_file_fix} && echo 'done' >> {log_file_fix} &"
                # cmd = f"clang-tidy {f} -p ./build/build-3D-debug > {log_file_fix2} && echo 'done' >> {log_file_fix2} &"
                run_command(cmd)
                count += 1
        else:
            print(f"File {log_file} does not exist")
        if count >= max_count:
            break


if __name__ == "__main__":
    main()

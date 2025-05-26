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
    stage = 2 # 1: fix it, 2: recheck after fix
    max_count = 200
    count = 0
    for f in glob("./src/problems/*/*.cpp"):
        base_name = os.path.basename(f)
        log_file_fix = f"./sims/tidy/tidy-all-problems-fix/{base_name}"
        log_file_fix2 = f"./sims/tidy/tidy-all-problems-after-fix/{base_name}"
        os.makedirs(os.path.dirname(log_file_fix), exist_ok=True)
        os.makedirs(os.path.dirname(log_file_fix2), exist_ok=True)

        # log_file_base = f"./sims/tidy/tidy-all-problems/{base_name}"
        log_file_base = f"./sims/tidy/tidy-all-fix1/{base_name}"

        is_tidy_clean = True
        if os.path.exists(log_file_base):
            with open(log_file_base, "r") as log_file_:
                n_lines = len(log_file_.readlines())
                if n_lines > 3:
                    is_tidy_clean = False
            if not is_tidy_clean:
                print(f"doing tidy for {base_name}")
                cmd = ""
                if stage == 1:
                    cmd = f"clang-tidy {f} -fix -p ./build/build-3D-debug > {log_file_fix} && echo 'done' >> {log_file_fix} &"
                elif stage == 2:
                    cmd = f"clang-tidy {f} -p ./build/build-3D-debug > {log_file_fix2} && echo 'done' >> {log_file_fix2} &"
                run_command(cmd)
                count += 1
        else:
            print(f"File {log_file_base} does not exist")
        if count >= max_count:
            break


if __name__ == "__main__":
    main()

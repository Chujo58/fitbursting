import sys
from pathlib import Path
import subprocess
import os

WINDOWS = sys.platform == "win32"
DONE = lambda: print("DONE!")

print("Installation script for the repository")
print("======================================")
print("This installation script will install:")
print("- requirements.txt")
print("- DM_phase")
print("- baseband-analysis")
print()
print("All installed files will be inside the")
print("current folder.")
print()

user_input = input("Continue with current installation? (Y/n) ").lower().strip("\r\n")
while user_input not in ["y", "n"]:
    user_input = input("Invalid answer. Try again: ").lower().strip("\r\n")

if user_input == "n":
    print("Aborting installation.")
    exit(1)


def run_py(*command, offset=""):
    if WINDOWS:
        python_path = Path(".venv") / "Scripts" / "python.exe"
    else:
        python_path = Path(".venv") / "bin" / "python"

    if offset:
        python_path = Path(offset) / python_path

    print(python_path)
    subprocess.run([str(python_path), *command], check=True)


# 0. Install virtualenv
print("Step 0: Install virtualenv")

subprocess.run(["curl", "-O", "https://bootstrap.pypa.io/virtualenv.pyz"])
DONE()

# 1. Create the .venv
print("Step 1: Create the python environment (.venv)")

if WINDOWS:
    subprocess.run(["python", "virtualenv.pyz", "-p", "python3.12", ".venv"])
else:
    subprocess.run(["python3", "virtualenv.pyz", "-p", "python3.12", ".venv"])
subprocess.run(["rm", "virtualenv.pyz"])
DONE()


# 2. Install the python dependencies (in requirements.txt)
print("Step 2: Install the python dependencies (requirements.txt)")
run_py("-m", "pip", "install", "-r", "requirements.txt")
DONE()


# 3. Install the GitHub dependencies
print("Step 3: Install the GitHub dependencies")
subprocess.run(["git", "clone", "https://github.com/CHIMEFRB/baseband-analysis"])
subprocess.run(["git", "clone", "https://github.com/danielemichilli/DM_phase"])
subprocess.run(["git", "checkout", "0ec3f4c3"], cwd="baseband-analysis")
DONE()

# 4. Build baseband-analysis
print("Step 4: Build baseband-analysis")
try:
    run_py(
        "-c",
        "import sys; sys.path.insert(0, 'baseband-analysis/'); import baseband_analysis; from baseband_analysis.analysis import snr; print('baseband-analysis installation is valid.')",
    )
except Exception as e:
    print("Invalid installation for baseband-analysis. Fixing...")
    print("Building Cython extensions...")
    os.chdir("baseband-analysis/")
    run_py("baseband_analysis/utilities/developer.py", offset="../")
    run_py("setup.py", "build_ext", "--inplace", offset="../")
    os.chdir("../")

run_py(
    "-c",
    "import sys; sys.path.insert(0, 'baseband-analysis/'); import baseband_analysis; from baseband_analysis.analysis import snr; print('baseband-analysis installation is valid.')",
)

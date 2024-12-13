import importlib.metadata
import subprocess
import sys

def generate_requirements():
    installed_packages = importlib.metadata.distributions()
    with open('requirements.txt', 'w') as f:
        for package in installed_packages:
            f.write(f"{package.metadata['Name']}=={package.version}\n")

generate_requirements()

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")

if __name__ == "__main__":
    install_requirements()
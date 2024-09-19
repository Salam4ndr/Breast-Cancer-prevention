import subprocess
import sys

# Function to check if a package is installed
def check_and_install():
    required_packages = [
        "pandas", "numpy", "seaborn", "matplotlib", "scipy", "scikit-learn", 
        "keras", "tensorflow", "graphviz", "plotly", "tensorflow"
    ]
    
    # Check installed packages
    installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
    
    # Only install if the package is missing
    missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]
    
    if missing_packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

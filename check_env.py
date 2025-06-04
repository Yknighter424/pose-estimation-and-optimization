import sys
import subprocess
import importlib

# Print Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# List of packages to check
packages = [
    "cv2",
    "numpy",
    "matplotlib",
    "mediapipe",
    "scipy",
    "pandas"
]

print("\nChecking installed packages:")
for package in packages:
    try:
        # Try to import the package
        module = importlib.import_module(package)
        if package == "cv2":
            print(f"✓ OpenCV (cv2) version: {module.__version__}")
        else:
            print(f"✓ {package} version: {module.__version__}")
    except ImportError:
        print(f"✗ {package} is not installed")
    except AttributeError:
        print(f"✓ {package} is installed (version not available)") 
#!/usr/bin/env python3
"""
Environment Verification Script for cvnav conda environment
Checks if all required packages for flight path visualization are available
"""

import sys
import importlib
import subprocess

def check_python_version():
    """Check if Python version is >= 3.10"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version >= (3, 10):
        print("✅ Python version requirement met (>= 3.10)")
        return True
    else:
        print("❌ Python version too old. Requires >= 3.10")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is available and get its version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def main():
    print("="*60)
    print("ENVIRONMENT VERIFICATION FOR FLIGHT PATH VISUALIZER")
    print("="*60)
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Check core packages needed for flight path visualizer
    print("Core Flight Visualizer Packages:")
    print("-" * 40)
    core_packages = [
        ('matplotlib', 'matplotlib'),
        ('numpy', 'numpy'), 
        ('plotly', 'plotly'),
    ]
    
    core_ok = all(check_package(pkg, imp) for pkg, imp in core_packages)
    print()
    
    # Check optional/additional packages
    print("Additional Packages:")
    print("-" * 40)
    additional_packages = [
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('Pillow', 'PIL'),
        ('opencv-python', 'cv2'),
    ]
    
    additional_ok = all(check_package(pkg, imp) for pkg, imp in additional_packages)
    print()
    
    # Summary
    print("="*60)
    print("SUMMARY:")
    if python_ok and core_ok:
        print("✅ Environment is ready for flight path visualization!")
        if additional_ok:
            print("✅ All additional packages are also available!")
        else:
            print("⚠️  Some additional packages are missing but core functionality will work")
    else:
        print("❌ Environment setup incomplete. Please install missing packages.")
    
    print("="*60)

if __name__ == "__main__":
    main()
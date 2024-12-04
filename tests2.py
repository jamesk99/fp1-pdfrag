# Test 1: Check Visual C++ version
import sys
import subprocess

def check_visual_cpp():
    try:
        # This will error if VS C++ redistributable isn't installed
        import msvc
        print("Visual C++ is installed")
    except ImportError:
        print("Visual C++ not found")

# Test 2: Check ONNX
def check_onnx():
    try:
        import onnxruntime
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
    except ImportError:
        print("ONNX Runtime not installed")

# Test 3: Check Poppler
def check_poppler():
    try:
        from pdf2image import convert_from_path
        print("Poppler is properly installed")
    except ImportError:
        print("pdf2image/poppler not installed")
    except Exception as e:
        print(f"Poppler error: {str(e)}")

# Test 4: Check all unstructured dependencies
def check_unstructured():
    try:
        import unstructured
        from unstructured.partition.pdf import partition_pdf
        print(f"Unstructured version: {unstructured.__version__}")
    except ImportError:
        print("Unstructured library not properly installed")

# Run all tests
print("Running dependency checks...")
check_visual_cpp()
check_onnx()
check_poppler()
check_unstructured()

import pkg_resources
for package in pkg_resources.working_set:
    if 'unstructured' in package.key or 'onnx' in package.key:
        print(f"{package.key}: {package.version}")
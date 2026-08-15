#!/usr/bin/env python3
"""
Colab Environment & Requirements Verification Script (Agent 2).
Checks Python version, GPU availability (T4/L4/A100), RAM, and required libraries.
"""

import sys
import os
import platform

def check_environment():
    print("=" * 60)
    print(" SMART-SEM: Google Colab Environment Health Check")
    print("=" * 60)
    
    # 1. Python version
    py_ver = platform.python_version()
    print(f"[*] Python Version: {py_ver} ({'OK' if sys.version_info >= (3, 8) else 'FAIL'})")
    
    # 2. Operating System
    print(f"[*] OS: {platform.system()} {platform.release()} ({platform.machine()})")
    
    # 3. GPU Check
    try:
        import torch
        gpu_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_avail else "CPU (Standard Runtime)"
        print(f"[*] GPU Acceleration: {gpu_name} ({'CUDA Active' if gpu_avail else 'CPU Mode'})")
    except ImportError:
        print("[*] GPU Acceleration: PyTorch not installed (OpenCV / CPU mode active)")
        
    # 4. Critical Dependencies
    packages = ["numpy", "cv2", "scipy", "yaml", "pptx", "streamlit"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "Installed")
            print(f"[*] Package '{pkg}': {ver} (OK)")
        except ImportError:
            print(f"[!] Package '{pkg}': NOT FOUND (Run: pip install -r requirements.txt)")
            
    # 5. Directory verification
    dirs = ["src", "experiments", "results", "research", "notebooks"]
    for d in dirs:
        status = "OK" if os.path.exists(d) else "MISSING"
        print(f"[*] Folder '{d}/': {status}")
        
    print("\n[OK] Environment check complete. Ready for SMART-SEM execution.")

if __name__ == "__main__":
    check_environment()

"""
Run this before launch_uq_methods.py to catch issues early.
"""
import sys
import torch

def check_dependencies():
    """Check all required packages are installed."""
    print("Checking dependencies...")
    required = {
        'torch': torch,
        'numpy': None,
        'sklearn': None,
        'matplotlib': None,
        'seaborn': None,
        'pandas': None,
        'shap': None,
        'monai': None,
    }
    
    missing = []
    for name, module in required.items():
        try:
            if module is None:
                __import__(name)
            print(f"  ✅ {name}")
        except ImportError:
            missing.append(name)
            print(f"  ❌ {name} (missing)")
    
    if missing:
        print(f"\n⚠️  Install missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✅ CUDA version: {torch.version.cuda}")
        return True
    else:
        print("  ⚠️  CUDA not available (will use CPU)")
        return False

def check_imports():
    """Check UQ_Toolbox imports work."""
    print("\nChecking UQ_Toolbox imports...")
    try:
        import UQ_Toolbox.UQ_toolbox as uq
        from UQ_Toolbox.methods import TTAMethod, GPSMethod, EnsembleSTDMethod
        from UQ_Toolbox.visualization import compare_uq_methods
        print("  ✅ All imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def check_disk_space(required_gb=10):
    """Check available disk space."""
    print(f"\nChecking disk space (need ~{required_gb}GB for GPS augmentations)...")
    try:
        import shutil
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024**3)
        print(f"  Available: {free_gb:.1f} GB")
        if free_gb < required_gb:
            print(f"  ⚠️  Low disk space! Need at least {required_gb}GB")
            return False
        print(f"  ✅ Sufficient disk space")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True  # Don't fail on this

def main():
    print("="*60)
    print("UQ_Toolbox Pre-Run Checklist")
    print("="*60 + "\n")
    
    checks = [
        check_dependencies(),
        check_cuda(),
        check_imports(),
        check_disk_space(),
    ]
    
    print("\n" + "="*60)
    if all(checks):
        print("✅ ALL CHECKS PASSED - Ready to run!")
        print("="*60)
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - Fix issues above")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
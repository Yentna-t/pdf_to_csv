"""
Tesseract OCR Setup Helper for Windows
Automatically checks, downloads, and configures Tesseract with Lao language support
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

TESSERACT_INSTALLER_URL = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
LAO_TRAINEDDATA_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/lao.traineddata"
TESSERACT_DEFAULT_PATH = r"C:\Program Files\Tesseract-OCR"

def check_tesseract_installed():
    """Check if Tesseract is already installed"""
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ Tesseract is already installed")
            print(result.stdout.split('\n')[0])
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check default installation path
    tesseract_exe = Path(TESSERACT_DEFAULT_PATH) / "tesseract.exe"
    if tesseract_exe.exists():
        print(f"✓ Tesseract found at: {tesseract_exe}")
        return True
    
    print("✗ Tesseract is not installed")
    return False

def check_lao_language():
    """Check if Lao language data is installed"""
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "lao" in result.stdout.lower():
            print("✓ Lao language support is installed")
            return True
        else:
            print("✗ Lao language support is NOT installed")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ Cannot check language support (Tesseract not in PATH)")
        return False

def download_lao_traineddata(tessdata_path):
    """Download Lao language traineddata file"""
    lao_file = tessdata_path / "lao.traineddata"
    
    if lao_file.exists():
        print(f"✓ Lao traineddata already exists at: {lao_file}")
        return True
    
    print(f"Downloading Lao language data from GitHub...")
    try:
        urllib.request.urlretrieve(LAO_TRAINEDDATA_URL, lao_file)
        print(f"✓ Downloaded lao.traineddata to: {lao_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to download Lao traineddata: {e}")
        print(f"\nManual download instructions:")
        print(f"1. Download from: {LAO_TRAINEDDATA_URL}")
        print(f"2. Save to: {lao_file}")
        return False

def setup_tesseract():
    """Main setup function"""
    print("=" * 60)
    print("Tesseract OCR Setup for Lao Language")
    print("=" * 60)
    
    # Check if Tesseract is installed
    tesseract_installed = check_tesseract_installed()
    
    if not tesseract_installed:
        print("\n" + "=" * 60)
        print("MANUAL INSTALLATION REQUIRED")
        print("=" * 60)
        print(f"\nPlease install Tesseract OCR manually:")
        print(f"1. Download installer from:")
        print(f"   {TESSERACT_INSTALLER_URL}")
        print(f"2. Run the installer")
        print(f"3. During installation, note the installation directory")
        print(f"4. Add Tesseract to your system PATH:")
        print(f"   - Search 'Environment Variables' in Windows")
        print(f"   - Edit 'Path' under System Variables")
        print(f"   - Add: C:\\Program Files\\Tesseract-OCR")
        print(f"5. Restart your terminal/IDE")
        print(f"6. Run this script again")
        return False
    
    # Find tessdata directory
    tessdata_path = None
    possible_paths = [
        Path(TESSERACT_DEFAULT_PATH) / "tessdata",
        Path(os.environ.get("TESSDATA_PREFIX", "")) if os.environ.get("TESSDATA_PREFIX") else None,
    ]
    
    for path in possible_paths:
        if path and path.exists():
            tessdata_path = path
            break
    
    if not tessdata_path:
        print("\n✗ Could not find tessdata directory")
        print("Please set TESSDATA_PREFIX environment variable")
        return False
    
    print(f"\n✓ Tessdata directory: {tessdata_path}")
    
    # Check and download Lao language data
    print("\nChecking Lao language support...")
    lao_installed = check_lao_language()
    
    if not lao_installed:
        print("\nInstalling Lao language support...")
        if download_lao_traineddata(tessdata_path):
            print("\n" + "=" * 60)
            print("✓ Setup completed successfully!")
            print("=" * 60)
            print("\nPlease restart your terminal/IDE to ensure changes take effect.")
            return True
        else:
            return False
    else:
        print("\n" + "=" * 60)
        print("✓ All components are already installed!")
        print("=" * 60)
        return True

if __name__ == "__main__":
    success = setup_tesseract()
    sys.exit(0 if success else 1)

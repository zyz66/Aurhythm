# Aurhythm - Film Negative Processor v1.0.0

**[中文](README.md) | ENGLISH**

Aurhythm is a desktop application for processing scanned film negatives. It provides a complete toolchain to help users convert scanned negatives into logarithmic space formats, suitable for digital post-processing workflows.

## Features

- **Three-stage workflow**: Input settings → Color sampling (black point/white point) → Kodak Cineon mapping and inversion → Channel alignment → Output settings
- **Color picker tools**: Offers normal picker, black point picker, and white point picker functions
- **RGB channel distribution charts**: Real-time display of color distribution for each channel
- **Multiple output spaces**: Supports Cineon, ARRI LogC3, ARRI LogC4, Sony S-Log3, and other logarithmic spaces
- **32-bit floating-point TIFF export**: Preserves full dynamic range

## Workflow

A[Input Settings] --> B[Color Sampling (Black/White Points)]
B --> C[Kodak Cineon Mapping & Inversion]
C --> D[Channel Alignment]
D --> E[Output Settings]
E --> F[Export 32-bit Floating-point TIFF]

## Installation Requirements

### Dependencies
- Python 3.8 or higher
- tkinter (usually installed with Python)
- Currently supports Windows operating system

### Step 1: Get the code

Please manually download the ZIP file, extract it, and navigate to the extracted directory.

### Step 2: Create a virtual environment
```bash
python -m venv Aurhythm
```
Activate:
```bash
Aurhythm\Scripts\activate
```
### Step 3: Install dependencies
```bash
pip install numpy pillow imageio matplotlib scipy colour-science psutil rawpy
```
Step 4: Run the application
On Windows, you can double-click run.bat, or run in the command line:
```bash
python Aurhythm.py
```
On other operating systems, run in the command line:
python Aurhythm.py

LicenseAurhythm is open-source under the MIT License. See the LICENSE file for details.
Contribution GuidelinesWe welcome contributions in any form, including but not limited to reporting bugs, suggesting new features, or submitting code improvements.
Reporting bugs: Please submit bug reports on the GitHub Issues page.
Code improvements: Pull Requests are welcome. Please fork this repository and make modifications on your branch.
AcknowledgementsThe principles of this program initially referenced the namicolor plugin, but Aurhythm's code is completely independently written and does not use any code from namicolor.

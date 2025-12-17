
```markdown
# Aurhythm - Film Negative Processor v1.0.0

**English | [中文](README.md)**

Aurhythm is a desktop application for processing film scan negatives. It provides a complete toolchain to help users convert scanned negatives into log space formats suitable for digital post-production workflows.

## Features

- **Three-stage Workflow**: Input Setup → Black/White Point Sampling → Kodak Cineon Mapping and Inversion → Channel Alignment → Output Settings
- **Color Picker Tools**: Normal pipette, black point pipette, and white point pipette
- **RGB Channel Display**: Real-time display of color distribution for each channel
- **Multiple Output Spaces**: Support for Cineon, ARRI LogC3, ARRI LogC4, Sony S-Log3, and other log spaces
- **32-bit Float TIFF Export**: Preserves full dynamic range

## Workflow

```mermaid
flowchart TD
    A[Input Setup] --> B[Black/White Point Sampling]
    B --> C[Kodak Cineon Mapping and Inversion]
    C --> D[Channel Alignment]
    D --> E[Output Settings]
    E --> F[Export 32-bit Float TIFF]

    Installation Requirements

## Required Dependencies
Python 3.8 or higher

tkinter (usually installed with Python)

Currently supports Windows operating system

Clone or download this repository

### Python Package Dependencies

```bash
pip install numpy pillow imageio matplotlib scipy colour-science psutil rawpy

Double-click run.bat to run

License
Aurhythm is open source under the MIT License. See LICENSE file for details.

Acknowledgments
The principles of this program were initially inspired by the namicolor plugin, but Aurhythm's code is completely independently written and does not use any code from namicolor.

Contribution Guidelines
We welcome all forms of contributions, including but not limited to bug reports, feature suggestions, or code improvements.

Report Issues: Please submit issue reports on the GitHub Issues page.

Improve Code: Pull requests are welcome. Fork this repository and make modifications on your branch.


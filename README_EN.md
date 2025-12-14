```markdown
# Aurhythm

**English | [中文](README.md)**

## Inspiration & Acknowledgments

The development of this project was inspired by the following works, for which we express our gratitude:

- **[NamiColor](https://github.com/Wavechaser/NamiColor)**: Provided the initial design concept for the film processing workflow.

**Important Note**: This project does **not directly use** any source code from NamiColor. All code is independently implemented. Both projects are open-sourced under the [GNU General Public License v3.0](LICENSE).

## Overview

Aurhythm is a tool for processing and converting scanned film negative images. It provides a complete graphical interface that simulates a professional film development and scanning color science pipeline to convert negative or log files into correctly colored positive images.

**Please Note**: Currently, the input is configured to accept only **16-bit linear TIFF files (gamma=1.0)**. Other input methods may not produce accurate results.

## Key Features
- **Scientific Processing Pipeline**: Implements a color space conversion of **Linear -> Log (Cineon) -> Linear**, simulating the film processing workflow.
- **Channel Alignment Tools**: Provides independent RGB channel offset and gain controls for color cast correction.
- **Real-Time Color Adjustment**: Integrates exposure, contrast, saturation, color temperature/tint adjustments with live preview.
- **Visualization Analysis**: Built-in RGB waveform (vectorscope) for color balance reference.
- **Graphical User Interface**: Developed with Tkinter, no command-line operation required.
- **Multi-Mode Support**: Capable of processing negatives, reversal films, and log files (Log2Log).

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Currently supports Windows operating systems

### Installation & Running

1. Clone or download this repository.
2. Install the required packages:
   ```bash
   pip install colour-science numpy imageio Pillow matplotlib
3. Double-click run.bat.

Contributing
We welcome all forms of contribution, including but not limited to reporting bugs, suggesting new features, or submitting code improvements.

Reporting Issues: Please submit problem reports on the GitHub Issues page.

Improving Code: Pull Requests are welcome. Please fork this repository and make your changes on your branch.

License
This project is released under the GNU General Public License v3.0 (GPL-3.0). For full details, please see the LICENSE file.

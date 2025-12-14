# Aurhythm

**中文 | [English](README_EN.md)**

## 灵感与致谢

本项目的开发受到了以下项目的启发，特此致谢：

- **[NamiColor](https://github.com/Wavechaser/NamiColor)**：为本项目提供了最初的胶片处理流程设计思路。

**重要说明**：本项目**未直接使用** NamiColor 的任何源代码，所有代码均为独立实现。两个项目均在 [GNU General Public License v3.0](LICENSE) 协议下开源。

## 项目简介
Aurhythm 是一个用于处理和转换胶片负片扫描图像的工具。它提供完整的图形化界面，通过模拟专业胶片冲扫的色彩科学流程，将负片或对数文件转换为色彩正确的正像。

**请注意**：目前输入只设置了允许16位线性tiff(gamma=1.0)文件，其他输入方式可能不准确。

## 主要功能
- **科学的处理管线**：实现 线性 -> 对数 (Cineon) -> 线性 的色彩空间转换，模拟胶片处理流程。
- **通道对齐工具**：提供独立的 RGB 通道偏移与增益控制，用于校正色偏。
- **实时色彩调整**：集成曝光、对比度、饱和度、色温/色调等调整工具，所有改动可实时预览。
- **可视化分析**：内置 RGB 分量图（示波器），为色彩平衡提供参考。
- **图形化界面**：基于 Tkinter 开发，无需命令行操作。
- **多模式支持**：支持处理负片、反转片及对数文件 (Log2Log)。

## 快速开始

### 前提条件
- Python 3.8 或更高版本
- 目前支持 Windows 操作系统

### 安装与运行

1. 克隆或下载本仓库
2. 安装依赖包：
   ```bash
   pip install numpy pillow imageio matplotlib scipy colour-science psutil
3. 双击运行 run.bat

贡献指南
我们欢迎任何形式的贡献，包括但不限于报告错误、提出新功能建议或提交代码改进。

报告问题：请在 GitHub Issues 页面提交问题报告。

改进代码：欢迎提交 Pull Request。请 Fork 本仓库，并在您的分支上进行修改。

许可证
本项目采用 GNU General Public License v3.0 (GPL-3.0) 许可证发布。详情请参阅 LICENSE 文件。

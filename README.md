# 🎞️ Aurhythm 胶片 Cineon 校准器 v3.6

> 将彩色负片 RAW 文件科学转换为 Cineon/LogC3 对数空间 TIFF  
> **核心理念**：密度法 + 解串扰矩阵 + Sigmoid H-D 曲线 = 物理映射，而非经验调色

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20|%20macOS%20|%20Linux-lightgrey)]()

---

## 📊 算法流程图

![算法流程图](./process.svg)

### 🚀 快速开始
安装依赖
```bash
pip install rawpy numpy pillow tifffile
```
运行软件
```bash
python Aurhythm.py
```
基本操作流程
添加 RAW 图像：点击「添加RAW图像」，选择 NEF/DNG/CR2 等文件

选择胶片类型：在下拉菜单中选择对应的胶片预设

片基采样：点击「采样」，然后在预览图齿孔或片基透明区域点击

自动对齐：点击「对齐」平衡三通道，修复阴影偏蓝

预览调整：切换预览分辨率，检查直方图

导出：选择输出色彩空间（Cineon/LogC3），导出 32-bit TIFF

📄 许可证
MIT License

# Aurhythm - 胶片负片处理器 v1.0.0

**中文 | [ENGLISH](README_EN.md)**

Aurhythm 是一个用于处理胶片扫描负片的桌面应用程序。它提供了一套完整的工具链，帮助用户将扫描的负片转换为对数空间格式，适合数字后期处理流程。

## 特点

- **三阶段工作流程**: 输入设置 → 取色黑点白点 → 柯达cineon以及反转 → 通道对齐 → 输出设置
- **取色吸管工具**: 提供正常吸管、黑点吸管和白点吸管功能
- **RGB色彩分量图**: 实时显示各通道的色彩分布
- **多种输出空间**: 支持 Cineon、ARRI LogC3、ARRI LogC4、Sony S-Log3 等对数空间
- **32位浮点TIFF导出**: 保留完整的动态范围

## 工作流程

flowchart TD
    A[输入设置] --> B[取色黑点白点]
    B --> C[柯达Cineon映射及反转]
    C --> D[通道对齐]
    D --> E[输出设置]
    E --> F[导出32位浮点TIFF]

## 安装要求

### 必需依赖
- Python 3.8 或更高版本
- tkinter (通常随Python一起安装)
- 目前支持 Windows 操作系统

克隆或下载本仓库
### Python包依赖
    ```bash
        pip install numpy pillow imageio matplotlib scipy colour-science psutil rawpy

双击运行 run.bat

许可证
Aurhythm 使用 MIT 许可证开源。详见 LICENSE 文件。

贡献指南
我们欢迎任何形式的贡献，包括但不限于报告错误、提出新功能建议或提交代码改进。

报告问题：请在 GitHub Issues 页面提交问题报告。

改进代码：欢迎提交 Pull Request。请 Fork 本仓库，并在您的分支上进行修改。

致谢
本程序的原理早期参考了namicolor插件，但Aurhythm的代码是完全独立编写的，未使用namicolor的任何代码。

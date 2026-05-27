# 🎞️ Aurhythm 胶片 Cineon 校准器 v3.6

> 将彩色负片 RAW 文件科学转换为 Cineon/LogC3 对数空间 TIFF  
> **核心理念**：密度法 + 解串扰矩阵 + Sigmoid H-D 曲线 = 物理映射，而非经验调色

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20|%20macOS%20|%20Linux-lightgrey)]()

---

## 📊 算法流程图
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 1400" font-family="monospace" font-size="12">
  <defs>
    <style>
      .box { fill: #1a1a1a; stroke: #00b4d8; stroke-width: 1.5; rx: 6; }
      .text { fill: #e0e0e0; text-anchor: middle; dominant-baseline: middle; }
      .arrow { stroke: #00b4d8; stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }
      .label { fill: #888; text-anchor: middle; font-size: 10; }
    </style>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#00b4d8"/>
    </marker>
  </defs>

  <!-- 背景 -->
  <rect width="900" height="1400" fill="#0d1117"/>

  <!-- ==================== 输入 ==================== -->
  <rect x="300" y="20" width="300" height="50" class="box"/>
  <text x="450" y="48" class="text" font-size="14">📷 输入: RAW 文件</text>
  <text x="450" y="65" class="label">NEF / DNG / CR2 / ARW / RAF / ORF / RW2</text>

  <line x1="450" y1="70" x2="450" y2="95" class="arrow"/>

  <!-- ==================== 步骤 1 ==================== -->
  <rect x="250" y="100" width="400" height="80" class="box"/>
  <text x="450" y="125" class="text" font-size="13">📦 步骤 1: RAW 解码</text>
  <text x="450" y="145" class="text" font-size="11">rawpy.postprocess(gamma=(1,1), no_auto_bright=True)</text>
  <text x="450" y="165" class="text" fill="#00b4d8">→ CMOS 电平 (0-1 浮点)</text>

  <line x1="450" y1="180" x2="450" y2="205" class="arrow"/>

  <!-- ==================== 步骤 2 ==================== -->
  <rect x="250" y="210" width="400" height="80" class="box"/>
  <text x="450" y="235" class="text" font-size="13">🎯 步骤 2: 片基采样</text>
  <text x="450" y="255" class="text" font-size="11">点击齿孔/透明区域 → 记录 base_val (RGB)</text>
  <text x="450" y="275" class="text" fill="#00b4d8">→ 定义白场: T=1, D=0</text>

  <line x1="450" y1="290" x2="450" y2="315" class="arrow"/>

  <!-- ==================== 步骤 3 ==================== -->
  <rect x="250" y="320" width="400" height="80" class="box"/>
  <text x="450" y="345" class="text" font-size="13">⚖️ 步骤 3: 密度域自动对齐</text>
  <text x="450" y="365" class="text" font-size="11">D = -log10(V) → target = mean(D) → 增益 = 10^(D_raw-D_target)</text>
  <text x="450" y="385" class="text" fill="#00b4d8">→ 修复阴影偏蓝</text>

  <line x1="450" y1="400" x2="450" y2="425" class="arrow"/>

  <!-- ==================== 步骤 4 ==================== -->
  <rect x="250" y="430" width="400" height="80" class="box"/>
  <text x="450" y="455" class="text" font-size="13">📐 步骤 4: 线性 → 密度</text>
  <text x="450" y="475" class="text" font-size="11">T = V_gained / base_val</text>
  <text x="450" y="495" class="text" font-size="11">D = -log10(T)  (Davidson 公式)</text>

  <line x1="450" y1="510" x2="450" y2="535" class="arrow"/>

  <!-- ==================== 步骤 5 ==================== -->
  <rect x="250" y="540" width="400" height="80" class="box"/>
  <text x="450" y="565" class="text" font-size="13">🔄 步骤 5: 解串扰矩阵</text>
  <text x="450" y="585" class="text" font-size="11">[D_C, D_M, D_Y]ᵀ = M_inv @ [D_R, D_G, D_B]ᵀ</text>
  <text x="450" y="605" class="text" fill="#00b4d8">→ 分离三层染料光谱串扰</text>

  <line x1="450" y1="620" x2="450" y2="645" class="arrow"/>

  <!-- ==================== 步骤 6 ==================== -->
  <rect x="250" y="650" width="400" height="95" class="box"/>
  <text x="450" y="675" class="text" font-size="13">📈 步骤 6: 软裁剪 Sigmoid H-D 曲线</text>
  <text x="450" y="695" class="text" font-size="11">t = (D - D_min) / (D_max - D_min)</text>
  <text x="450" y="715" class="text" font-size="11">t_soft = 1 / (1 + exp(-k×(t-0.5)))  (Logistic 软裁剪)</text>
  <text x="450" y="735" class="text" font-size="11">logH = b - (1/a) × ln(t_soft/(1-t_soft))</text>

  <line x1="450" y1="745" x2="450" y2="770" class="arrow"/>

  <!-- ==================== 步骤 7 ==================== -->
  <rect x="250" y="775" width="400" height="80" class="box"/>
  <text x="450" y="800" class="text" font-size="13">🎞️ 步骤 7: Cineon 编码</text>
  <text x="450" y="820" class="text" font-size="11">CineonCode = 95 + 500 × (logH_ref - logH)</text>
  <text x="450" y="840" class="text" fill="#00b4d8">→ 柯达工业标准对数空间 (0-1023)</text>

  <line x1="450" y1="855" x2="450" y2="880" class="arrow"/>

  <!-- ==================== 分支：预览 vs 导出 ==================== -->
  <line x1="450" y1="880" x2="200" y2="920" class="arrow"/>
  <line x1="450" y1="880" x2="700" y2="920" class="arrow"/>
  <text x="300" y="900" class="label">预览</text>
  <text x="580" y="900" class="label">导出</text>

  <!-- ==================== 步骤 8a: 预览 ==================== -->
  <rect x="80" y="925" width="240" height="95" class="box"/>
  <text x="200" y="950" class="text" font-size="12">👁️ 预览显示</text>
  <text x="200" y="970" class="text" font-size="10">display = 1.0 - data (反相)</text>
  <text x="200" y="990" class="text" font-size="10">sRGB 伽马编码</text>
  <text x="200" y="1010" class="text" fill="#00b4d8">→ 8-bit RGB 屏幕显示</text>

  <!-- ==================== 步骤 8b: LogC3 可选 ==================== -->
  <rect x="580" y="925" width="240" height="80" class="box"/>
  <text x="700" y="950" class="text" font-size="12">🎥 步骤 8: LogC3 转换 (可选)</text>
  <text x="700" y="970" class="text" font-size="10">E = 10^((Code-95)/500)</text>
  <text x="700" y="990" class="text" font-size="10">LogC3 = 0.0925×ln(E+0.005)+0.391</text>

  <line x1="700" y1="1005" x2="700" y2="1030" class="arrow"/>

  <!-- ==================== 步骤 8c: LUT ==================== -->
  <rect x="580" y="1035" width="240" height="80" class="box"/>
  <text x="700" y="1060" class="text" font-size="12">🎨 步骤 9: LUT 套用</text>
  <text x="700" y="1080" class="text" font-size="10">加载 .cube 3D LUT</text>
  <text x="700" y="1100" class="text" fill="#00b4d8">→ LogC3 → Rec.709</text>

  <line x1="700" y1="1115" x2="700" y2="1140" class="arrow"/>

  <!-- ==================== 最终输出 ==================== -->
  <rect x="520" y="1145" width="360" height="65" class="box" stroke="#00ff00"/>
  <text x="700" y="1170" class="text" font-size="14" fill="#00ff00">💾 输出</text>
  <text x="700" y="1190" class="text" font-size="11">32-bit 浮点 TIFF</text>
  <text x="700" y="1205" class="text" font-size="11">Cineon 或 LogC3 对数空间</text>

  <!-- ==================== 右侧预览区域 ==================== -->
  <rect x="80" y="1060" width="240" height="150" class="box" stroke="#888" stroke-dasharray="4"/>
  <text x="200" y="1085" class="text" font-size="12" fill="#888">📺 预览显示</text>
  <text x="200" y="1105" class="text" font-size="10" fill="#888">• 反相负片转正</text>
  <text x="200" y="1125" class="text" font-size="10" fill="#888">• sRGB 伽马校正</text>
  <text x="200" y="1145" class="text" font-size="10" fill="#888">• 实时鼠标读数</text>
  <text x="200" y="1165" class="text" font-size="10" fill="#888">• RGB 直方图</text>

  <!-- 标题 -->
  <text x="450" y="1330" class="text" font-size="16" fill="#00b4d8" font-weight="bold">Aurhythm 胶片 Cineon 校准器 v3.6</text>
  <text x="450" y="1360" class="text" font-size="12" fill="#888">RAW → 密度 → 解串扰 → Sigmoid H-D → Cineon → LogC3</text>
</svg>
```
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

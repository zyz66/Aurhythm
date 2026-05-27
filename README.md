# 🎞️ Aurhythm 胶片 Cineon 校准器 v3.6

> 将彩色负片 RAW 文件科学转换为 Cineon/LogC3 对数空间 TIFF  
> **核心理念**：密度法 + 解串扰矩阵 + Sigmoid H-D 曲线 = 物理映射，而非经验调色

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20|%20macOS%20|%20Linux-lightgrey)]()

---

## 📊 算法流程图

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 1400" font-family="monospace" font-size="12">
  <defs>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#00b4d8"/>
    </marker>
  </defs>

  <rect width="900" height="1400" fill="#0d1117"/>

  <!-- 输入 -->
  <rect x="300" y="20" width="300" height="50" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="450" y="45" fill="#e0e0e0" text-anchor="middle" font-size="14">📷 输入: RAW 文件</text>
  <text x="450" y="65" fill="#888" text-anchor="middle" font-size="10">NEF / DNG / CR2 / ARW / RAF / ORF / RW2</text>
  <line x1="450" y1="70" x2="450" y2="95" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- 步骤1 -->
  <rect x="250" y="100" width="400" height="80" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="450" y="125" fill="#e0e0e0" text-anchor="middle" font-size="13">📦 步骤 1: RAW 解码</text>
  <text x="450" y="145" fill="#e0e0e0" text-anchor="middle" font-size="11">rawpy.postprocess(gamma=(1,1), no_auto_bright=True)</text>
  <text x="450" y="165" fill="#00b4d8" text-anchor="middle" font-size="11">→ CMOS 电平 (0-1 浮点)</text>
  <line x1="450" y1="180" x2="450" y2="205" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- 步骤2 -->
  <rect x="250" y="210" width="400" height="80" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="450" y="235" fill="#e0e0e0" text-anchor="middle" font-size="13">🎯 步骤 2: 片基采样</text>
  <text x="450" y="255" fill="#e0e0e0" text-anchor="middle" font-size="11">点击齿孔/透明区域 → 记录 base_val (RGB)</text>
  <text x="450" y="275" fill="#00b4d8" text-anchor="middle" font-size="11">→ 定义白场: T=1, D=0</text>
  <line x1="450" y1="290" x2="450" y2="315" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- 步骤3 -->
  <rect x="250" y="320" width="400" height="80" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="450" y="345" fill="#e0e0e0" text-anchor="middle" font-size="13">⚖️ 步骤 3: 密度域自动对齐</text>
  <text x="450" y="365" fill="#e0e0e0" text-anchor="middle" font-size="11">D = -log10(V) → target = mean(D) → 增益 = 10^(D_raw-D_target)</text>
  <text x="450" y="385" fill="#00b4d8" text-anchor="middle" font-size="11">→ 修复阴影偏蓝</text>
  <line x1="450" y1="400" x2="450" y2="425" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- 步骤4 -->
  <rect x="250" y="430" width="400" height="80" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="450" y="455" fill="#e0e0e0" text-anchor="middle" font-size="13">📐 步骤 4: 线性 → 密度</text>
  <text x="450" y="475" fill="#e0e0e0" text-anchor="middle" font-size="11">T = V_gained / base_val</text>
  <text x="450" y="495" fill="#e0e0e0" text-anchor="middle" font-size="11">D = -log10(T) (Davidson 公式)</text>
  <line x1="450" y1="510" x2="450" y2="535" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- 步骤5 -->
  <rect x="250" y="540" width="400" height="80" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="450" y="565" fill="#e0e0e0" text-anchor="middle" font-size="13">🔄 步骤 5: 解串扰矩阵</text>
  <text x="450" y="585" fill="#e0e0e0" text-anchor="middle" font-size="11">[D_C, D_M, D_Y]ᵀ = M_inv @ [D_R, D_G, D_B]ᵀ</text>
  <text x="450" y="605" fill="#00b4d8" text-anchor="middle" font-size="11">→ 分离三层染料光谱串扰</text>
  <line x1="450" y1="620" x2="450" y2="645" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- 步骤6 -->
  <rect x="250" y="650" width="400" height="95" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="450" y="675" fill="#e0e0e0" text-anchor="middle" font-size="13">📈 步骤 6: 软裁剪 Sigmoid H-D 曲线</text>
  <text x="450" y="695" fill="#e0e0e0" text-anchor="middle" font-size="11">t = (D - D_min) / (D_max - D_min)</text>
  <text x="450" y="715" fill="#e0e0e0" text-anchor="middle" font-size="11">t_soft = 1 / (1 + exp(-k×(t-0.5))) (Logistic 软裁剪)</text>
  <text x="450" y="735" fill="#e0e0e0" text-anchor="middle" font-size="11">logH = b - (1/a) × ln(t_soft/(1-t_soft))</text>
  <line x1="450" y1="745" x2="450" y2="770" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- 步骤7 -->
  <rect x="250" y="775" width="400" height="80" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="450" y="800" fill="#e0e0e0" text-anchor="middle" font-size="13">🎞️ 步骤 7: Cineon 编码</text>
  <text x="450" y="820" fill="#e0e0e0" text-anchor="middle" font-size="11">CineonCode = 95 + 500 × (logH_ref - logH)</text>
  <text x="450" y="840" fill="#00b4d8" text-anchor="middle" font-size="11">→ 柯达工业标准对数空间 (0-1023)</text>
  <line x1="450" y1="855" x2="450" y2="880" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- 分支 -->
  <line x1="450" y1="880" x2="200" y2="920" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="450" y1="880" x2="700" y2="920" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <text x="300" y="900" fill="#888" text-anchor="middle" font-size="10">预览</text>
  <text x="580" y="900" fill="#888" text-anchor="middle" font-size="10">导出</text>

  <!-- 预览 -->
  <rect x="80" y="925" width="240" height="95" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="200" y="950" fill="#e0e0e0" text-anchor="middle" font-size="12">👁️ 预览显示</text>
  <text x="200" y="970" fill="#e0e0e0" text-anchor="middle" font-size="10">display = 1.0 - data (反相)</text>
  <text x="200" y="990" fill="#e0e0e0" text-anchor="middle" font-size="10">sRGB 伽马编码</text>
  <text x="200" y="1010" fill="#00b4d8" text-anchor="middle" font-size="10">→ 8-bit RGB 屏幕显示</text>

  <!-- LogC3 -->
  <rect x="580" y="925" width="240" height="80" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="700" y="950" fill="#e0e0e0" text-anchor="middle" font-size="12">🎥 步骤 8: LogC3 转换</text>
  <text x="700" y="970" fill="#e0e0e0" text-anchor="middle" font-size="10">E = 10^((Code-95)/500)</text>
  <text x="700" y="990" fill="#e0e0e0" text-anchor="middle" font-size="10">LogC3 = 0.0925×ln(E+0.005)+0.391</text>
  <line x1="700" y1="1005" x2="700" y2="1030" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- LUT -->
  <rect x="580" y="1035" width="240" height="80" fill="#1a1a1a" stroke="#00b4d8" stroke-width="1.5" rx="6"/>
  <text x="700" y="1060" fill="#e0e0e0" text-anchor="middle" font-size="12">🎨 步骤 9: LUT 套用</text>
  <text x="700" y="1080" fill="#e0e0e0" text-anchor="middle" font-size="10">加载 .cube 3D LUT</text>
  <text x="700" y="1100" fill="#00b4d8" text-anchor="middle" font-size="10">→ LogC3 → Rec.709</text>
  <line x1="700" y1="1115" x2="700" y2="1140" stroke="#00b4d8" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- 输出 -->
  <rect x="520" y="1145" width="360" height="65" fill="#1a1a1a" stroke="#00ff00" stroke-width="1.5" rx="6"/>
  <text x="700" y="1170" fill="#00ff00" text-anchor="middle" font-size="14">💾 输出</text>
  <text x="700" y="1190" fill="#e0e0e0" text-anchor="middle" font-size="11">32-bit 浮点 TIFF</text>
  <text x="700" y="1205" fill="#e0e0e0" text-anchor="middle" font-size="11">Cineon 或 LogC3 对数空间</text>

  <!-- 说明 -->
  <rect x="80" y="1060" width="240" height="150" fill="#1a1a1a" stroke="#888" stroke-width="1.5" rx="6" stroke-dasharray="4"/>
  <text x="200" y="1085" fill="#888" text-anchor="middle" font-size="12">📺 预览功能</text>
  <text x="200" y="1105" fill="#888" text-anchor="middle" font-size="10">• 反相负片转正</text>
  <text x="200" y="1125" fill="#888" text-anchor="middle" font-size="10">• sRGB 伽马校正</text>
  <text x="200" y="1145" fill="#888" text-anchor="middle" font-size="10">• 实时鼠标读数</text>
  <text x="200" y="1165" fill="#888" text-anchor="middle" font-size="10">• RGB 直方图</text>

  <!-- 标题 -->
  <text x="450" y="1330" fill="#00b4d8" text-anchor="middle" font-size="16" font-weight="bold">Aurhythm 胶片 Cineon 校准器 v3.6</text>
  <text x="450" y="1360" fill="#888" text-anchor="middle" font-size="12">RAW → 密度 → 解串扰 → Sigmoid H-D → Cineon → LogC3</text>
</svg>

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

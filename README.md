Aurhythm 胶片 Cineon 校准器 v3.6
将彩色负片 RAW 文件科学转换为 Cineon/LogC3 对数空间 TIFF
核心理念：密度法 + 解串扰矩阵 + Sigmoid H-D 曲线 = 物理映射

算法流程图

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 1400" font-family="monospace" font-size="12"> <defs> <style> .box { fill: #1a1a1a; stroke: #00b4d8; stroke-width: 1.5; rx: 6; } .text { fill: #e0e0e0; text-anchor: middle; dominant-baseline: middle; } .arrow { stroke: #00b4d8; stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); } .label { fill: #888; text-anchor: middle; font-size: 10; } </style> <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"> <polygon points="0 0, 8 3, 0 6" fill="#00b4d8"/> </marker> </defs><rect width="900" height="1400" fill="#0d1117"/><rect x="300" y="20" width="300" height="50" class="box"/> <text x="450" y="48" class="text" font-size="14">输入: RAW 文件</text> <text x="450" y="65" class="label">NEF / DNG / CR2 / ARW / RAF / ORF / RW2</text> <line x1="450" y1="70" x2="450" y2="95" class="arrow"/><rect x="250" y="100" width="400" height="80" class="box"/> <text x="450" y="125" class="text" font-size="13">步骤 1: RAW 解码</text> <text x="450" y="145" class="text" font-size="11">rawpy.postprocess(gamma=(1,1), no_auto_bright=True)</text> <text x="450" y="165" class="text" fill="#00b4d8">-> CMOS 电平 (0-1 浮点)</text> <line x1="450" y1="180" x2="450" y2="205" class="arrow"/><rect x="250" y="210" width="400" height="80" class="box"/> <text x="450" y="235" class="text" font-size="13">步骤 2: 片基采样</text> <text x="450" y="255" class="text" font-size="11">点击齿孔/透明区域 -> 记录 base_val (RGB)</text> <text x="450" y="275" class="text" fill="#00b4d8">-> 定义白场: T=1, D=0</text> <line x1="450" y1="290" x2="450" y2="315" class="arrow"/><rect x="250" y="320" width="400" height="80" class="box"/> <text x="450" y="345" class="text" font-size="13">步骤 3: 密度域自动对齐</text> <text x="450" y="365" class="text" font-size="11">D = -log10(V) -> target = mean(D) -> 增益 = 10^(D_raw-D_target)</text> <text x="450" y="385" class="text" fill="#00b4d8">-> 修复阴影偏蓝</text> <line x1="450" y1="400" x2="450" y2="425" class="arrow"/><rect x="250" y="430" width="400" height="80" class="box"/> <text x="450" y="455" class="text" font-size="13">步骤 4: 线性 -> 密度</text> <text x="450" y="475" class="text" font-size="11">T = V_gained / base_val</text> <text x="450" y="495" class="text" font-size="11">D = -log10(T) (Davidson 公式)</text> <line x1="450" y1="510" x2="450" y2="535" class="arrow"/><rect x="250" y="540" width="400" height="80" class="box"/> <text x="450" y="565" class="text" font-size="13">步骤 5: 解串扰矩阵</text> <text x="450" y="585" class="text" font-size="11">[D_C, D_M, D_Y]T = M_inv @ [D_R, D_G, D_B]T</text> <text x="450" y="605" class="text" fill="#00b4d8">-> 分离三层染料光谱串扰</text> <line x1="450" y1="620" x2="450" y2="645" class="arrow"/><rect x="250" y="650" width="400" height="95" class="box"/> <text x="450" y="675" class="text" font-size="13">步骤 6: 软裁剪 Sigmoid H-D 曲线</text> <text x="450" y="695" class="text" font-size="11">t = (D - D_min) / (D_max - D_min)</text> <text x="450" y="715" class="text" font-size="11">t_soft = 1 / (1 + exp(-k*(t-0.5))) (Logistic 软裁剪)</text> <text x="450" y="735" class="text" font-size="11">logH = b - (1/a) * ln(t_soft/(1-t_soft))</text> <line x1="450" y1="745" x2="450" y2="770" class="arrow"/><rect x="250" y="775" width="400" height="80" class="box"/> <text x="450" y="800" class="text" font-size="13">步骤 7: Cineon 编码</text> <text x="450" y="820" class="text" font-size="11">CineonCode = 95 + 500 * (logH_ref - logH)</text> <text x="450" y="840" class="text" fill="#00b4d8">-> 柯达工业标准对数空间 (0-1023)</text> <line x1="450" y1="855" x2="450" y2="880" class="arrow"/><line x1="450" y1="880" x2="200" y2="920" class="arrow"/> <line x1="450" y1="880" x2="700" y2="920" class="arrow"/> <text x="300" y="900" class="label">预览</text> <text x="580" y="900" class="label">导出</text><rect x="80" y="925" width="240" height="95" class="box"/> <text x="200" y="950" class="text" font-size="12">预览显示</text> <text x="200" y="970" class="text" font-size="10">display = 1.0 - data (反相)</text> <text x="200" y="990" class="text" font-size="10">sRGB 伽马编码</text> <text x="200" y="1010" class="text" fill="#00b4d8">-> 8-bit RGB 屏幕显示</text><rect x="580" y="925" width="240" height="80" class="box"/> <text x="700" y="950" class="text" font-size="12">步骤 8: LogC3 转换 (可选)</text> <text x="700" y="970" class="text" font-size="10">E = 10^((Code-95)/500)</text> <text x="700" y="990" class="text" font-size="10">LogC3 = 0.0925*ln(E+0.005)+0.391</text> <line x1="700" y1="1005" x2="700" y2="1030" class="arrow"/><rect x="580" y="1035" width="240" height="80" class="box"/> <text x="700" y="1060" class="text" font-size="12">步骤 9: LUT 套用</text> <text x="700" y="1080" class="text" font-size="10">加载 .cube 3D LUT</text> <text x="700" y="1100" class="text" fill="#00b4d8">-> LogC3 -> Rec.709</text> <line x1="700" y1="1115" x2="700" y2="1140" class="arrow"/><rect x="520" y="1145" width="360" height="65" class="box" stroke="#00ff00"/> <text x="700" y="1170" class="text" font-size="14" fill="#00ff00">输出</text> <text x="700" y="1190" class="text" font-size="11">32-bit 浮点 TIFF</text> <text x="700" y="1205" class="text" font-size="11">Cineon 或 LogC3 对数空间</text><rect x="80" y="1060" width="240" height="150" class="box" stroke="#888" stroke-dasharray="4"/> <text x="200" y="1085" class="text" font-size="12" fill="#888">预览功能</text> <text x="200" y="1105" class="text" font-size="10" fill="#888">- 负片反相转正</text> <text x="200" y="1125" class="text" font-size="10" fill="#888">- sRGB 伽马校正</text> <text x="200" y="1145" class="text" font-size="10" fill="#888">- 实时鼠标读数</text> <text x="200" y="1165" class="text" font-size="10" fill="#888">- RGB 直方图</text>
<text x="450" y="1330" class="text" font-size="16" fill="#00b4d8" font-weight="bold">Aurhythm 胶片 Cineon 校准器 v3.6</text>
<text x="450" y="1360" class="text" font-size="12" fill="#888">RAW -> 密度 -> 解串扰 -> Sigmoid H-D -> Cineon -> LogC3</text>
</svg>

快速开始

系统要求

Python 3.8 或更高版本

tkinter (通常随 Python 一起安装)

支持 Windows / macOS / Linux

安装步骤

克隆仓库

git clone https://github.com/yourusername/Aurhythm.git
cd Aurhythm

创建虚拟环境

python -m venv Aurhythm

Windows:
Aurhythm\Scripts\activate

macOS/Linux:
source Aurhythm/bin/activate

安装依赖

pip install numpy pillow imageio rawpy tifffile

运行程序

python Aurhythm.py

核心算法

本程序基于以下公开技术文档实现：

+----------------------+-----------------------+---------------------------------+
| 技术 | 来源 | 用途 |
+----------------------+-----------------------+---------------------------------+
| Cineon 编码 | Kodak (1993) | 对数空间编码标准 |
| LogC3 | ARRI | 对数空间色彩科学 |
| Vision3 光谱数据 | Kodak | 串扰矩阵计算 |
| H-D 曲线 | Hurter & Driffield (1890) | 密度-曝光特性 |
| 密度法 | Davidson 公式 | CMOS 电平转密度 |
+----------------------+-----------------------+---------------------------------+

所有数学公式均为公开领域知识，代码为独立编写。

胶片预设库

+---------------------------+----------+--------+---------------------------+
| 胶片 | 类型 | 色温 | 用途 |
+---------------------------+----------+--------+---------------------------+
| Kodak Portra 400 | 民用负片 | 日光 | 人像肤色优化 |
| Kodak Portra 800 | 民用负片 | 日光 | 高感光度，弱光 |
| Kodak Vision3 250D (5207) | 电影负片 | 日光 | ECN-2，宽宽容度 |
| Kodak Vision3 500T (5219) | 电影负片 | 钨丝灯 | 室内拍摄，暖光 |
| Kodak Gold 200 | 民用负片 | 日光 | 日常卷，高饱和 |
| Fuji C200 | 民用负片 | 日光 | 绿色表现优秀 |
| 通用负片 (默认) | 通用 | 日光 | 未知胶片类型 |
+---------------------------+----------+--------+---------------------------+

采样报警阈值

+----------------+----------------+----------------------------------------+
| 片基值 | 状态 | 建议 |
+----------------+----------------+----------------------------------------+
| < 0.4 | 严重欠曝 | 提高光源亮度 / 降低快门速度 / 开大光圈 |
| 0.4 - 0.55 | 略欠 | 提高 0.3-0.7 档曝光 |
| 0.55 - 0.8 | 正常 | 最佳动态范围 |
| > 0.85 | 可能过曝 | 降低曝光 |
+----------------+----------------+----------------------------------------+

正确采样位置：齿孔、片孔旁边、两帧之间的透明区域（不是画面内容）
齿孔值 1.0 是翻拍曝光达标的标志。

输出格式

+--------------+---------------+--------------+-------+--------------------------------+
| 色彩空间 | 文件后缀 | 数据类型 | 范围 | 用途 |
+--------------+---------------+--------------+-------+--------------------------------+
| Cineon | _cineon.tif | 32-bit 浮点 | 0-1 | DaVinci Resolve CST 节点 |
| LogC3 | _logc3.tif | 32-bit 浮点 | 0-1 | 套 ARRI 官方 LUT 到 Rec.709 |
+--------------+---------------+--------------+-------+--------------------------------+

在 DaVinci Resolve 中使用

导出 _logc3.tif 文件

在 Resolve 中创建项目，色彩管理设为 DaVinci YRGB

导入 TIFF，添加 Color Space Transform (CST) 节点

CST 设置：

输入色彩空间：ARRI LogC3

输入伽马：ARRI LogC3

输出色彩空间：Rec.709

输出伽马：Gamma 2.4

调整 Lift/Gamma/Gain 完成最终调色

常见问题

Q: 为什么预览画面是灰的？

A: 正常的。正确曝光的 RAW 在对数空间就是"灰但细节全"的中间状态。
导出 LogC3 后在达芬奇套 LUT 即可恢复。

Q: 片基采样值多少正常？

A: 0.55-0.8。齿孔区域应该在这个范围。如果 <0.4，说明翻拍曝光严重不足。

Q: 为什么阴影偏蓝？

A: 通常是两个原因：

翻拍曝光不足（片基 <0.4）

自动对齐在线性域做的

v3.6 已改用密度域对齐。

Q: 可以在 Adobe Lightroom 里用吗？

A: 不能。需要支持色彩管理的工作流（DaVinci Resolve、Nuke、Photoshop 等）。

Q: Vision3 250D 和 500T 的区别？

A: 250D 是日光平衡（5500K），用于室外；500T 是钨丝灯平衡（3200K），用于室内暖光。

许可证

MIT License

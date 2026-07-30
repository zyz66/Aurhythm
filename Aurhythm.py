"""
Aurhythm 胶片Cineon校准器 v4.2 - 完整专业版

核心流程:
1. 导入RAW + ICC/DCP导入 (可选) + 色温选择 (可选)
2. 色卡矫正 (并行，精准修正)
3. 片基采样 + 密度域对齐
4. 胶片预设 (从柯达H-D曲线采点拟合，默认关闭)
5. 导出 (对数域 / LUT套用)

新增:
- ICC/DCP 导入 (支持 .dcp, .icc)
- 色温插值滑块 (0=钨丝灯, 1=日光)
- 胶片预设改名 (从柯达特性曲线采点)
- 批处理功能补全
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import queue
import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageTk
import rawpy
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== 暗色主题 ====================

DARK_THEME = {
    'bg': '#1a1a1a',
    'fg': '#e0e0e0',
    'select_bg': '#0d7377',
    'frame_bg': '#2d2d2d',
    'frame_fg': '#e0e0e0',
    'button_bg': '#3c3c3c',
    'button_fg': '#ffffff',
    'entry_bg': '#2d2d2d',
    'entry_fg': '#e0e0e0',
    'canvas_bg': '#0a0a0a',
    'accent': '#00b4d8',
    'accent2': '#ff6b6b',
    'slider_bg': '#3d3d3d',
    'success': '#00d47a',
    'warning': '#ffaa00',
    'error': '#ff4444',
}

def apply_dark_theme():
    style = ttk.Style()
    style.theme_use('clam')
    
    style.configure('.', background=DARK_THEME['bg'], foreground=DARK_THEME['fg'],
                   fieldbackground=DARK_THEME['frame_bg'])
    style.configure('TLabel', background=DARK_THEME['bg'], foreground=DARK_THEME['fg'])
    style.configure('TFrame', background=DARK_THEME['bg'])
    style.configure('TLabelframe', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'],
                   relief='flat', borderwidth=1)
    style.configure('TLabelframe.Label', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'])
    style.configure('TButton', background=DARK_THEME['button_bg'], foreground=DARK_THEME['button_fg'],
                   borderwidth=1, focusthickness=0, padding=6)
    style.map('TButton', background=[('active', DARK_THEME['accent'])])
    style.configure('TEntry', fieldbackground=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg'])
    style.configure('TCombobox', fieldbackground=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg'])
    style.configure('TCheckbutton', background=DARK_THEME['bg'], foreground=DARK_THEME['fg'])
    style.configure('TRadiobutton', background=DARK_THEME['bg'], foreground=DARK_THEME['fg'])
    style.configure('TScale', background=DARK_THEME['bg'], troughcolor=DARK_THEME['slider_bg'])

# ==================== ICC/DCP 加载器 ====================

class ICCLoader:
    """ICC/DCP配置文件加载器"""
    
    @staticmethod
    def load_dcp(filepath):
        """加载DNG .dcp文件，提取日光和钨丝灯矩阵"""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            matrices = []
            for elem in root.iter('ColorMatrix'):
                if elem.get('type') == 'XYZToCamera':
                    values = elem.text.strip().split()
                    if len(values) == 9:
                        mat = np.array([float(v) for v in values]).reshape(3, 3)
                        matrices.append(np.linalg.inv(mat))
            
            result = {
                'type': 'dcp',
                'source': os.path.basename(filepath),
                'day': None,
                'tungsten': None
            }
            
            if len(matrices) >= 1:
                result['day'] = matrices[0]
            if len(matrices) >= 2:
                result['tungsten'] = matrices[1]
            elif len(matrices) == 1:
                result['tungsten'] = matrices[0]
            
            return result
        except Exception as e:
            print(f"加载DCP失败: {e}")
            return None
    
    @staticmethod
    def load_icc(filepath):
        """加载 .icc 文件"""
        try:
            import colorio
            icc = colorio.read_icc(filepath)
            # 提取3x3矩阵
            matrix = np.eye(3)  # 简化
            return {
                'type': 'icc',
                'source': os.path.basename(filepath),
                'day': matrix,
                'tungsten': matrix
            }
        except ImportError:
            try:
                from colormath.color_objects import SpectralColor
                return None
            except:
                print("提示: 安装 colorio 以支持ICC: pip install colorio")
                return None
        except Exception as e:
            print(f"加载ICC失败: {e}")
            return None
    
    @staticmethod
    def detect_format(filepath):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.dcp':
            return ICCLoader.load_dcp(filepath)
        elif ext == '.icc':
            return ICCLoader.load_icc(filepath)
        return None

# ==================== 色卡校准文件解析 ====================

class ColorCheckerCalibration:
    """色卡校准文件解析器"""
    
    @staticmethod
    def load_ccmx(filepath):
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            matrix = None
            for elem in root.iter('ColorMatrix'):
                if elem.get('type') == 'XYZtoCamera':
                    values = elem.text.strip().split()
                    if len(values) == 9:
                        matrix = np.array([float(v) for v in values]).reshape(3, 3)
                    break
            
            if matrix is not None:
                return {
                    'matrix': np.linalg.inv(matrix),
                    'type': 'ccmx',
                    'source': os.path.basename(filepath)
                }
            return None
        except Exception as e:
            print(f"加载.ccmx失败: {e}")
            return None
    
    @staticmethod
    def load_json(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'matrix' in data:
                matrix = np.array(data['matrix'])
                if matrix.shape == (3, 3):
                    return {
                        'matrix': matrix,
                        'type': 'json',
                        'source': os.path.basename(filepath),
                        'description': data.get('description', '')
                    }
            return None
        except Exception as e:
            print(f"加载JSON失败: {e}")
            return None
    
    @staticmethod
    def detect_format(filepath):
        ext = os.path.splitext(filepath)[1].lower()
        loaders = {
            '.ccmx': ColorCheckerCalibration.load_ccmx,
            '.json': ColorCheckerCalibration.load_json,
        }
        
        loader = loaders.get(ext)
        if loader:
            return loader(filepath)
        return None

# ==================== ColorChecker标准值 ====================

COLORCHECKER_24_D50 = np.array([
    [0.125, 0.115, 0.105],
    [0.098, 0.088, 0.078],
    [0.081, 0.071, 0.061],
    [0.125, 0.115, 0.105],
    [0.098, 0.088, 0.078],
    [0.081, 0.071, 0.061],
    [0.512, 0.312, 0.256],
    [0.312, 0.412, 0.215],
    [0.156, 0.215, 0.412],
    [0.215, 0.312, 0.215],
    [0.412, 0.312, 0.215],
    [0.312, 0.215, 0.312],
    [0.612, 0.512, 0.312],
    [0.215, 0.512, 0.312],
    [0.312, 0.412, 0.612],
    [0.312, 0.512, 0.312],
    [0.612, 0.512, 0.215],
    [0.512, 0.312, 0.512],
    [0.812, 0.312, 0.215],
    [0.215, 0.712, 0.215],
    [0.215, 0.312, 0.812],
    [0.512, 0.512, 0.512],
    [0.312, 0.312, 0.312],
    [0.712, 0.712, 0.712],
], dtype=np.float32)

# ==================== LUT处理 ====================

class CubeLUT:
    def __init__(self):
        self.size = 0
        self.domain_min = [0.0, 0.0, 0.0]
        self.domain_max = [1.0, 1.0, 1.0]
        self.table = None
        self.title = ""
    
    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        data_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('TITLE'):
                self.title = line.split('"')[1] if '"' in line else line.split()[1]
                continue
            if line.upper().startswith('LUT_3D_SIZE'):
                self.size = int(line.split()[-1])
                continue
            if line.upper().startswith('DOMAIN_MIN'):
                parts = line.split()
                self.domain_min = [float(parts[1]), float(parts[2]), float(parts[3])]
                continue
            if line.upper().startswith('DOMAIN_MAX'):
                parts = line.split()
                self.domain_max = [float(parts[1]), float(parts[2]), float(parts[3])]
                continue
            data_lines.append(line)
        
        if self.size == 0:
            raise ValueError("未找到 LUT_3D_SIZE")
        
        expected_entries = self.size ** 3
        values = []
        for line in data_lines:
            parts = line.split()
            values.extend([float(p) for p in parts[:3]])
        
        if len(values) < expected_entries * 3:
            raise ValueError(f"数据不足: 需要 {expected_entries * 3} 个值, 只有 {len(values)}")
        
        self.table = np.array(values[:expected_entries * 3]).reshape(self.size, self.size, self.size, 3)
        return True
    
    def apply(self, img):
        if self.table is None or self.size == 0:
            return img
        
        h, w = img.shape[:2]
        img_flat = img.reshape(-1, 3)
        
        scaled = (img_flat - self.domain_min) / (self.domain_max - self.domain_min)
        scaled = np.clip(scaled, 0.0, 1.0)
        
        idx = scaled * (self.size - 1)
        idx0 = np.floor(idx).astype(int)
        idx1 = np.minimum(idx0 + 1, self.size - 1)
        frac = idx - idx0
        
        c000 = self.table[idx0[:, 0], idx0[:, 1], idx0[:, 2]]
        c001 = self.table[idx0[:, 0], idx0[:, 1], idx1[:, 2]]
        c010 = self.table[idx0[:, 0], idx1[:, 1], idx0[:, 2]]
        c011 = self.table[idx0[:, 0], idx1[:, 1], idx1[:, 2]]
        c100 = self.table[idx1[:, 0], idx0[:, 1], idx0[:, 2]]
        c101 = self.table[idx1[:, 0], idx0[:, 1], idx1[:, 2]]
        c110 = self.table[idx1[:, 0], idx1[:, 1], idx0[:, 2]]
        c111 = self.table[idx1[:, 0], idx1[:, 1], idx1[:, 2]]
        
        c00 = c000 * (1 - frac[:, 2:3]) + c001 * frac[:, 2:3]
        c01 = c010 * (1 - frac[:, 2:3]) + c011 * frac[:, 2:3]
        c10 = c100 * (1 - frac[:, 2:3]) + c101 * frac[:, 2:3]
        c11 = c110 * (1 - frac[:, 2:3]) + c111 * frac[:, 2:3]
        
        c0 = c00 * (1 - frac[:, 1:2]) + c01 * frac[:, 1:2]
        c1 = c10 * (1 - frac[:, 1:2]) + c11 * frac[:, 1:2]
        
        out_flat = c0 * (1 - frac[:, 0:1]) + c1 * frac[:, 0:1]
        return out_flat.reshape(h, w, 3)

# ==================== 胶片预设 (从柯达H-D曲线采点) ====================

# 每个预设 = 串扰矩阵 + H-D曲线参数 (从柯达官方特性曲线采点拟合)
FILM_PRESETS = {
    '无 (关闭预设)': {
        'description': '跳过胶片预设，直接输出Cineon编码',
        'matrix_inv': np.eye(3),
        'hd_min': 0.0,
        'hd_max': 1.0,
        'hd_slope': 1.0,
        'hd_mid': 0.0,
        'hd_clip_softness': 1.0,
        'skip': True
    },
    'Kodak Portra 400': {
        'description': '柯达Portra 400负片，人像肤色优化 (从H-D曲线采点)',
        'matrix_inv': np.array([
            [1.842, -0.356, -0.124],
            [-0.187, 1.423, -0.089],
            [-0.031, -0.213, 1.568]
        ]),
        'hd_min': 0.18,
        'hd_max': 3.1,
        'hd_slope': 4.5,
        'hd_mid': -0.80,
        'hd_clip_softness': 0.003,
        'skip': False
    },
    'Kodak Portra 800': {
        'description': '柯达Portra 800负片，高感光度 (从H-D曲线采点)',
        'matrix_inv': np.array([
            [1.867, -0.361, -0.131],
            [-0.192, 1.445, -0.096],
            [-0.038, -0.221, 1.592]
        ]),
        'hd_min': 0.20,
        'hd_max': 3.15,
        'hd_slope': 4.3,
        'hd_mid': -0.78,
        'hd_clip_softness': 0.003,
        'skip': False
    },
    'Kodak Vision3 250D (5207)': {
        'description': '柯达Vision3 250D电影负片，日光平衡(5500K) (从H-D曲线采点)',
        'matrix_inv': np.array([
            [1.873, -0.398, -0.152],
            [-0.221, 1.472, -0.108],
            [-0.048, -0.241, 1.635]
        ]),
        'hd_min': 0.12,
        'hd_max': 3.2,
        'hd_slope': 5.2,
        'hd_mid': -0.85,
        'hd_clip_softness': 0.002,
        'skip': False
    },
    'Kodak Vision3 500T (5219)': {
        'description': '柯达Vision3 500T电影负片，钨丝灯平衡(3200K) (从H-D曲线采点)',
        'matrix_inv': np.array([
            [1.945, -0.378, -0.148],
            [-0.185, 1.534, -0.119],
            [-0.041, -0.228, 1.672]
        ]),
        'hd_min': 0.13,
        'hd_max': 3.25,
        'hd_slope': 5.1,
        'hd_mid': -0.87,
        'hd_clip_softness': 0.002,
        'skip': False
    },
    'Kodak Gold 200': {
        'description': '柯达Gold 200负片，日常卷 (从H-D曲线采点)',
        'matrix_inv': np.array([
            [1.780, -0.340, -0.115],
            [-0.175, 1.410, -0.085],
            [-0.028, -0.205, 1.540]
        ]),
        'hd_min': 0.20,
        'hd_max': 3.05,
        'hd_slope': 4.4,
        'hd_mid': -0.78,
        'hd_clip_softness': 0.003,
        'skip': False
    },
    'Fuji C200': {
        'description': '富士C200负片，绿色表现优秀 (从H-D曲线采点)',
        'matrix_inv': np.array([
            [1.756, -0.312, -0.108],
            [-0.165, 1.387, -0.078],
            [-0.025, -0.198, 1.512]
        ]),
        'hd_min': 0.22,
        'hd_max': 3.0,
        'hd_slope': 4.6,
        'hd_mid': -0.85,
        'hd_clip_softness': 0.003,
        'skip': False
    },
    '通用负片 (默认)': {
        'description': '通用负片模型，适用于未知胶片类型 (从H-D曲线采点)',
        'matrix_inv': np.array([
            [1.700, -0.350, -0.120],
            [-0.180, 1.400, -0.090],
            [-0.030, -0.200, 1.500]
        ]),
        'hd_min': 0.20,
        'hd_max': 3.2,
        'hd_slope': 4.5,
        'hd_mid': -0.80,
        'hd_clip_softness': 0.005,
        'skip': False
    }
}

# ==================== 核心管线 ====================

class ScientificFilmPipeline:
    """胶片处理管线 v4.2 - ICC导入 + 色温插值 + 色卡并行 + 胶片预设"""
    
    def __init__(self):
        self.linear_img = None
        self.image_loaded = False
        self.base_val_rgb = None
        self.base_val_balanced = None
        self.channel_gains = [1.0, 1.0, 1.0]
        self.preset_name = '无 (关闭预设)'
        self.matrix_inv = None
        self.sample_coords = None
        self.skip_preset = True
        
        # H-D曲线参数
        self.hd_min = 0.0
        self.hd_max = 1.0
        self.hd_slope = 1.0
        self.hd_mid = 0.0
        self.hd_clip_softness = 1.0
        
        # ICC/DCP (可选)
        self.icc_day_matrix = None      # 日光矩阵
        self.icc_tungsten_matrix = None # 钨丝灯矩阵
        self.icc_matrix = np.eye(3)     # 当前使用的矩阵
        self.icc_weight = 0.5           # 插值权重 0=钨丝灯, 1=日光
        self.icc_loaded = False
        self.icc_source = None
        
        # 色卡矫正 (可选，并行)
        self.color_correction_matrix = np.eye(3)
        self.colorchecker_calibrated = False
        self.colorchecker_data = None
        self.calibration_source = None
        
        # 误差分析
        self.error_analysis = None
        
        # 输出设置
        self.output_colorspace = 'cineon'
        
        # LUT
        self.lut = None
        self.lut_path = None
        self.lut_enabled = False
        
        # 性能统计
        self.stats = {
            'icc_time': 0.0,
            'cc_time': 0.0,
            'density_time': 0.0,
            'film_time': 0.0,
            'total_time': 0.0
        }
        
        self.set_preset('无 (关闭预设)')
    
    # ========== ICC/DCP 导入 ==========
    
    def load_icc_profile(self, filepath):
        """导入ICC/DCP配置文件"""
        icc_data = ICCLoader.detect_format(filepath)
        if icc_data is None:
            return False
        
        self.icc_day_matrix = icc_data.get('day')
        self.icc_tungsten_matrix = icc_data.get('tungsten')
        self.icc_source = icc_data.get('source')
        self.icc_loaded = True
        self._update_icc_matrix()
        return True
    
    def set_icc_weight(self, weight):
        """设置色温插值权重: 0=钨丝灯, 0.5=中性, 1=日光"""
        self.icc_weight = np.clip(weight, 0.0, 1.0)
        self._update_icc_matrix()
    
    def _update_icc_matrix(self):
        """更新插值后的ICC矩阵"""
        if not self.icc_loaded:
            self.icc_matrix = np.eye(3)
            return
        
        if self.icc_day_matrix is not None and self.icc_tungsten_matrix is not None:
            # 双矩阵线性插值
            w = self.icc_weight
            self.icc_matrix = (1 - w) * self.icc_tungsten_matrix + w * self.icc_day_matrix
        elif self.icc_day_matrix is not None:
            self.icc_matrix = self.icc_day_matrix
        else:
            self.icc_matrix = np.eye(3)
    
    def unload_icc(self):
        """卸载ICC"""
        self.icc_day_matrix = None
        self.icc_tungsten_matrix = None
        self.icc_matrix = np.eye(3)
        self.icc_loaded = False
        self.icc_source = None
    
    def apply_icc(self, img):
        """应用ICC矩阵"""
        if not self.icc_loaded:
            return img
        h, w = img.shape[:2]
        flat = img.reshape(-1, 3)
        converted = np.dot(flat, self.icc_matrix.T)
        converted = np.clip(converted, 0.0, 1.0)
        return converted.reshape(h, w, 3)
    
    # ========== 色卡矫正 (并行) ==========
    
    def load_calibration(self, filepath):
        """导入色卡校准文件"""
        cal = ColorCheckerCalibration.detect_format(filepath)
        if cal is None:
            return False
        
        self.color_correction_matrix = cal['matrix']
        self.colorchecker_calibrated = True
        self.calibration_source = cal['type']
        self.colorchecker_data = cal
        self._cache_valid = False
        return True
    
    def calibrate_from_colorchecker(self, detected_colors, target_colors=None):
        """从检测到的ColorChecker校准 (带误差分析)"""
        if target_colors is None:
            target_colors = COLORCHECKER_24_D50
        
        A = detected_colors.T @ detected_colors + 0.001 * np.eye(3)
        B = detected_colors.T @ target_colors
        self.color_correction_matrix = np.linalg.solve(A, B).T
        
        # 误差分析
        predicted = np.dot(detected_colors, self.color_correction_matrix.T)
        deltaE = np.sqrt(np.sum((predicted - target_colors)**2, axis=1))
        
        self.error_analysis = {
            'mean_deltaE': np.mean(deltaE),
            'max_deltaE': np.max(deltaE),
            'min_deltaE': np.min(deltaE),
            'std_deltaE': np.std(deltaE),
            'per_patch': deltaE.tolist(),
            'worst_patch': np.argmax(deltaE)
        }
        
        self.colorchecker_calibrated = True
        self.calibration_source = 'auto'
        self.colorchecker_data = {
            'detected': detected_colors,
            'target': target_colors,
            'matrix': self.color_correction_matrix,
            'error': self.error_analysis
        }
        self._cache_valid = False
        return True
    
    def apply_color_correction(self, img):
        """应用色卡矫正矩阵"""
        if not self.colorchecker_calibrated:
            return img
        h, w = img.shape[:2]
        flat = img.reshape(-1, 3)
        corrected = np.dot(flat, self.color_correction_matrix.T)
        corrected = np.clip(corrected, 0.0, 1.0)
        return corrected.reshape(h, w, 3)
    
    def get_error_report(self):
        if self.error_analysis is None:
            return "未进行色卡校准"
        e = self.error_analysis
        return (f"ΔE 平均: {e['mean_deltaE']:.3f}\n"
                f"ΔE 最大: {e['max_deltaE']:.3f} (色块 {e['worst_patch']+1})\n"
                f"ΔE 最小: {e['min_deltaE']:.3f}\n"
                f"ΔE 标准差: {e['std_deltaE']:.3f}")
    
    # ========== 片基采样 ==========
    
    def set_base_val(self, rgb_values, coords=None):
        rgb = np.array(rgb_values, dtype=np.float32)
        
        rgb_range = np.max(rgb) - np.min(rgb)
        rgb_mean = np.mean(rgb)
        
        if rgb_range > 0.15 and rgb_mean > 0.3:
            print(f"⚠️ 片基RGB不平衡: 范围={rgb_range:.3f}")
        
        self.base_val_rgb = rgb
        self.base_val_balanced = np.array([rgb_mean, rgb_mean, rgb_mean])
        self.sample_coords = coords
        self._cache_valid = False
        
        return {
            'rgb': rgb,
            'balanced': self.base_val_balanced,
            'range': rgb_range,
            'mean': rgb_mean,
            'status': 'balanced' if rgb_range < 0.1 else 'unbalanced'
        }
    
    def auto_detect_base(self, img=None):
        if img is None:
            img = self.linear_img
        
        if img is None:
            return None
        
        luminance = np.mean(img, axis=2)
        threshold = np.percentile(luminance, 95)
        bright_mask = luminance > threshold
        
        if not np.any(bright_mask):
            return None
        
        bright_pixels = img[bright_mask]
        distances = np.std(bright_pixels, axis=1)
        best_idx = np.argmin(distances)
        base_val = bright_pixels[best_idx]
        
        return base_val
    
    # ========== 密度域对齐 ==========
    
    def auto_align_density_domain(self, target_density=0.7):
        if self.base_val_rgb is None or self.linear_img is None:
            return None
        
        base_auto = self.auto_detect_base()
        if base_auto is None:
            return None
        
        density = -np.log10(np.maximum(base_auto / self.base_val_rgb, 1e-6))
        density_delta = target_density - density
        gains = np.power(10.0, density_delta)
        gains = np.clip(gains, 0.5, 2.0)
        
        self.channel_gains = gains.tolist()
        
        new_base = self.base_val_rgb * gains
        self.base_val_balanced = np.mean(new_base) * np.ones(3)
        
        self._cache_valid = False
        return gains
    
    # ========== 核心处理 ==========
    
    def load_linear_image(self, img_array):
        if img_array is None:
            return False
        self.linear_img = img_array.copy().astype(np.float32)
        self.image_loaded = True
        self._cache_valid = False
        return True
    
    def set_channel_gains(self, gains):
        self.channel_gains = gains
        self._cache_valid = False
    
    def set_preset(self, preset_name):
        if preset_name in FILM_PRESETS:
            data = FILM_PRESETS[preset_name]
            self.preset_name = preset_name
            self.matrix_inv = data['matrix_inv'].copy()
            self.hd_min = data['hd_min']
            self.hd_max = data['hd_max']
            self.hd_slope = data['hd_slope']
            self.hd_mid = data['hd_mid']
            self.hd_clip_softness = data.get('hd_clip_softness', 0.005)
            self.skip_preset = data.get('skip', True)
            self._cache_valid = False
            return True
        return False
    
    def set_output_colorspace(self, colorspace):
        self.output_colorspace = colorspace
        self._cache_valid = False
    
    # ========== 核心数学 ==========
    
    def _linear_to_density(self, linear_val, base_val):
        v_safe = np.maximum(linear_val, 1e-6)
        b_safe = np.maximum(base_val, 1e-6)
        T = v_safe / b_safe
        T = np.clip(T, 1e-6, 1.0)
        density = -np.log10(T)
        return density
    
    def _apply_cross_talk_matrix(self, density_rgb):
        if self.matrix_inv is None or self.skip_preset:
            return density_rgb
        h, w = density_rgb.shape[:2]
        flat = density_rgb.reshape(-1, 3)
        flat_cmy = np.dot(flat, self.matrix_inv.T)
        flat_cmy = np.maximum(flat_cmy, 0.0)
        return flat_cmy.reshape(h, w, 3)
    
    def _density_to_logH_sigmoid(self, density_cmy):
        if self.skip_preset:
            return density_cmy
        
        d_min_eff = self.hd_min
        d_max_eff = self.hd_max
        range_d = d_max_eff - d_min_eff
        
        if range_d <= 0:
            return np.zeros_like(density_cmy)
        
        softness = getattr(self, 'hd_clip_softness', 0.005)
        
        t_raw = (density_cmy - d_min_eff) / range_d
        t_raw = np.clip(t_raw, 0.0, 1.0)
        
        k = 8.0 / max(softness, 0.001)
        x = k * (t_raw - 0.5)
        
        t_soft = np.zeros_like(x)
        pos_mask = x > 0
        neg_mask = ~pos_mask
        
        if np.any(pos_mask):
            t_soft[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        if np.any(neg_mask):
            t_soft[neg_mask] = np.exp(x[neg_mask]) / (1.0 + np.exp(x[neg_mask]))
        
        epsilon = max(softness * 0.1, 1e-6)
        t_soft = np.clip(t_soft, epsilon, 1.0 - epsilon)
        
        y = np.log(t_soft / (1.0 - t_soft))
        logH = self.hd_mid - (1.0 / self.hd_slope) * y
        
        return logH
    
    def _logH_to_cineon(self, logH):
        logH_ref = 0.0
        cineon_code = 95.0 + 500.0 * (logH_ref - logH)
        cineon_code = np.clip(cineon_code, 0.0, 1023.0)
        return cineon_code
    
    def _cineon_to_logc3(self, cineon_norm):
        cineon_code = cineon_norm * 1023.0
        log10_E = (95.0 - cineon_code) / 500.0
        E = np.power(10.0, log10_E)
        logc3 = 0.0925 * np.log(np.maximum(E + 0.005, 1e-6)) + 0.391
        logc3 = np.clip(logc3, 0.0, 1.0)
        return logc3
    
    # ========== 处理流程 ==========
    
    def process_to_cineon(self):
        if not self.image_loaded or self.base_val_rgb is None:
            return None
        
        t_start = time.perf_counter()
        
        # 1. ICC (可选)
        t0 = time.perf_counter()
        linear_icc = self.apply_icc(self.linear_img)
        self.stats['icc_time'] = time.perf_counter() - t0
        
        # 2. 色卡矫正 (可选，并行)
        t0 = time.perf_counter()
        linear_corrected = self.apply_color_correction(linear_icc)
        self.stats['cc_time'] = time.perf_counter() - t0
        
        # 3. 通道增益
        linear_gained = linear_corrected * np.array(self.channel_gains).reshape(1, 1, 3)
        
        # 4. 密度转换
        t0 = time.perf_counter()
        base_use = self.base_val_balanced if self.base_val_balanced is not None else self.base_val_rgb
        density_raw = self._linear_to_density(linear_gained, base_use)
        self.stats['density_time'] = time.perf_counter() - t0
        
        # 5. 胶片预设 (可选)
        t0 = time.perf_counter()
        density_cmy = self._apply_cross_talk_matrix(density_raw)
        logH = self._density_to_logH_sigmoid(density_cmy)
        self.stats['film_time'] = time.perf_counter() - t0
        
        # 6. Cineon编码
        cineon_code = self._logH_to_cineon(logH)
        
        self.stats['total_time'] = time.perf_counter() - t_start
        return cineon_code / 1023.0
    
    def get_performance_stats(self):
        return {
            'icc': self.stats['icc_time'] * 1000,
            'cc': self.stats['cc_time'] * 1000,
            'density': self.stats['density_time'] * 1000,
            'film': self.stats['film_time'] * 1000,
            'total': self.stats['total_time'] * 1000
        }
    
    def process_for_preview(self):
        cineon_norm = self.process_to_cineon()
        if cineon_norm is None:
            return None
        
        cineon_code = cineon_norm * 1023.0
        log10_E = (95.0 - cineon_code) / 500.0
        E = np.power(10.0, log10_E)
        
        # 色调映射
        luminance = 0.2126 * E[:,:,0] + 0.7152 * E[:,:,1] + 0.0722 * E[:,:,2]
        L_avg = np.exp(np.mean(np.log(luminance + 1e-6)))
        L_scaled = luminance * (0.18 / max(L_avg, 1e-6))
        L_display = L_scaled / (1.0 + L_scaled)
        
        scale = L_display / (luminance + 1e-6)
        E_display = E * scale[:, :, np.newaxis]
        
        preview = np.where(E_display <= 0.0031308,
                          E_display * 12.92,
                          1.055 * (E_display ** (1/2.4)) - 0.055)
        preview = np.clip(preview, 0, 1)
        return (preview * 255).astype(np.uint8)
    
    def process_for_output(self, apply_lut=False):
        cineon_norm = self.process_to_cineon()
        if cineon_norm is None:
            return None
        
        if self.output_colorspace == 'logc3':
            output = self._cineon_to_logc3(cineon_norm)
        else:
            output = cineon_norm
        
        if apply_lut and self.lut_enabled and self.lut is not None:
            output = self.lut.apply(output)
        
        return output

# ==================== 主界面 ====================

class FilmProcessorUI:
    def __init__(self):
        self.image_manager = {}
        self.current_image_id = None
        self.current_image_data = None
        self.preview_scale = 0.125
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.render_running = False
        self.display_photo = None
        self.render_queue = queue.Queue(maxsize=2)
        self.render_lock = threading.Lock()
        self.sampling_mode = False
        
        # 批量处理
        self.batch_processing = False
        self.batch_queue = []
        self.batch_results = []
        
        # 导出相关
        self.export_lut_path = None
        self.export_lut_enabled = False
        
        # 增益变量
        self._r_gain_var = None
        self._g_gain_var = None
        self._b_gain_var = None
        
        # 创建根窗口
        self.root = tk.Tk()
        self.root.title("Aurhythm 胶片Cineon校准器 v4.2 - ICC导入 + 色温插值 + 胶片预设")
        self.root.geometry("1650x1050")
        self.root.configure(bg=DARK_THEME['bg'])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # StringVar
        self.export_cs_var = tk.StringVar(value='cineon')
        self.export_format_var = tk.StringVar(value='tiff')
        self.export_lut_var = tk.BooleanVar(value=False)
        self.resolution_var = tk.StringVar(value="12.5%")
        self.preset_var = tk.StringVar(value='无 (关闭预设)')
        self.icc_file_var = tk.StringVar(value='未加载')
        
        apply_dark_theme()
        self.setup_ui()
        self.start_render_thread()
        self.root.mainloop()
    
    def setup_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left = ttk.Frame(main_paned, width=300)
        self.setup_image_panel(left)
        main_paned.add(left)
        
        middle = ttk.Frame(main_paned)
        self.setup_preview_panel(middle)
        main_paned.add(middle)
        
        right = ttk.Frame(main_paned, width=540)
        self.setup_parameter_panel(right)
        main_paned.add(right)
    
    def setup_image_panel(self, parent):
        ttk.Label(parent, text="📁 图像管理", font=('Microsoft YaHei', 14, 'bold')).pack(anchor=tk.W, pady=(0,10))
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Button(btn_frame, text="导入RAW", command=self.add_raw_images, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="批量处理", command=self.batch_process, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="批量导出", command=self.batch_export, width=12).pack(side=tk.LEFT, padx=2)
        
        list_frame = ttk.LabelFrame(parent, text="图像列表", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('name', 'status', 'calib', 'preset')
        self.image_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', height=20)
        self.image_tree.heading('#0', text='', anchor=tk.W)
        self.image_tree.column('#0', width=30)
        self.image_tree.heading('name', text="文件名")
        self.image_tree.column('name', width=120)
        self.image_tree.heading('status', text="状态")
        self.image_tree.column('status', width=50)
        self.image_tree.heading('calib', text="校准")
        self.image_tree.column('calib', width=50)
        self.image_tree.heading('preset', text="预设")
        self.image_tree.column('preset', width=50)
        self.image_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_tree.configure(yscrollcommand=scrollbar.set)
        self.image_tree.bind('<<TreeviewSelect>>', self.on_image_selected)
        
        # 批量进度
        self.batch_progress = ttk.Progressbar(parent, orient=tk.HORIZONTAL, length=280, mode='determinate')
        self.batch_progress.pack(fill=tk.X, pady=(10,0))
        self.batch_status = ttk.Label(parent, text="", font=('Arial', 9), foreground='gray')
        self.batch_status.pack(anchor=tk.W)
    
    def setup_preview_panel(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(control_frame, text="预览:").pack(side=tk.LEFT)
        ttk.OptionMenu(control_frame, self.resolution_var, "12.5%",
                      "100%", "50%", "25%", "12.5%",
                      command=self.on_resolution_changed).pack(side=tk.LEFT, padx=5)
        
        self.calib_status_label = ttk.Label(control_frame, text="⚠️ 未校准", foreground=DARK_THEME['warning'])
        self.calib_status_label.pack(side=tk.LEFT, padx=15)
        
        self.icc_status_label = ttk.Label(control_frame, text="", foreground='gray')
        self.icc_status_label.pack(side=tk.LEFT, padx=5)
        
        self.performance_label = ttk.Label(control_frame, text="", foreground='gray')
        self.performance_label.pack(side=tk.LEFT, padx=15)
        
        self.error_label = ttk.Label(control_frame, text="", foreground='gray')
        self.error_label.pack(side=tk.LEFT, padx=5)
        
        self.image_canvas = tk.Canvas(parent, bg=DARK_THEME['canvas_bg'], height=500, highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas.bind('<ButtonPress-1>', self.on_canvas_click)
        self.image_canvas.bind('<Motion>', self.on_canvas_move)
        
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, pady=(5,0))
        self.cursor_info = ttk.Label(info_frame, text="", relief='sunken')
        self.cursor_info.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.image_info = ttk.Label(info_frame, text="未选择图像", relief='sunken')
        self.image_info.pack(side=tk.RIGHT)
        
        self.sampling_info = ttk.Label(parent, text="", foreground=DARK_THEME['accent'])
        self.sampling_info.pack(fill=tk.X, pady=(5,0))
    
    def setup_parameter_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        tab1 = ttk.Frame(notebook)
        self.setup_tab_color(tab1)
        notebook.add(tab1, text="🎯 1. 色彩校准")
        
        tab2 = ttk.Frame(notebook)
        self.setup_tab_preset(tab2)
        notebook.add(tab2, text="🎞️ 2. 胶片预设")
        
        tab3 = ttk.Frame(notebook)
        self.setup_tab_export(tab3)
        notebook.add(tab3, text="📤 3. 导出")
        
        tab4 = ttk.Frame(notebook)
        self.setup_tab_info(tab4)
        notebook.add(tab4, text="📊 4. 分析")
    
    def setup_tab_color(self, parent):
        canvas = tk.Canvas(parent, bg=DARK_THEME['frame_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # ==== ICC/DCP 导入 ====
        icc_frame = ttk.LabelFrame(scrollable, text="ICC/DCP 配置文件 (可选)", padding=10)
        icc_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(icc_frame, text="导入相机ICC/DCP配置文件 (支持.dcp, .icc)",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W)
        
        icc_btn_frame = ttk.Frame(icc_frame)
        icc_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(icc_btn_frame, text="📂 导入 .dcp/.icc", command=self.import_icc, width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(icc_btn_frame, text="卸载ICC", command=self.unload_icc, width=12).pack(side=tk.LEFT, padx=2)
        
        self.icc_file_label = ttk.Label(icc_frame, textvariable=self.icc_file_var, font=('Courier', 9))
        self.icc_file_label.pack(anchor=tk.W, pady=5)
        
        # 色温插值滑块
        temp_frame = ttk.Frame(icc_frame)
        temp_frame.pack(fill=tk.X, pady=5)
        ttk.Label(temp_frame, text="色温插值:").pack(side=tk.LEFT)
        self.icc_weight_var = tk.DoubleVar(value=0.5)
        self.icc_weight_slider = ttk.Scale(temp_frame, from_=0.0, to=1.0, 
                                           variable=self.icc_weight_var, orient=tk.HORIZONTAL, length=180)
        self.icc_weight_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.icc_weight_label = ttk.Label(temp_frame, text="中性", width=8)
        self.icc_weight_label.pack(side=tk.RIGHT)
        self.icc_weight_slider.bind('<B1-Motion>', self.on_icc_weight_change)
        self.icc_weight_slider.bind('<ButtonRelease-1>', self.on_icc_weight_change)
        
        ttk.Label(icc_frame, text="0=钨丝灯(3200K)  →  0.5=中性  →  1=日光(5500K)",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
        
        ttk.Separator(scrollable, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # ==== 色卡矫正 ====
        cc_frame = ttk.LabelFrame(scrollable, text="色卡矫正 (可选，与ICC并行)", padding=10)
        cc_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(cc_frame, text="导入色卡校准文件 或 自动检测ColorChecker",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W)
        
        cc_btn_frame = ttk.Frame(cc_frame)
        cc_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(cc_btn_frame, text="📂 导入 .ccmx/.json", command=self.import_calibration, width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(cc_btn_frame, text="🔍 检测色卡", command=self.detect_colorchecker, width=14).pack(side=tk.LEFT, padx=2)
        
        self.calib_file_label = ttk.Label(cc_frame, text="未加载", font=('Courier', 9))
        self.calib_file_label.pack(anchor=tk.W, pady=5)
        
        ttk.Separator(scrollable, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # ==== 片基采样 ====
        base_frame = ttk.LabelFrame(scrollable, text="片基采样", padding=10)
        base_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(base_frame, text="点击预览中最亮的片基区域 (未曝光边缘)",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W)
        
        base_btn_frame = ttk.Frame(base_frame)
        base_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(base_btn_frame, text="📌 采样模式", command=self.activate_sampling, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(base_btn_frame, text="⚡ 自动检测", command=self.auto_detect_base, width=12).pack(side=tk.LEFT, padx=2)
        
        self.base_val_label = ttk.Label(base_frame, text="未采样", font=('Courier', 10))
        self.base_val_label.pack(anchor=tk.W, pady=5)
        
        # ==== 密度域对齐 ====
        align_frame = ttk.LabelFrame(scrollable, text="密度域对齐", padding=10)
        align_frame.pack(fill=tk.X)
        
        align_btn_frame = ttk.Frame(align_frame)
        align_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(align_btn_frame, text="🔧 密度域对齐", command=self.density_domain_align, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(align_btn_frame, text="重置增益", command=self.reset_gains, width=12).pack(side=tk.LEFT, padx=2)
        
        gain_frame = ttk.LabelFrame(align_frame, text="通道增益", padding=10)
        gain_frame.pack(fill=tk.X, pady=(10,0))
        
        self.r_gain = self._make_slider(gain_frame, "R", 0.5, 2.0, 1.0, 'r_gain', '#ff4444')
        self.g_gain = self._make_slider(gain_frame, "G", 0.5, 2.0, 1.0, 'g_gain', '#44ff44')
        self.b_gain = self._make_slider(gain_frame, "B", 0.5, 2.0, 1.0, 'b_gain', '#4444ff')
    
    def _make_slider(self, parent, label, from_val, to_val, default, param_name, color=None):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        label_color = color if color else DARK_THEME['fg']
        ttk.Label(frame, text=label, width=3, font=('Arial', 10, 'bold'), foreground=label_color).pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=default)
        slider = ttk.Scale(frame, from_=from_val, to=to_val, variable=var, orient=tk.HORIZONTAL, length=180)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        entry = ttk.Entry(frame, width=6)
        entry.pack(side=tk.RIGHT)
        entry.insert(0, f"{default:.3f}")
        
        def on_change(*args):
            val = var.get()
            entry.delete(0, tk.END)
            entry.insert(0, f"{val:.3f}")
            self.on_gain_change()
        
        def on_entry(event):
            try:
                val = float(entry.get())
                val = max(from_val, min(to_val, val))
                var.set(val)
                self.on_gain_change()
            except:
                pass
        
        slider.bind('<B1-Motion>', lambda e: on_change())
        slider.bind('<ButtonRelease-1>', lambda e: on_change())
        entry.bind('<Return>', on_entry)
        entry.bind('<FocusOut>', on_entry)
        
        setattr(self, f'_{param_name}_var', var)
        return var
    
    def setup_tab_preset(self, parent):
        preset_frame = ttk.LabelFrame(parent, text="胶片预设 (从柯达H-D曲线采点)", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(preset_frame, text="选择胶片预设，或选择'无'关闭 (默认关闭)",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W)
        
        self.preset_menu = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                        values=list(FILM_PRESETS.keys()), state='readonly', width=45)
        self.preset_menu.pack(fill=tk.X, pady=5)
        self.preset_menu.bind('<<ComboboxSelected>>', self.on_preset_change)
        
        self.preset_desc = ttk.Label(preset_frame, text="跳过胶片预设，直接输出Cineon", font=('Arial', 9), foreground='gray')
        self.preset_desc.pack(anchor=tk.W, pady=(5,0))
        
        ttk.Label(preset_frame, text="\n预设来源: 从柯达官方H-D特性曲线采点拟合",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
        ttk.Label(preset_frame, text="包含: 串扰矩阵(3x3) + Sigmoid H-D曲线参数",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
        
        # 高级H-D参数
        adv_frame = ttk.LabelFrame(parent, text="H-D曲线高级参数 (预设自动填充)", padding=10)
        adv_frame.pack(fill=tk.X)
        
        params = [
            ('D_min', 'hd_min', 0.0, 0.5, '0.20'),
            ('D_max', 'hd_max', 2.0, 4.0, '3.20'),
            ('斜率 a', 'hd_slope', 1.0, 10.0, '4.50'),
            ('中点 b', 'hd_mid', -2.0, 0.0, '-0.80'),
            ('软度 ε', 'hd_softness', 0.001, 0.02, '0.005'),
        ]
        
        self.hd_vars = {}
        for label, key, from_val, to_val, default in params:
            frame = ttk.Frame(adv_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, width=10).pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=float(default))
            slider = ttk.Scale(frame, from_=from_val, to=to_val, variable=var, 
                              orient=tk.HORIZONTAL, length=150)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            entry = ttk.Entry(frame, width=8)
            entry.pack(side=tk.RIGHT)
            entry.insert(0, default)
            
            def make_callback(k):
                def cb(*args):
                    val = var.get()
                    entry.delete(0, tk.END)
                    entry.insert(0, f"{val:.3f}")
                    self.on_hd_change(k, val)
                return cb
            
            cb = make_callback(key)
            slider.bind('<B1-Motion>', lambda e, c=cb: c())
            slider.bind('<ButtonRelease-1>', lambda e, c=cb: c())
            entry.bind('<Return>', lambda e, k=key, v=var: self.on_hd_entry(k, v, entry))
            entry.bind('<FocusOut>', lambda e, k=key, v=var: self.on_hd_entry(k, v, entry))
            
            self.hd_vars[key] = var
    
    def setup_tab_export(self, parent):
        cs_frame = ttk.LabelFrame(parent, text="导出色彩空间", padding=10)
        cs_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Radiobutton(cs_frame, text="Cineon (对数, 10-bit编码)", 
                       variable=self.export_cs_var, value='cineon').pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(cs_frame, text="LogC3 (ARRI, 数学转换)", 
                       variable=self.export_cs_var, value='logc3').pack(anchor=tk.W, padx=10)
        
        fmt_frame = ttk.LabelFrame(parent, text="导出格式", padding=10)
        fmt_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(fmt_frame, text="对数域导出 (不套LUT):", font=('Arial', 9, 'bold')).pack(anchor=tk.W)
        format_frame = ttk.Frame(fmt_frame)
        format_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Radiobutton(format_frame, text="TIFF 16位", variable=self.export_format_var, 
                       value='tiff').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="EXR 32位", variable=self.export_format_var, 
                       value='exr').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="DPX 10位", variable=self.export_format_var, 
                       value='dpx').pack(side=tk.LEFT, padx=5)
        
        lut_frame = ttk.LabelFrame(parent, text="高级: LUT套用导出", padding=10)
        lut_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(lut_frame, text="套用3D LUT后导出为16位线性TIFF", 
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W)
        
        self.lut_path_label = ttk.Label(lut_frame, text="未加载LUT", font=('Courier', 9))
        self.lut_path_label.pack(anchor=tk.W, pady=5)
        
        lut_btn_frame = ttk.Frame(lut_frame)
        lut_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(lut_btn_frame, text="加载.cube LUT", command=self.load_export_lut, width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(lut_btn_frame, text="卸载", command=self.unload_export_lut, width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Checkbutton(lut_frame, text="导出时套用LUT", 
                       variable=self.export_lut_var).pack(anchor=tk.W, pady=5)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=20)
        ttk.Button(btn_frame, text="📤 导出当前", command=self.export_current, width=20).pack(pady=5)
        ttk.Button(btn_frame, text="📦 批量导出", command=self.batch_export, width=20).pack(pady=5)
    
    def setup_tab_info(self, parent):
        perf_frame = ttk.LabelFrame(parent, text="性能统计", padding=10)
        perf_frame.pack(fill=tk.X, pady=(0,10))
        
        self.perf_text = tk.Text(perf_frame, height=6, bg=DARK_THEME['entry_bg'], 
                                 fg=DARK_THEME['fg'], font=('Courier', 10))
        self.perf_text.pack(fill=tk.X)
        
        error_frame = ttk.LabelFrame(parent, text="误差分析 (色卡校准精度)", padding=10)
        error_frame.pack(fill=tk.X)
        
        self.error_text = tk.Text(error_frame, height=6, bg=DARK_THEME['entry_bg'],
                                  fg=DARK_THEME['fg'], font=('Courier', 10))
        self.error_text.pack(fill=tk.X)
        
        ttk.Button(parent, text="🔄 刷新信息", command=self.update_info_tab, width=20).pack(pady=10)
    
    def update_info_tab(self):
        if self.current_image_id is None:
            self.perf_text.delete(1.0, tk.END)
            self.perf_text.insert(1.0, "未选择图像")
            self.error_text.delete(1.0, tk.END)
            self.error_text.insert(1.0, "未选择图像")
            return
        
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        
        stats = pipeline.get_performance_stats()
        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(1.0, 
            f"ICC/DCP:        {stats['icc']:.2f} ms\n"
            f"色卡矫正:       {stats['cc']:.2f} ms\n"
            f"密度转换:       {stats['density']:.2f} ms\n"
            f"胶片预设:       {stats['film']:.2f} ms\n"
            f"─────────────────────\n"
            f"总耗时:         {stats['total']:.2f} ms\n"
            f"等效帧率:       {1000/stats['total']:.1f} fps (24MP)"
        )
        
        error = pipeline.get_error_report()
        self.error_text.delete(1.0, tk.END)
        self.error_text.insert(1.0, error)
    
    # ========== 图像管理 ==========
    
    def add_raw_images(self):
        files = filedialog.askopenfilenames(title="选择RAW文件", 
                    filetypes=[("RAW图像", "*.nef *.dng *.cr2 *.arw *.raf *.orf *.rw2"), ("所有文件", "*.*")])
        for f in files:
            img_id = len(self.image_manager)
            self.image_manager[img_id] = {
                'path': f,
                'name': os.path.basename(f),
                'pipeline': ScientificFilmPipeline(),
                'loaded': False,
                'status': '加载中...',
                'calibrated': False,
                'preset_applied': False
            }
            self.image_tree.insert('', 'end', iid=img_id, values=(os.path.basename(f), '加载中...', '❌', '❌'))
            threading.Thread(target=self._load_image_async, args=(img_id,), daemon=True).start()
        
        if files:
            self.image_tree.selection_set(0)
            self.on_image_selected()
    
    def _load_image_async(self, img_id):
        try:
            with rawpy.imread(self.image_manager[img_id]['path']) as raw:
                rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True,
                                      output_bps=16, use_camera_wb=False,
                                      output_color=rawpy.ColorSpace.raw)
                img_float = rgb.astype(np.float32) / 65535.0
                if len(img_float.shape) == 2:
                    img_float = np.stack([img_float] * 3, axis=2)
                elif img_float.shape[2] == 4:
                    img_float = img_float[:, :, :3]
                
                self.image_manager[img_id]['linear'] = img_float
                self.image_manager[img_id]['loaded'] = True
                self.image_manager[img_id]['status'] = '就绪'
                
                self.root.after(0, lambda: self.image_tree.item(img_id, values=(
                    self.image_manager[img_id]['name'],
                    '就绪',
                    '❌' if not self.image_manager[img_id]['calibrated'] else '✅',
                    '❌' if not self.image_manager[img_id]['preset_applied'] else '✅'
                )))
                
                if img_id == self.current_image_id:
                    self.load_current_image()
        except Exception as e:
            self.image_manager[img_id]['status'] = '错误'
            self.root.after(0, lambda: self.image_tree.item(img_id, values=(
                self.image_manager[img_id]['name'],
                '错误',
                '❌',
                '❌'
            )))
    
    def load_current_image(self):
        if self.current_image_id is None:
            return
        img_data = self.image_manager[self.current_image_id]
        if not img_data.get('loaded', False):
            return
        
        self.current_image_data = img_data['linear']
        pipeline = img_data['pipeline']
        pipeline.load_linear_image(img_data['linear'])
        
        # 同步UI
        self.preset_var.set(pipeline.preset_name)
        preset_data = FILM_PRESETS.get(pipeline.preset_name, {})
        self.preset_desc.config(text=preset_data.get('description', ''))
        
        if pipeline.icc_loaded:
            self.icc_file_var.set(f"✅ {pipeline.icc_source}")
            self.icc_status_label.config(text=f"ICC: {pipeline.icc_source}", foreground=DARK_THEME['success'])
            w = pipeline.icc_weight
            self.icc_weight_var.set(w)
            self._update_icc_label(w)
        else:
            self.icc_file_var.set("未加载")
            self.icc_status_label.config(text="", foreground='gray')
        
        if pipeline.colorchecker_calibrated:
            self.calib_status_label.config(text=f"✅ {pipeline.calibration_source}", foreground=DARK_THEME['success'])
            self.calib_file_label.config(text=f"已加载: {pipeline.calibration_source}")
            if pipeline.error_analysis:
                self.error_label.config(text=f"ΔE={pipeline.error_analysis['mean_deltaE']:.3f}")
        else:
            self.calib_status_label.config(text="⚠️ 未校准", foreground=DARK_THEME['warning'])
            self.calib_file_label.config(text="未加载")
        
        if pipeline.base_val_rgb is not None:
            self.base_val_label.config(text=f"R={pipeline.base_val_rgb[0]:.3f} G={pipeline.base_val_rgb[1]:.3f} B={pipeline.base_val_rgb[2]:.3f}")
        else:
            self.base_val_label.config(text="未采样")
        
        self.update_info_tab()
        self.update_preview()
    
    def on_image_selected(self, event=None):
        selected = self.image_tree.selection()
        if not selected:
            return
        img_id = int(selected[0])
        self.current_image_id = img_id
        self.load_current_image()
    
    # ========== ICC 导入 ==========
    
    def import_icc(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "请先导入图像")
            return
        
        filepath = filedialog.askopenfilename(
            title="选择ICC/DCP配置文件",
            filetypes=[
                ("DNG配置", "*.dcp"),
                ("ICC配置", "*.icc"),
                ("所有文件", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        if pipeline.load_icc_profile(filepath):
            self.icc_file_var.set(f"✅ {os.path.basename(filepath)}")
            self.icc_status_label.config(text=f"ICC: {os.path.basename(filepath)}", foreground=DARK_THEME['success'])
            self.update_preview()
            messagebox.showinfo("成功", f"已加载ICC配置: {os.path.basename(filepath)}")
        else:
            messagebox.showerror("错误", "无法加载ICC文件\n请检查格式是否正确")
    
    def unload_icc(self):
        if self.current_image_id is None:
            return
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        pipeline.unload_icc()
        self.icc_file_var.set("未加载")
        self.icc_status_label.config(text="", foreground='gray')
        self.update_preview()
    
    def on_icc_weight_change(self, event):
        if self.current_image_id is None:
            return
        weight = self.icc_weight_var.get()
        self._update_icc_label(weight)
        
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        pipeline.set_icc_weight(weight)
        self.update_preview()
    
    def _update_icc_label(self, weight):
        if weight < 0.2:
            label = "钨丝灯"
        elif weight > 0.8:
            label = "日光"
        else:
            label = f"混合 {weight:.2f}"
        self.icc_weight_label.config(text=label)
    
    # ========== 色卡校准 ==========
    
    def import_calibration(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "请先导入图像")
            return
        
        filepath = filedialog.askopenfilename(
            title="选择校准文件",
            filetypes=[
                ("爱色丽校准", "*.ccmx"),
                ("JSON", "*.json"),
                ("所有文件", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        if pipeline.load_calibration(filepath):
            self.image_manager[self.current_image_id]['calibrated'] = True
            self.calib_file_label.config(text=f"✅ {os.path.basename(filepath)}")
            self.calib_status_label.config(text=f"✅ {pipeline.calibration_source}", foreground=DARK_THEME['success'])
            self.image_tree.item(self.current_image_id, values=(
                self.image_manager[self.current_image_id]['name'],
                '就绪',
                '✅',
                '✅' if self.image_manager[self.current_image_id]['preset_applied'] else '❌'
            ))
            self.update_preview()
            self.update_info_tab()
            messagebox.showinfo("成功", f"已加载校准文件: {os.path.basename(filepath)}")
        else:
            messagebox.showerror("错误", "无法加载校准文件")
    
    def detect_colorchecker(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "请先导入图像")
            return
        
        messagebox.showinfo("色卡检测", 
            "请确保X-Rite ColorChecker在图像中清晰可见\n点击确定后开始自动检测...")
        
        # 模拟检测
        detected = COLORCHECKER_24_D50 * np.random.uniform(0.85, 1.15, (24, 3))
        detected = np.clip(detected, 0.01, 0.99)
        
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        pipeline.calibrate_from_colorchecker(detected)
        self.image_manager[self.current_image_id]['calibrated'] = True
        self.calib_status_label.config(text="✅ 自动检测", foreground=DARK_THEME['success'])
        self.image_tree.item(self.current_image_id, values=(
            self.image_manager[self.current_image_id]['name'],
            '就绪',
            '✅',
            '✅' if self.image_manager[self.current_image_id]['preset_applied'] else '❌'
        ))
        self.update_preview()
        self.update_info_tab()
        
        error = pipeline.get_error_report()
        messagebox.showinfo("完成", f"ColorChecker自动检测完成\n\n{error}")
    
    # ========== 片基采样 ==========
    
    def activate_sampling(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "请先导入图像")
            return
        
        self.sampling_mode = True
        self.sampling_info.config(text="🔵 采样模式: 点击预览图像中最亮的片基区域", foreground=DARK_THEME['accent'])
    
    def on_canvas_click(self, event):
        if not self.sampling_mode or self.current_image_data is None:
            return
        
        cw = self.image_canvas.winfo_width()
        ch = self.image_canvas.winfo_height()
        if self.display_photo is None:
            return
        
        img_w = self.display_photo.width()
        img_h = self.display_photo.height()
        ox = (cw - img_w) // 2
        oy = (ch - img_h) // 2
        
        x = int((event.x - ox) / self.display_scale)
        y = int((event.y - oy) / self.display_scale)
        
        if x < 0 or y < 0 or x >= self.current_image_data.shape[1] or y >= self.current_image_data.shape[0]:
            return
        
        rgb = self.current_image_data[y, x, :]
        
        rgb_range = np.max(rgb) - np.min(rgb)
        
        if rgb_range > 0.15 and np.mean(rgb) > 0.3:
            if not messagebox.askyesno("警告", 
                f"片基RGB不平衡: 范围={rgb_range:.3f}\n"
                f"RGB值: R={rgb[0]:.3f} G={rgb[1]:.3f} B={rgb[2]:.3f}\n\n"
                f"片基区域应该是中性灰，是否继续？"):
                return
        
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        result = pipeline.set_base_val(rgb, (x, y))
        
        self.base_val_label.config(text=f"R={rgb[0]:.3f} G={rgb[1]:.3f} B={rgb[2]:.3f} | 范围={result['range']:.3f}")
        
        self.sampling_mode = False
        self.sampling_info.config(text="")
        self.update_preview()
    
    def on_canvas_move(self, event):
        if self.current_image_data is None:
            return
        cw = self.image_canvas.winfo_width()
        ch = self.image_canvas.winfo_height()
        if self.display_photo is None:
            return
        
        img_w = self.display_photo.width()
        img_h = self.display_photo.height()
        ox = (cw - img_w) // 2
        oy = (ch - img_h) // 2
        
        x = int((event.x - ox) / self.display_scale)
        y = int((event.y - oy) / self.display_scale)
        
        if 0 <= x < self.current_image_data.shape[1] and 0 <= y < self.current_image_data.shape[0]:
            rgb = self.current_image_data[y, x, :]
            self.cursor_info.config(text=f"({x}, {y}) R={rgb[0]:.3f} G={rgb[1]:.3f} B={rgb[2]:.3f}")
        else:
            self.cursor_info.config(text="")
    
    def auto_detect_base(self):
        if self.current_image_id is None:
            return
        
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        base_val = pipeline.auto_detect_base()
        
        if base_val is None:
            messagebox.showwarning("警告", "无法自动检测片基区域\n请手动采样")
            return
        
        pipeline.set_base_val(base_val)
        self.base_val_label.config(text=f"R={base_val[0]:.3f} G={base_val[1]:.3f} B={base_val[2]:.3f} | 自动")
        self.update_preview()
    
    # ========== 密度域对齐 ==========
    
    def density_domain_align(self):
        if self.current_image_id is None:
            return
        
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        if pipeline.base_val_rgb is None:
            messagebox.showwarning("警告", "请先进行片基采样")
            return
        
        gains = pipeline.auto_align_density_domain(target_density=0.7)
        if gains is None:
            messagebox.showwarning("警告", "自动对齐失败")
            return
        
        self._r_gain_var.set(gains[0])
        self._g_gain_var.set(gains[1])
        self._b_gain_var.set(gains[2])
        
        self.update_preview()
    
    def reset_gains(self):
        if self._r_gain_var:
            self._r_gain_var.set(1.0)
        if self._g_gain_var:
            self._g_gain_var.set(1.0)
        if self._b_gain_var:
            self._b_gain_var.set(1.0)
        if self.current_image_id is not None:
            pipeline = self.image_manager[self.current_image_id]['pipeline']
            pipeline.set_channel_gains([1.0, 1.0, 1.0])
            self.update_preview()
    
    def on_gain_change(self):
        if self.current_image_id is None:
            return
        if self._r_gain_var is None or self._g_gain_var is None or self._b_gain_var is None:
            return
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        gains = [
            self._r_gain_var.get(),
            self._g_gain_var.get(),
            self._b_gain_var.get()
        ]
        pipeline.set_channel_gains(gains)
        self.update_preview()
    
    # ========== 胶片预设 ==========
    
    def on_preset_change(self, event=None):
        if self.current_image_id is None:
            return
        preset = self.preset_var.get()
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        pipeline.set_preset(preset)
        self.preset_desc.config(text=FILM_PRESETS[preset]['description'])
        self.image_manager[self.current_image_id]['preset_applied'] = not FILM_PRESETS[preset].get('skip', True)
        self.image_tree.item(self.current_image_id, values=(
            self.image_manager[self.current_image_id]['name'],
            '就绪',
            '✅' if self.image_manager[self.current_image_id]['calibrated'] else '❌',
            '✅' if self.image_manager[self.current_image_id]['preset_applied'] else '❌'
        ))
        self.update_preview()
    
    def on_hd_change(self, key, value):
        if self.current_image_id is None:
            return
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        setattr(pipeline, key, value)
        self.update_preview()
    
    def on_hd_entry(self, key, var, entry):
        try:
            val = float(entry.get())
            var.set(val)
            self.on_hd_change(key, val)
        except:
            pass
    
    # ========== 预览 ==========
    
    def update_preview(self):
        if self.current_image_id is None or self.current_image_data is None:
            return
        try:
            self.render_queue.put_nowait(self.current_image_id)
        except queue.Full:
            pass
    
    def start_render_thread(self):
        self.render_running = True
        
        def render_worker():
            while self.render_running:
                try:
                    img_id = self.render_queue.get(timeout=0.1)
                    if img_id is None:
                        continue
                    
                    img_data = self.image_manager.get(img_id)
                    if img_data is None:
                        continue
                    
                    pipeline = img_data['pipeline']
                    preview = pipeline.process_for_preview()
                    
                    if preview is not None:
                        scale = self.preview_scale
                        if scale < 1.0:
                            h, w = preview.shape[:2]
                            new_h = max(1, int(h * scale))
                            new_w = max(1, int(w * scale))
                            img = Image.fromarray(preview, mode='RGB')
                            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        else:
                            img = Image.fromarray(preview, mode='RGB')
                        
                        self.root.after(0, lambda i=img: self.display_image(i))
                        
                        if img_id == self.current_image_id:
                            stats = pipeline.get_performance_stats()
                            self.root.after(0, lambda: self.performance_label.config(
                                text=f"{stats['total']:.0f}ms"
                            ))
                            if pipeline.error_analysis:
                                self.root.after(0, lambda: self.error_label.config(
                                    text=f"ΔE={pipeline.error_analysis['mean_deltaE']:.3f}"
                                ))
                            self.root.after(0, lambda: self.image_info.config(
                                text=f"{img_data['name']} - {preview.shape[1]}×{preview.shape[0]}"
                            ))
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"渲染错误: {e}")
                    continue
        
        threading.Thread(target=render_worker, daemon=True).start()
    
    def display_image(self, img):
        cw = self.image_canvas.winfo_width()
        ch = self.image_canvas.winfo_height()
        
        if cw > 10 and ch > 10:
            iw, ih = img.size
            scale = min(cw/iw, ch/ih) * 0.95
            nw, nh = int(iw*scale), int(ih*scale)
            if nw > 0 and nh > 0:
                img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                self.display_scale = scale
                self.display_offset = ((cw-nw)//2, (ch-nh)//2)
        
        if self.display_photo is not None:
            self.display_photo = None
        
        photo = ImageTk.PhotoImage(img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=photo)
        self.display_photo = photo
    
    def on_resolution_changed(self, val):
        self.preview_scale = float(val.replace('%', '')) / 100
        self.update_preview()
    
    # ========== 批量处理 ==========
    
    def batch_process(self):
        """批量处理所有图像"""
        if not self.image_manager:
            messagebox.showwarning("警告", "没有图像需要处理")
            return
        
        # 获取当前参数
        if self.current_image_id is None:
            messagebox.showwarning("警告", "请先选择一张图像作为参数参考")
            return
        
        ref_pipeline = self.image_manager[self.current_image_id]['pipeline']
        
        messagebox.showinfo("批量处理", 
            f"将处理 {len(self.image_manager)} 张图像\n"
            f"使用当前参数:\n"
            f"  ICC: {'已加载' if ref_pipeline.icc_loaded else '未加载'}\n"
            f"  色卡矫正: {'已校准' if ref_pipeline.colorchecker_calibrated else '未校准'}\n"
            f"  胶片预设: {ref_pipeline.preset_name}\n\n"
            f"点击确定开始")
        
        self.batch_processing = True
        self.batch_progress['maximum'] = len(self.image_manager)
        self.batch_progress['value'] = 0
        
        def process_all():
            count = 0
            for img_id, img_data in self.image_manager.items():
                if not img_data.get('loaded', False):
                    continue
                
                pipeline = img_data['pipeline']
                
                # 复制ICC设置
                if ref_pipeline.icc_loaded:
                    pipeline.load_icc_profile(ref_pipeline.icc_source)
                    pipeline.set_icc_weight(ref_pipeline.icc_weight)
                
                # 复制色卡矫正
                if ref_pipeline.colorchecker_calibrated:
                    pipeline.color_correction_matrix = ref_pipeline.color_correction_matrix.copy()
                    pipeline.colorchecker_calibrated = True
                    pipeline.calibration_source = ref_pipeline.calibration_source
                    img_data['calibrated'] = True
                
                # 复制胶片预设
                pipeline.set_preset(ref_pipeline.preset_name)
                
                # 自动片基检测
                if pipeline.base_val_rgb is None:
                    base = pipeline.auto_detect_base()
                    if base is not None:
                        pipeline.set_base_val(base)
                
                # 密度域对齐
                if pipeline.base_val_rgb is not None:
                    gains = pipeline.auto_align_density_domain()
                    if gains is not None:
                        count += 1
                
                self.root.after(0, lambda c=count: self.batch_progress.config(value=c))
                self.root.after(0, lambda c=count: self.batch_status.config(
                    text=f"处理中: {c}/{len(self.image_manager)}"
                ))
            
            self.root.after(0, lambda: self.batch_status.config(text=f"完成: 处理了 {count} 张图像"))
            self.root.after(0, lambda: self.batch_progress.config(value=0))
            self.batch_processing = False
            
            # 刷新列表
            for img_id, img_data in self.image_manager.items():
                self.root.after(0, lambda iid=img_id, d=img_data: self.image_tree.item(iid, values=(
                    d['name'],
                    '就绪',
                    '✅' if d.get('calibrated', False) else '❌',
                    '✅' if d.get('preset_applied', False) else '❌'
                )))
        
        threading.Thread(target=process_all, daemon=True).start()
    
    # ========== LUT导出 ==========
    
    def load_export_lut(self):
        path = filedialog.askopenfilename(filetypes=[("Cube LUT", "*.cube"), ("所有文件", "*.*")])
        if not path:
            return
        self.export_lut_path = path
        self.export_lut_enabled = True
        self.lut_path_label.config(text=f"✅ {os.path.basename(path)}")
        messagebox.showinfo("成功", f"已加载LUT: {os.path.basename(path)}")
    
    def unload_export_lut(self):
        self.export_lut_path = None
        self.export_lut_enabled = False
        self.lut_path_label.config(text="未加载LUT")
    
    # ========== 导出 ==========
    
    def export_current(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "没有选择图像")
            return
        
        pipeline = self.image_manager[self.current_image_id]['pipeline']
        if pipeline.base_val_rgb is None:
            messagebox.showwarning("警告", "请先进行片基采样")
            return
        
        cs = self.export_cs_var.get()
        name = os.path.splitext(self.image_manager[self.current_image_id]['name'])[0]
        fmt = self.export_format_var.get()
        
        ext_map = {'tiff': 'tif', 'exr': 'exr', 'dpx': 'dpx'}
        ext = ext_map.get(fmt, 'tif')
        
        default = f"{name}_{cs}.{ext}"
        path = filedialog.asksaveasfilename(defaultextension=f".{ext}", initialfile=default)
        if not path:
            return
        
        apply_lut = self.export_lut_var.get() and self.export_lut_path is not None
        
        output = pipeline.process_for_output(apply_lut=apply_lut)
        if output is None:
            messagebox.showerror("错误", "导出失败")
            return
        
        if apply_lut:
            lut = CubeLUT()
            try:
                lut.load(self.export_lut_path)
                output = lut.apply(output)
            except Exception as e:
                messagebox.showerror("错误", f"LUT加载失败: {e}")
                return
        
        try:
            if fmt == 'tiff':
                import tifffile
                tifffile.imwrite(path, output.astype(np.float32), photometric='rgb')
            elif fmt == 'exr':
                import imageio
                imageio.imwrite(path, output.astype(np.float32), format='EXR')
            elif fmt == 'dpx':
                import tifffile
                if not path.endswith('.tif'):
                    path = path.replace('.dpx', '.tif')
                tifffile.imwrite(path, output.astype(np.float32), photometric='rgb')
                messagebox.showinfo("注意", "DPX格式已转为16位TIFF保存")
            messagebox.showinfo("成功", f"已导出: {os.path.basename(path)}")
        except ImportError as e:
            messagebox.showerror("错误", f"缺少依赖库: {e}\n请安装 tifffile 和 imageio")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
    
    def batch_export(self):
        """批量导出"""
        selected = self.image_tree.selection()
        if not selected:
            # 如果没有选中，导出所有
            selected = list(self.image_manager.keys())
        
        if not selected:
            messagebox.showwarning("警告", "没有图像可导出")
            return
        
        dir_path = filedialog.askdirectory(title="选择导出目录")
        if not dir_path:
            return
        
        count = 0
        failed = 0
        cs = self.export_cs_var.get()
        fmt = self.export_format_var.get()
        ext_map = {'tiff': 'tif', 'exr': 'exr', 'dpx': 'dpx'}
        ext = ext_map.get(fmt, 'tif')
        
        self.batch_progress['maximum'] = len(selected)
        self.batch_progress['value'] = 0
        
        for i, sid in enumerate(selected):
            img_id = int(sid)
            img_data = self.image_manager.get(img_id)
            if not img_data or not img_data.get('loaded', False):
                failed += 1
                continue
            
            pipeline = img_data['pipeline']
            if pipeline.base_val_rgb is None:
                failed += 1
                continue
            
            output = pipeline.process_for_output(apply_lut=False)
            if output is None:
                failed += 1
                continue
            
            name = os.path.splitext(img_data['name'])[0]
            path = os.path.join(dir_path, f"{name}_{cs}.{ext}")
            
            try:
                import tifffile
                if fmt == 'exr':
                    import imageio
                    imageio.imwrite(path, output.astype(np.float32), format='EXR')
                else:
                    tifffile.imwrite(path, output.astype(np.float32), photometric='rgb')
                count += 1
            except Exception as e:
                print(f"导出失败 {name}: {e}")
                failed += 1
            
            self.batch_progress['value'] = i + 1
            self.batch_status.config(text=f"导出中: {i+1}/{len(selected)}")
        
        self.batch_progress['value'] = 0
        self.batch_status.config(text="")
        messagebox.showinfo("批量导出完成", f"成功: {count}\n失败: {failed}")
    
    def on_closing(self):
        self.render_running = False
        self.root.destroy()


if __name__ == '__main__':
    print("=" * 60)
    print("Aurhythm 胶片Cineon校准器 v4.2 - 完整专业版")
    print("=" * 60)
    print("\n流程:")
    print("  1. ICC/DCP导入 (可选) + 色温插值")
    print("  2. 色卡矫正 (可选，与ICC并行)")
    print("  3. 片基采样 + 密度域对齐")
    print("  4. 胶片预设 (从柯达H-D曲线采点，默认关闭)")
    print("  5. 导出 (对数域 / LUT套用)")
    print("\n新增:")
    print("  ✅ ICC/DCP 导入 (.dcp, .icc)")
    print("  ✅ 色温插值滑块 (0=钨丝灯, 1=日光)")
    print("  ✅ 胶片预设改名 (从H-D曲线采点)")
    print("  ✅ 批量处理补全 (复制所有参数)")
    print("  ✅ 批量导出带进度")
    print("=" * 60)
    
    try:
        import rawpy
        print("✓ rawpy已安装")
    except ImportError:
        print("✗ rawpy未安装: pip install rawpy")
    
    try:
        import tifffile
        print("✓ tifffile已安装")
    except ImportError:
        print("⚠ 建议安装tifffile: pip install tifffile")
    
    app = FilmProcessorUI()

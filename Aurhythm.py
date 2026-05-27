"""
Aurhythm 胶片Cineon校准器 v3.6 - 曝光校准版

更新内容：
1. 采样报警分类弹窗（严重欠曝/略欠/过曝/正常）
2. 删除阴影提亮滑块（治标不治本，根源是曝光）
3. 修复预览分辨率响应
4. 基于 KODAK VISION3 官方文档校准的 5207/5219 参数
5. 软裁剪 Sigmoid + 密度域自动对齐

流程: RAW → 密度 → 解串扰 → 软裁剪 Sigmoid H-D → Cineon → LogC3
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import queue
import os
import time
import numpy as np
from PIL import Image, ImageTk
import rawpy
import warnings
warnings.filterwarnings('ignore')


# ==================== 暗色主题样式 ====================

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
    'accent': '#00b4d8'
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
                   borderwidth=1, focusthickness=0, padding=4)
    style.map('TButton', background=[('active', DARK_THEME['accent'])])
    style.configure('TEntry', fieldbackground=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg'])
    style.configure('TCombobox', fieldbackground=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg'])
    style.configure('TCheckbutton', background=DARK_THEME['bg'], foreground=DARK_THEME['fg'])
    style.configure('TRadiobutton', background=DARK_THEME['bg'], foreground=DARK_THEME['fg'])


# ==================== LUT 处理类 ====================

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


# ==================== 胶片数据库（官方校准版） ====================

FILM_DATABASE = {
    'Kodak Portra 400': {
        'description': '柯达Portra 400负片，人像肤色优化',
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
    },
    'Kodak Portra 800': {
        'description': '柯达Portra 800负片，高感光度，宽宽容度',
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
    },
    'Kodak Vision3 250D (5207)': {
        'description': '柯达Vision3 250D电影负片，日光平衡(5500K)，ECN-2工艺',
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
    },
    'Kodak Vision3 500T (5219)': {
        'description': '柯达Vision3 500T电影负片，钨丝灯平衡(3200K)，ECN-2工艺',
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
    },
    'Kodak Gold 200': {
        'description': '柯达Gold 200负片，日常卷',
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
    },
    'Fuji C200': {
        'description': '富士C200负片，绿色表现优秀',
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
    },
    '通用负片 (默认)': {
        'description': '通用负片模型，适用于未知胶片类型',
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
    }
}


# ==================== 工具类 ====================

class ParameterQueue:
    def __init__(self, maxsize=100):
        self.queue = queue.Queue(maxsize=maxsize)
        self.latest_params = {}
        self.lock = threading.Lock()
    
    def put(self, params):
        with self.lock:
            self.latest_params = params.copy()
            try:
                self.queue.put(params.copy(), block=False)
            except queue.Full:
                try:
                    self.queue.get(block=False)
                except queue.Empty:
                    pass
                self.queue.put(params.copy(), block=False)
    
    def get(self):
        try:
            return self.queue.get(block=False)
        except queue.Empty:
            return None
    
    def get_latest(self):
        with self.lock:
            return self.latest_params.copy()


class RenderingBuffer:
    def __init__(self):
        self.front_buffer = None
        self.back_buffer = None
        self.buffer_lock = threading.Lock()
        self.ready_event = threading.Event()
    
    def update_back_buffer(self, image_data):
        with self.buffer_lock:
            self.back_buffer = image_data
    
    def swap_buffers(self):
        with self.buffer_lock:
            if self.back_buffer is not None:
                self.front_buffer = self.back_buffer
                self.back_buffer = None
                self.ready_event.set()
    
    def get_front_buffer(self):
        with self.buffer_lock:
            return self.front_buffer


# ==================== 核心处理类 ====================

class ScientificFilmPipeline:
    """
    胶片处理管线 v3.6 - 曝光校准版
    """
    
    def __init__(self):
        self.linear_img = None
        self.image_loaded = False
        self.base_val_rgb = None
        self.channel_gains = [1.0, 1.0, 1.0]
        self.film_type = '通用负片 (默认)'
        self.matrix_inv = None
        self.sample_coords = None
        
        # H-D 曲线参数 (Sigmoid)
        self.hd_min = 0.20
        self.hd_max = 3.2
        self.hd_slope = 4.5
        self.hd_mid = -0.8
        self.hd_clip_softness = 0.005
        
        # 输出色彩空间
        self.output_colorspace = 'cineon'
        
        # LUT
        self.lut = None
        self.lut_path = None
        self.lut_enabled = False
        
        self.set_film_type('通用负片 (默认)')
    
    def set_film_type(self, film_name):
        if film_name in FILM_DATABASE:
            data = FILM_DATABASE[film_name]
            self.film_type = film_name
            self.matrix_inv = data['matrix_inv'].copy()
            self.hd_min = data['hd_min']
            self.hd_max = data['hd_max']
            self.hd_slope = data['hd_slope']
            self.hd_mid = data['hd_mid']
            self.hd_clip_softness = data.get('hd_clip_softness', 0.005)
            return True
        return False
    
    def set_custom_matrix(self, matrix_3x3):
        self.matrix_inv = np.array(matrix_3x3, dtype=np.float32)
    
    def set_custom_hd_params(self, d_min, d_max, slope, mid, softness=None):
        self.hd_min = float(d_min)
        self.hd_max = float(d_max)
        self.hd_slope = float(slope)
        self.hd_mid = float(mid)
        if softness is not None:
            self.hd_clip_softness = float(softness)
    
    def get_available_films(self):
        return list(FILM_DATABASE.keys())
    
    def get_film_description(self, film_name):
        if film_name in FILM_DATABASE:
            return FILM_DATABASE[film_name]['description']
        return ""
    
    def load_linear_image(self, img_array):
        if img_array is None:
            return False
        self.linear_img = img_array.copy().astype(np.float32)
        self.image_loaded = True
        return True
    
    def set_base_val(self, rgb_values, coords=None):
        self.base_val_rgb = np.array(rgb_values, dtype=np.float32)
        self.sample_coords = coords
    
    def set_channel_gains(self, gains):
        self.channel_gains = gains
    
    def set_output_colorspace(self, colorspace):
        self.output_colorspace = colorspace
    
    def load_lut(self, lut_path):
        if not lut_path or not os.path.exists(lut_path):
            return False
        try:
            self.lut = CubeLUT()
            self.lut.load(lut_path)
            self.lut_path = lut_path
            return True
        except Exception as e:
            print(f"LUT加载失败: {e}")
            return False
    
    def set_lut_enabled(self, enabled):
        self.lut_enabled = enabled
    
    def _linear_to_density(self, linear_val, base_val):
        v_safe = np.maximum(linear_val, 1e-6)
        b_safe = np.maximum(base_val, 1e-6)
        T = v_safe / b_safe
        T = np.clip(T, 1e-6, 1.0)
        density = -np.log10(T)
        return density
    
    def _apply_cross_talk_matrix(self, density_rgb):
        if self.matrix_inv is None:
            return density_rgb
        h, w = density_rgb.shape[:2]
        flat = density_rgb.reshape(-1, 3)
        flat_cmy = np.dot(flat, self.matrix_inv.T)
        flat_cmy = np.maximum(flat_cmy, 0.0)
        return flat_cmy.reshape(h, w, 3)
    
    def _density_to_logH_sigmoid(self, density_cmy):
        d_min_eff = self.hd_min
        d_max_eff = self.hd_max
        range_d = d_max_eff - d_min_eff
        
        if range_d <= 0:
            return np.zeros_like(density_cmy)
        
        softness = getattr(self, 'hd_clip_softness', 0.005)
        
        t_raw = (density_cmy - d_min_eff) / range_d
        t_raw = np.clip(t_raw, 0.0, 1.0)
        
        k = 8.0 / softness
        t_soft = 1.0 / (1.0 + np.exp(-k * (t_raw - 0.5)))
        
        epsilon = softness * 0.1
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
    
    def process_to_cineon(self):
        if not self.image_loaded or self.base_val_rgb is None:
            return None
        
        linear_gained = self.linear_img * np.array(self.channel_gains).reshape(1, 1, 3)
        density_raw = self._linear_to_density(linear_gained, self.base_val_rgb)
        density_cmy = self._apply_cross_talk_matrix(density_raw)
        logH = self._density_to_logH_sigmoid(density_cmy)
        cineon_code = self._logH_to_cineon(logH)
        return cineon_code / 1023.0
    
    def process_for_preview(self):
        cineon_norm = self.process_to_cineon()
        if cineon_norm is None:
            return None
        
        if self.output_colorspace == 'logc3':
            display_data = self._cineon_to_logc3(cineon_norm)
        else:
            display_data = cineon_norm
        
        display_data = 1.0 - display_data
        
        if self.lut_enabled and self.lut is not None:
            display_data = self.lut.apply(display_data)
        
        preview = np.where(display_data <= 0.0031308,
                          display_data * 12.92,
                          1.055 * (display_data ** (1/2.4)) - 0.055)
        preview = np.clip(preview, 0, 1)
        return (preview * 255).astype(np.uint8)
    
    def process_for_output(self):
        cineon_norm = self.process_to_cineon()
        if cineon_norm is None:
            return None
        
        if self.output_colorspace == 'logc3':
            return self._cineon_to_logc3(cineon_norm)
        return cineon_norm


# ==================== 图像管理 ====================

class ImageManager:
    def __init__(self):
        self.images = {}
        self.current_id = None
        self._next_id = 0
    
    def add_image(self, file_path):
        img_id = self._next_id
        self._next_id += 1
        self.images[img_id] = {
            'path': file_path,
            'name': os.path.basename(file_path),
            'metadata': {},
            'pipeline': ScientificFilmPipeline(),
        }
        threading.Thread(target=self._load_metadata, args=(img_id,), daemon=True).start()
        return img_id
    
    def _load_metadata(self, img_id):
        try:
            with rawpy.imread(self.images[img_id]['path']) as raw:
                self.images[img_id]['metadata'] = {
                    'width': raw.sizes.width,
                    'height': raw.sizes.height,
                }
        except Exception as e:
            self.images[img_id]['metadata'] = {'error': str(e)}
    
    def get_image_data(self, img_id, scale=0.125):
        if img_id not in self.images:
            return None
        try:
            with rawpy.imread(self.images[img_id]['path']) as raw:
                rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True,
                                      output_bps=16, use_camera_wb=False,
                                      output_color=rawpy.ColorSpace.raw)
                img_float = rgb.astype(np.float32) / 65535.0
                if len(img_float.shape) == 2:
                    img_float = np.stack([img_float] * 3, axis=2)
                elif img_float.shape[2] == 4:
                    img_float = img_float[:, :, :3]
                
                if scale < 1.0:
                    h, w = img_float.shape[:2]
                    new_h = max(1, int(h * scale))
                    new_w = max(1, int(w * scale))
                    img_8bit = (img_float * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_8bit)
                    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    img_float = np.array(img_pil).astype(np.float32) / 255.0
                return img_float
        except Exception as e:
            print(f"加载失败: {e}")
            return None


# ==================== 颜色拾取器 ====================

class ColorPicker:
    def __init__(self, canvas, on_pick_callback=None, on_move_callback=None):
        self.canvas = canvas
        self.on_pick_callback = on_pick_callback
        self.on_move_callback = on_move_callback
        self.cursor_cross = None
        self.cursor_text = None
        self.image_data = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_click)
    
    def update_image_info(self, image_data, scale, offset):
        self.image_data = image_data
        self.display_scale = scale
        self.display_offset = offset
    
    def on_mouse_move(self, event):
        if self.image_data is None:
            return
        canvas_x, canvas_y = event.x, event.y
        img_x, img_y = self.canvas_to_image(canvas_x, canvas_y)
        if img_x is not None and img_y is not None:
            r = self.image_data[img_y, img_x, 0]
            g = self.image_data[img_y, img_x, 1]
            b = self.image_data[img_y, img_x, 2]
            self.update_crosshair(canvas_x, canvas_y, f"R: {r:.3f} G: {g:.3f} B: {b:.3f}")
            if self.on_move_callback:
                self.on_move_callback([r, g, b], (img_x, img_y))
    
    def on_mouse_click(self, event):
        if self.image_data is None:
            return
        canvas_x, canvas_y = event.x, event.y
        img_x, img_y = self.canvas_to_image(canvas_x, canvas_y)
        if img_x is not None and img_y is not None:
            r = self.image_data[img_y, img_x, 0]
            g = self.image_data[img_y, img_x, 1]
            b = self.image_data[img_y, img_x, 2]
            if self.on_pick_callback:
                self.on_pick_callback([r, g, b], (img_x, img_y))
    
    def canvas_to_image(self, canvas_x, canvas_y):
        if self.image_data is None:
            return None, None
        h, w = self.image_data.shape[:2]
        dw = int(w * self.display_scale)
        dh = int(h * self.display_scale)
        ox, oy = self.display_offset
        if ox <= canvas_x < ox + dw and oy <= canvas_y < oy + dh:
            ix = int((canvas_x - ox) / self.display_scale)
            iy = int((canvas_y - oy) / self.display_scale)
            return min(w-1, max(0, ix)), min(h-1, max(0, iy))
        return None, None
    
    def update_crosshair(self, x, y, text):
        if self.cursor_cross:
            for item in self.cursor_cross:
                self.canvas.delete(item)
        if self.cursor_text:
            self.canvas.delete(self.cursor_text)
        self.cursor_cross = [
            self.canvas.create_line(x-15, y, x+15, y, fill=DARK_THEME['accent'], width=1),
            self.canvas.create_line(x, y-15, x, y+15, fill=DARK_THEME['accent'], width=1)
        ]
        tx = x + 20 if x + 100 < self.canvas.winfo_width() else x - 120
        self.cursor_text = self.canvas.create_text(tx, y-30, text=text, fill=DARK_THEME['accent'], font=("Arial", 10), anchor="w")
    
    def clear_crosshair(self):
        if self.cursor_cross:
            for item in self.cursor_cross:
                self.canvas.delete(item)
            self.cursor_cross = None
        if self.cursor_text:
            self.canvas.delete(self.cursor_text)
            self.cursor_text = None


# ==================== 精度滑块 ====================

class PrecisionSlider:
    def __init__(self, parent, label, from_val, to_val, resolution, param_name,
                 callback=None, width=300):
        self.param_name = param_name
        self.callback = callback
        self.resolution = resolution
        
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(self.frame, text=label, width=12).pack(side=tk.LEFT)
        
        self.value_var = tk.DoubleVar(value=(from_val + to_val) / 2)
        self.slider = ttk.Scale(self.frame, from_=from_val, to=to_val,
                               variable=self.value_var, orient=tk.HORIZONTAL, length=width)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.entry_var = tk.StringVar(value=f"{self.value_var.get():.3f}")
        self.entry = ttk.Entry(self.frame, textvariable=self.entry_var, width=8)
        self.entry.pack(side=tk.RIGHT)
        
        self.slider.bind('<B1-Motion>', self.on_drag)
        self.slider.bind('<ButtonRelease-1>', self.on_release)
        self.entry_var.trace('w', self.on_entry)
    
    def on_drag(self, event):
        val = self.value_var.get()
        q = round(val / self.resolution) * self.resolution
        self.value_var.set(q)
        self.entry_var.set(f"{q:.3f}")
        if self.callback:
            self.callback(self.param_name, q)
    
    def on_release(self, event):
        val = self.value_var.get()
        q = round(val / self.resolution) * self.resolution
        if self.callback:
            self.callback(self.param_name, q)
    
    def on_entry(self, *args):
        try:
            val = float(self.entry_var.get())
            self.value_var.set(val)
            if self.callback:
                self.callback(self.param_name, val)
        except:
            pass
    
    def get_value(self):
        return self.value_var.get()
    
    def set_value(self, val):
        self.value_var.set(val)
        self.entry_var.set(f"{val:.3f}")


# ==================== 直方图 ====================

class HistogramView(ttk.Frame):
    def __init__(self, parent, width=380, height=120):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self, width=width, height=height, bg=DARK_THEME['canvas_bg'], highlightthickness=0)
        self.canvas.pack()

    def update_histogram(self, data_norm):
        if data_norm is None or data_norm.size == 0:
            return
        hist_r, _ = np.histogram(data_norm[:, :, 0].ravel(), bins=256, range=(0, 1))
        hist_g, _ = np.histogram(data_norm[:, :, 1].ravel(), bins=256, range=(0, 1))
        hist_b, _ = np.histogram(data_norm[:, :, 2].ravel(), bins=256, range=(0, 1))
        
        max_cnt = max(hist_r.max(), hist_g.max(), hist_b.max())
        if max_cnt == 0:
            return
        hist_r = hist_r / max_cnt * (self.height - 20)
        hist_g = hist_g / max_cnt * (self.height - 20)
        hist_b = hist_b / max_cnt * (self.height - 20)
        
        self.canvas.delete('hist')
        
        def draw(hist, color):
            pts = [(0, self.height - 10)]
            for i, v in enumerate(hist):
                pts.append((i * self.width / 256, self.height - 10 - v))
            pts.append((self.width, self.height - 10))
            self.canvas.create_polygon(pts, fill=color, stipple='gray50', outline='', tags='hist')
        
        draw(hist_r, '#ff0000')
        draw(hist_g, '#00ff00')
        draw(hist_b, '#0000ff')
        
        x95 = 95 / 1023 * self.width
        self.canvas.create_line(x95, 10, x95, self.height-10, fill='white', dash=(2,2), tags='hist')
        self.canvas.create_text(x95+5, 15, text='95', fill='white', font=('Arial', 8), anchor='nw', tags='hist')


# ==================== 主界面 ====================

class FilmProcessorUI:
    def __init__(self):
        self.image_manager = ImageManager()
        self.param_queue = ParameterQueue()
        self.render_buffer = RenderingBuffer()
        self.render_running = False
        self.current_image_id = None
        self.current_image_data = None
        self.display_photo = None
        self.color_picker = None
        self.preview_scale = 0.125
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        
        self.root = tk.Tk()
        self.root.title("Aurhythm 胶片Cineon校准器 v3.6 - 曝光校准版")
        self.root.geometry("1500x1000")
        self.root.configure(bg=DARK_THEME['bg'])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        apply_dark_theme()
        self.setup_ui()
        self.start_render_thread()
        self.root.mainloop()
    
    def setup_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left = ttk.Frame(main_paned, width=350)
        self.setup_image_panel(left)
        main_paned.add(left)
        
        middle = ttk.Frame(main_paned)
        self.setup_preview_panel(middle)
        main_paned.add(middle)
        
        right = ttk.Frame(main_paned, width=400)
        self.setup_parameter_panel(right)
        main_paned.add(right)
    
    def setup_image_panel(self, parent):
        ttk.Label(parent, text="图像管理", font=('Microsoft YaHei', 14, 'bold')).pack(anchor=tk.W, pady=(0,10))
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Button(btn_frame, text="添加RAW图像", command=self.add_raw_images, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="批量导出", command=self.batch_export, width=15).pack(side=tk.LEFT, padx=2)
        
        list_frame = ttk.LabelFrame(parent, text="图像列表", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('name', 'size', 'status')
        self.image_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', height=20, selectmode='extended')
        self.image_tree.heading('#0', text='', anchor=tk.W)
        self.image_tree.column('#0', width=30)
        self.image_tree.heading('name', text="文件名")
        self.image_tree.column('name', width=180)
        self.image_tree.heading('size', text="尺寸")
        self.image_tree.column('size', width=80)
        self.image_tree.heading('status', text="状态")
        self.image_tree.column('status', width=60)
        self.image_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_tree.configure(yscrollcommand=scrollbar.set)
        
        self.image_tree.bind('<<TreeviewSelect>>', self.on_image_selected)
        self.root.bind('<Delete>', self.delete_selected_image)
    
    def setup_preview_panel(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(control_frame, text="预览分辨率:").pack(side=tk.LEFT)
        self.resolution_var = tk.StringVar(value="12.5%")
        self.resolution_menu = ttk.OptionMenu(control_frame, self.resolution_var, "12.5%",
                                              "100%", "50%", "25%", "12.5%",
                                              command=self.on_resolution_changed)
        self.resolution_menu.config(width=10)
        self.resolution_menu.pack(side=tk.LEFT, padx=5)
        
        self.image_canvas = tk.Canvas(parent, bg=DARK_THEME['canvas_bg'], height=500, highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.cursor_info = ttk.Label(parent, text="", relief='sunken')
        self.cursor_info.pack(fill=tk.X, pady=(5,0))
        
        self.color_picker = ColorPicker(self.image_canvas,
                                        on_pick_callback=self.on_base_pick,
                                        on_move_callback=self.on_mouse_move)
        self.image_info = ttk.Label(parent, text="未选择图像", relief='sunken')
        self.image_info.pack(fill=tk.X, pady=(10,0))
    
    def setup_parameter_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        calib = ttk.Frame(notebook)
        self.setup_calib_tab(calib)
        notebook.add(calib, text="胶片校准")
        
        output = ttk.Frame(notebook)
        self.setup_output_tab(output)
        notebook.add(output, text="输出设置")
        
        lut_tab = ttk.Frame(notebook)
        self.setup_lut_tab(lut_tab)
        notebook.add(lut_tab, text="LUT套用")
        
        hist_frame = ttk.LabelFrame(parent, text="RGB直方图 (Cineon域)", padding=5)
        hist_frame.pack(fill=tk.X, pady=(10,0), side=tk.BOTTOM)
        self.hist_view = HistogramView(hist_frame, width=380, height=120)
        self.hist_view.pack()
    
    def setup_calib_tab(self, parent):
        canvas = tk.Canvas(parent, bg=DARK_THEME['frame_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        ttk.Label(scrollable, text="负片流程: 片基采样 → 增益平衡 → 自动对齐", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        ttk.Label(scrollable, text="1. 点击'片基采样'，在图像片基区域点击（最亮的未曝光区）\n2. 调整红/绿/蓝增益，使片基Cineon代码接近95\n3. 点击'自动对齐'快速平衡三通道", justify=tk.LEFT).pack(anchor=tk.W, pady=(0,15))
        
        film_frame = ttk.LabelFrame(scrollable, text="胶片类型", padding=10)
        film_frame.pack(fill=tk.X, pady=(0,10))
        self.film_type_var = tk.StringVar(value='通用负片 (默认)')
        self.film_menu = ttk.Combobox(film_frame, textvariable=self.film_type_var,
                                      values=list(FILM_DATABASE.keys()), state='readonly', width=35)
        self.film_menu.pack(fill=tk.X)
        self.film_menu.bind('<<ComboboxSelected>>', self.on_film_change)
        self.film_desc = ttk.Label(film_frame, text="", font=('Arial', 9), foreground='gray')
        self.film_desc.pack(anchor=tk.W, pady=(5,0))
        
        # 高级参数面板
        self.advanced_frame = ttk.LabelFrame(scrollable, text="▼ 高级参数 (手动输入柯达官方值)", padding=10)
        self.advanced_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(self.advanced_frame, text="串扰矩阵 (3x3, M_inv):", font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0,5))
        matrix_frame = ttk.Frame(self.advanced_frame)
        matrix_frame.pack(fill=tk.X, pady=5)
        
        self.matrix_entries = []
        for i in range(3):
            row_frame = ttk.Frame(matrix_frame)
            row_frame.pack(fill=tk.X, pady=1)
            entries = []
            for j in range(3):
                var = tk.StringVar(value="0.000")
                entry = ttk.Entry(row_frame, textvariable=var, width=10, justify='center')
                entry.pack(side=tk.LEFT, padx=2)
                entries.append(var)
            self.matrix_entries.append(entries)
        
        matrix_btn_frame = ttk.Frame(self.advanced_frame)
        matrix_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(matrix_btn_frame, text="应用矩阵", command=self.apply_custom_matrix, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(matrix_btn_frame, text="从当前胶片加载", command=self.update_matrix_display, width=15).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.advanced_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(self.advanced_frame, text="H-D 曲线参数 (Sigmoid):", font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0,5))
        hd_frame = ttk.Frame(self.advanced_frame)
        hd_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(hd_frame, text="D_min:").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        self.hd_min_var = tk.StringVar(value="0.20")
        ttk.Entry(hd_frame, textvariable=self.hd_min_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(hd_frame, text="D_max:").grid(row=0, column=2, padx=5, pady=2, sticky='e')
        self.hd_max_var = tk.StringVar(value="3.20")
        ttk.Entry(hd_frame, textvariable=self.hd_max_var, width=8).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(hd_frame, text="a (斜率):").grid(row=1, column=0, padx=5, pady=2, sticky='e')
        self.hd_slope_var = tk.StringVar(value="4.50")
        ttk.Entry(hd_frame, textvariable=self.hd_slope_var, width=8).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(hd_frame, text="b (中点):").grid(row=1, column=2, padx=5, pady=2, sticky='e')
        self.hd_mid_var = tk.StringVar(value="-0.80")
        ttk.Entry(hd_frame, textvariable=self.hd_mid_var, width=8).grid(row=1, column=3, padx=5, pady=2)
        
        ttk.Label(hd_frame, text="软度:").grid(row=2, column=0, padx=5, pady=2, sticky='e')
        self.hd_softness_var = tk.StringVar(value="0.005")
        ttk.Entry(hd_frame, textvariable=self.hd_softness_var, width=8).grid(row=2, column=1, padx=5, pady=2)
        
        hd_btn_frame = ttk.Frame(self.advanced_frame)
        hd_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(hd_btn_frame, text="应用 H-D 参数", command=self.apply_custom_hd, width=15).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.advanced_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Button(self.advanced_frame, text="保存为预设", command=self.save_custom_preset, width=15).pack(pady=5)
        
        base_frame = ttk.LabelFrame(scrollable, text="片基采样", padding=10)
        base_frame.pack(fill=tk.X, pady=(0,10))
        self.base_val_label = ttk.Label(base_frame, text="未采样", font=('Courier', 10))
        self.base_val_label.pack(anchor=tk.W, pady=5)
        btn_frame = ttk.Frame(base_frame)
        btn_frame.pack(fill=tk.X)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        ttk.Button(btn_frame, text="采样", command=self.activate_base_sampler).grid(row=0, column=0, padx=2, sticky='ew')
        ttk.Button(btn_frame, text="对齐", command=self.auto_align).grid(row=0, column=1, padx=2, sticky='ew')
        
        gain_frame = ttk.LabelFrame(scrollable, text="通道增益 (曝光补偿)", padding=10)
        gain_frame.pack(fill=tk.X)
        self.r_gain = PrecisionSlider(gain_frame, "红增益:", 0.5, 2.0, 0.001, 'r_gain', self.on_param_change)
        self.g_gain = PrecisionSlider(gain_frame, "绿增益:", 0.5, 2.0, 0.001, 'g_gain', self.on_param_change)
        self.b_gain = PrecisionSlider(gain_frame, "蓝增益:", 0.5, 2.0, 0.001, 'b_gain', self.on_param_change)
        ttk.Button(gain_frame, text="重置增益", command=self.reset_gains, width=15).pack(pady=10)
    
    def setup_output_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(frame, text="输出色彩空间", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        self.colorspace_var = tk.StringVar(value='cineon')
        ttk.Radiobutton(frame, text="Cineon (标准对数空间)", variable=self.colorspace_var, value='cineon',
                       command=self.on_colorspace_change).pack(anchor=tk.W)
        ttk.Radiobutton(frame, text="ARRI LogC3 (数学转换)", variable=self.colorspace_var, value='logc3',
                       command=self.on_colorspace_change).pack(anchor=tk.W)
        
        ttk.Label(frame, text="\n转换公式:", font=('Arial', 9), foreground='gray').pack(anchor=tk.W)
        ttk.Label(frame, text="Cineon → LogC3:\nE = 10^((Code-95)/500)\nLogC3 = 0.0925×ln(E+0.005)+0.391",
                 font=('Courier', 8), foreground='gray').pack(anchor=tk.W)
        
        ttk.Button(frame, text="导出当前图像", command=self.export_current, width=20).pack(pady=20)
    
    def setup_lut_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(frame, text="3D LUT套用", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(frame, text="支持 .cube 格式，可将 Cineon/LogC3 映射到 Rec.709 等色彩空间",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W, pady=(0,10))
        
        self.lut_path_label = ttk.Label(frame, text="未加载 LUT", font=('Courier', 9))
        self.lut_path_label.pack(anchor=tk.W, pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="加载 .cube LUT", command=self.load_lut, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="卸载 LUT", command=self.unload_lut, width=15).pack(side=tk.LEFT, padx=2)
        
        self.lut_enable_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="启用 LUT 预览", variable=self.lut_enable_var,
                       command=self.on_lut_toggle).pack(anchor=tk.W, pady=10)
    
    # ========== 事件处理 ==========
    
    def add_raw_images(self):
        files = filedialog.askopenfilenames(title="选择RAW文件", 
                    filetypes=[("RAW图像", "*.nef *.dng *.cr2 *.arw *.raf *.orf *.rw2"), ("所有文件", "*.*")])
        for f in files:
            img_id = self.image_manager.add_image(f)
            self.image_tree.insert('', 'end', iid=img_id, values=(os.path.basename(f), "加载中", "待处理"))
        if files and self.current_image_id is None:
            self.image_tree.selection_set(img_id)
            self.on_image_selected()
    
    def delete_selected_image(self, event=None):
        selected = self.image_tree.selection()
        if not selected:
            return
        if messagebox.askyesno("确认", f"删除 {len(selected)} 张图像？"):
            for img_id in selected:
                if int(img_id) == self.current_image_id:
                    self.current_image_id = None
                    self.clear_preview()
                del self.image_manager.images[int(img_id)]
                self.image_tree.delete(img_id)
    
    def on_image_selected(self, event=None):
        selected = self.image_tree.selection()
        if not selected:
            return
        img_id = int(selected[0])
        if img_id == self.current_image_id:
            return
        self.current_image_id = img_id
        self.reset_gains()
        self.base_val_label.config(text="未采样")
        pipeline = self.image_manager.images[img_id]['pipeline']
        self.film_type_var.set(pipeline.film_type)
        self.film_desc.config(text=pipeline.get_film_description(pipeline.film_type))
        self.update_matrix_display()
        self.load_image_preview(img_id)
    
    def load_image_preview(self, img_id):
        def load():
            data = self.image_manager.get_image_data(img_id, self.preview_scale)
            if data is not None:
                self.root.after(0, lambda: self.on_image_loaded(img_id, data))
        threading.Thread(target=load, daemon=True).start()
    
    def on_image_loaded(self, img_id, data):
        if img_id != self.current_image_id:
            return
        self.current_image_data = data
        self.image_manager.images[img_id]['pipeline'].load_linear_image(data)
        self.color_picker.update_image_info(data, self.display_scale, self.display_offset)
        self.image_info.config(text=f"{self.image_manager.images[img_id]['name']} - {data.shape[1]}x{data.shape[0]}")
        self.show_raw_preview(data)
    
    def show_raw_preview(self, data):
        img = (data * 255).astype(np.uint8)
        self.display_image(Image.fromarray(img, mode='RGB'))
    
    def activate_base_sampler(self):
        self.color_picker.clear_crosshair()
        messagebox.showinfo("片基采样", "点击图像中最亮的片基区域（未曝光边缘或片孔）\n理想片基值应在 0.55-0.8 之间")
    
    def on_base_pick(self, rgb, coords):
        if self.current_image_id is None:
            return
        
        max_val = max(rgb)
        
        # 分类弹窗
        if max_val < 0.4:
            result = messagebox.askyesno("曝光严重不足", 
                f"片基采样值 {max_val:.3f} (正常范围 0.55-0.8)\n\n"
                f"后果：\n"
                f"• 暗部细节丢失\n"
                f"• 阴影偏蓝/紫\n"
                f"• 拉亮后噪点爆炸\n\n"
                f"建议：\n"
                f"• 提高光源亮度\n"
                f"• 降低快门速度（如 1秒 → 2秒）\n"
                f"• 开大光圈\n\n"
                f"是否仍然使用这个采样点？")
            if not result:
                return
            status = "⚠️严重欠曝"
        elif max_val < 0.55:
            result = messagebox.askyesno("曝光略欠", 
                f"片基采样值 {max_val:.3f} (理想范围 0.55-0.8)\n\n"
                f"可能的问题：\n"
                f"• 暗部轻微偏色\n"
                f"• 阴影噪点略多\n\n"
                f"建议：稍微提高曝光（+0.3-0.7档）\n\n"
                f"是否继续？")
            if not result:
                return
            status = "⚠️略欠"
        elif max_val > 0.85:
            result = messagebox.askyesno("可能过曝", 
                f"片基采样值 {max_val:.3f} (理想范围 0.55-0.8)\n\n"
                f"检查：片基区域是否完全发白（高光溢出）？\n\n"
                f"如果过曝，高光细节会永久丢失\n\n"
                f"是否继续？")
            if not result:
                return
            status = "⚠️可能过曝"
        else:
            messagebox.showinfo("采样成功", 
                f"片基采样值 {max_val:.3f} 在理想范围内 (0.55-0.8)\n\n"
                f"可以继续调色。")
            status = "✅正常"
        
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        pipeline.set_base_val(rgb, coords)
        self.base_val_label.config(text=f"R={rgb[0]:.4f} G={rgb[1]:.4f} B={rgb[2]:.4f} | {status} (max={max_val:.3f})")
        self.param_queue.put(self.get_current_params())
    
    def on_mouse_move(self, rgb, coords):
        if self.current_image_id is None:
            return
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        if pipeline.base_val_rgb is not None:
            gains = pipeline.channel_gains
            linear = [rgb[i] * gains[i] for i in range(3)]
            T = [linear[i] / (pipeline.base_val_rgb[i] + 1e-6) for i in range(3)]
            D = [-np.log10(max(t, 1e-6)) for t in T]
            cineon = [95 + 500 * D[i] for i in range(3)]
            self.cursor_info.config(text=f"密度: R={D[0]:.2f} G={D[1]:.2f} B={D[2]:.2f} | Cineon: {int(cineon[0])}/{int(cineon[1])}/{int(cineon[2])}")
        else:
            self.cursor_info.config(text=f"线性: R={rgb[0]:.3f} G={rgb[1]:.3f} B={rgb[2]:.3f}")
    
    def on_param_change(self, name, value):
        self.param_queue.put(self.get_current_params())
    
    def get_current_params(self):
        return {'r_gain': self.r_gain.get_value(), 'g_gain': self.g_gain.get_value(), 'b_gain': self.b_gain.get_value()}
    
    def reset_gains(self):
        self.r_gain.set_value(1.0)
        self.g_gain.set_value(1.0)
        self.b_gain.set_value(1.0)
        self.param_queue.put(self.get_current_params())
    
    def auto_align(self):
        if self.current_image_id is None:
            return
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        if pipeline.base_val_rgb is None or pipeline.sample_coords is None:
            messagebox.showwarning("警告", "请先进行片基采样")
            return
        
        x, y = pipeline.sample_coords
        linear_raw = pipeline.linear_img[y, x, :]
        
        density_raw = -np.log10(np.maximum(linear_raw, 1e-6))
        target_density = np.mean(density_raw)
        density_delta = target_density - density_raw
        linear_gains = np.power(10.0, -density_delta)
        linear_gains = np.clip(linear_gains, 0.5, 2.0)
        
        self.r_gain.set_value(linear_gains[0])
        self.g_gain.set_value(linear_gains[1])
        self.b_gain.set_value(linear_gains[2])
        pipeline.set_channel_gains(linear_gains.tolist())
        
        new_base = (pipeline.linear_img * np.array(linear_gains).reshape(1,1,3))[y, x, :]
        pipeline.set_base_val(new_base, (x, y))
        self.base_val_label.config(text=f"R={new_base[0]:.4f} G={new_base[1]:.4f} B={new_base[2]:.4f}")
        
        self.param_queue.put(self.get_current_params())
        messagebox.showinfo("完成", "密度域自动对齐完成\n阴影偏色应有所改善")
    
    def on_resolution_changed(self, val):
        scale_map = {'100%': 1.0, '50%': 0.5, '25%': 0.25, '12.5%': 0.125}
        new_scale = scale_map.get(val, 0.125)
        
        if new_scale == self.preview_scale:
            return
        
        self.preview_scale = new_scale
        
        if self.current_image_id is not None:
            self.load_image_preview(self.current_image_id)
            pipeline = self.image_manager.images[self.current_image_id]['pipeline']
            if pipeline.base_val_rgb is not None:
                self.param_queue.put(self.get_current_params())
    
    def on_film_change(self, event=None):
        if self.current_image_id is None:
            return
        film = self.film_type_var.get()
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        pipeline.set_film_type(film)
        self.film_desc.config(text=pipeline.get_film_description(film))
        self.update_matrix_display()
        self.param_queue.put(self.get_current_params())
    
    def on_colorspace_change(self):
        if self.current_image_id is None:
            return
        cs = self.colorspace_var.get()
        self.image_manager.images[self.current_image_id]['pipeline'].set_output_colorspace(cs)
        self.param_queue.put(self.get_current_params())
    
    def update_matrix_display(self):
        if self.current_image_id is None:
            return
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        if pipeline.matrix_inv is not None:
            for i in range(3):
                for j in range(3):
                    self.matrix_entries[i][j].set(f"{pipeline.matrix_inv[i, j]:.3f}")
        
        self.hd_min_var.set(f"{pipeline.hd_min:.3f}")
        self.hd_max_var.set(f"{pipeline.hd_max:.3f}")
        self.hd_slope_var.set(f"{pipeline.hd_slope:.3f}")
        self.hd_mid_var.set(f"{pipeline.hd_mid:.3f}")
        self.hd_softness_var.set(f"{pipeline.hd_clip_softness:.4f}")
    
    def apply_custom_matrix(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "请先选择图像")
            return
        try:
            matrix = []
            for i in range(3):
                row = []
                for j in range(3):
                    val = float(self.matrix_entries[i][j].get())
                    row.append(val)
                matrix.append(row)
            pipeline = self.image_manager.images[self.current_image_id]['pipeline']
            pipeline.set_custom_matrix(matrix)
            self.param_queue.put(self.get_current_params())
            messagebox.showinfo("成功", "串扰矩阵已应用")
        except Exception as e:
            messagebox.showerror("错误", f"矩阵格式错误: {e}")
    
    def apply_custom_hd(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "请先选择图像")
            return
        try:
            d_min = float(self.hd_min_var.get())
            d_max = float(self.hd_max_var.get())
            slope = float(self.hd_slope_var.get())
            mid = float(self.hd_mid_var.get())
            softness = float(self.hd_softness_var.get())
            pipeline = self.image_manager.images[self.current_image_id]['pipeline']
            pipeline.set_custom_hd_params(d_min, d_max, slope, mid, softness)
            self.param_queue.put(self.get_current_params())
            messagebox.showinfo("成功", "H-D 曲线参数已应用")
        except Exception as e:
            messagebox.showerror("错误", f"参数格式错误: {e}")
    
    def save_custom_preset(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "请先选择图像")
            return
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        
        name = simpledialog.askstring("保存预设", "请输入胶片名称:")
        if not name:
            return
        
        FILM_DATABASE[name] = {
            'description': f"用户自定义: {name}",
            'matrix_inv': pipeline.matrix_inv.copy(),
            'hd_min': pipeline.hd_min,
            'hd_max': pipeline.hd_max,
            'hd_slope': pipeline.hd_slope,
            'hd_mid': pipeline.hd_mid,
            'hd_clip_softness': pipeline.hd_clip_softness,
        }
        
        self.film_menu['values'] = list(FILM_DATABASE.keys())
        self.film_type_var.set(name)
        self.film_desc.config(text=FILM_DATABASE[name]['description'])
        messagebox.showinfo("成功", f"已保存预设: {name}")
    
    def load_lut(self):
        path = filedialog.askopenfilename(filetypes=[("Cube LUT", "*.cube"), ("所有文件", "*.*")])
        if not path or self.current_image_id is None:
            return
        if self.image_manager.images[self.current_image_id]['pipeline'].load_lut(path):
            self.lut_path_label.config(text=os.path.basename(path))
            self.lut_enable_var.set(True)
            self.image_manager.images[self.current_image_id]['pipeline'].set_lut_enabled(True)
            self.param_queue.put(self.get_current_params())
            messagebox.showinfo("成功", f"已加载: {os.path.basename(path)}")
    
    def unload_lut(self):
        if self.current_image_id is None:
            return
        self.image_manager.images[self.current_image_id]['pipeline'].set_lut_enabled(False)
        self.image_manager.images[self.current_image_id]['pipeline'].lut = None
        self.lut_path_label.config(text="未加载 LUT")
        self.lut_enable_var.set(False)
        self.param_queue.put(self.get_current_params())
    
    def on_lut_toggle(self):
        if self.current_image_id is None:
            return
        enabled = self.lut_enable_var.get()
        self.image_manager.images[self.current_image_id]['pipeline'].set_lut_enabled(enabled)
        self.param_queue.put(self.get_current_params())
    
    def export_current(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "没有选择图像")
            return
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        if pipeline.base_val_rgb is None:
            messagebox.showwarning("警告", "请先进行片基采样")
            return
        cs = pipeline.output_colorspace
        name = os.path.splitext(self.image_manager.images[self.current_image_id]['name'])[0]
        default = f"{name}_{cs}.tif"
        path = filedialog.asksaveasfilename(defaultextension=".tif", initialfile=default)
        if not path:
            return
        
        def export():
            try:
                full = self.image_manager.get_image_data(self.current_image_id, scale=1.0)
                if full is None:
                    self.root.after(0, lambda: messagebox.showerror("错误", "加载失败"))
                    return
                exp = ScientificFilmPipeline()
                exp.load_linear_image(full)
                exp.base_val_rgb = pipeline.base_val_rgb.copy()
                exp.channel_gains = pipeline.channel_gains.copy()
                exp.set_film_type(pipeline.film_type)
                exp.set_output_colorspace(cs)
                out = exp.process_for_output()
                if out is None:
                    self.root.after(0, lambda: messagebox.showerror("错误", "处理失败"))
                    return
                try:
                    import tifffile
                    tifffile.imwrite(path, out.astype(np.float32), photometric='rgb')
                except ImportError:
                    try:
                        import imageio
                        imageio.imwrite(path, out.astype(np.float32), format='TIFF')
                    except ImportError:
                        self.root.after(0, lambda: messagebox.showerror("错误", "请安装 tifffile 或 imageio"))
                        return
                self.root.after(0, lambda: messagebox.showinfo("成功", f"已导出: {os.path.basename(path)}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        threading.Thread(target=export, daemon=True).start()
    
    def batch_export(self):
        selected = self.image_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "没有选中图像")
            return
        dir_path = filedialog.askdirectory(title="选择导出目录")
        if not dir_path:
            return
        to_export = []
        for sid in selected:
            pid = int(sid)
            if self.image_manager.images[pid]['pipeline'].base_val_rgb is not None:
                to_export.append(pid)
        if not to_export:
            messagebox.showwarning("警告", "没有已完成采样的图像")
            return
        
        def batch():
            ok = 0
            fail = 0
            for pid in to_export:
                try:
                    img = self.image_manager.images[pid]
                    pipeline = img['pipeline']
                    full = self.image_manager.get_image_data(pid, scale=1.0)
                    if full is None:
                        fail += 1
                        continue
                    exp = ScientificFilmPipeline()
                    exp.load_linear_image(full)
                    exp.base_val_rgb = pipeline.base_val_rgb.copy()
                    exp.channel_gains = pipeline.channel_gains.copy()
                    exp.set_film_type(pipeline.film_type)
                    exp.set_output_colorspace(pipeline.output_colorspace)
                    out = exp.process_for_output()
                    if out is None:
                        fail += 1
                        continue
                    name = os.path.splitext(img['name'])[0]
                    out_path = os.path.join(dir_path, f"{name}_{pipeline.output_colorspace}.tif")
                    try:
                        import tifffile
                        tifffile.imwrite(out_path, out.astype(np.float32), photometric='rgb')
                    except ImportError:
                        import imageio
                        imageio.imwrite(out_path, out.astype(np.float32), format='TIFF')
                    ok += 1
                except Exception as e:
                    print(f"导出失败: {e}")
                    fail += 1
            self.root.after(0, lambda: messagebox.showinfo("批量导出", f"成功: {ok}\n失败: {fail}"))
        threading.Thread(target=batch, daemon=True).start()
    
    def start_render_thread(self):
        self.render_running = True
        def worker():
            while self.render_running:
                params = self.param_queue.get_latest()
                if self.current_image_id is not None and params:
                    pipe = self.image_manager.images[self.current_image_id]['pipeline']
                    if pipe.base_val_rgb is not None:
                        gains = [params.get('r_gain', 1.0), params.get('g_gain', 1.0), params.get('b_gain', 1.0)]
                        pipe.set_channel_gains(gains)
                        preview = pipe.process_for_preview()
                        if preview is not None:
                            self.render_buffer.update_back_buffer(Image.fromarray(preview, mode='RGB'))
                            self.render_buffer.swap_buffers()
                            cineon = pipe.process_to_cineon()
                            if cineon is not None:
                                if cineon.shape[0] > 500:
                                    h = int(500 / cineon.shape[0] * cineon.shape[1])
                                    small = np.array(Image.fromarray((cineon*255).astype(np.uint8)).resize((h,500)))
                                    cineon_small = small.astype(np.float32)/255.0
                                else:
                                    cineon_small = cineon
                                self.root.after(0, lambda c=cineon_small: self.hist_view.update_histogram(c))
                time.sleep(0.03)
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(33, self.update_display)
    
    def update_display(self):
        img = self.render_buffer.get_front_buffer()
        if img is not None:
            self.display_image(img)
        self.root.after(33, self.update_display)
    
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
        photo = ImageTk.PhotoImage(img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=photo)
        self.display_photo = photo
        if self.current_image_data is not None:
            self.color_picker.update_image_info(self.current_image_data, self.display_scale, self.display_offset)
    
    def clear_preview(self):
        self.image_canvas.delete("all")
        self.color_picker.clear_crosshair()
        self.image_info.config(text="未选择图像")
        self.cursor_info.config(text="")
        self.current_image_data = None
        self.base_val_label.config(text="未采样")
        self.hist_view.canvas.delete('hist')
    
    def on_closing(self):
        self.render_running = False
        self.root.destroy()


if __name__ == '__main__':
    print("=" * 60)
    print("Aurhythm 胶片Cineon校准器 v3.6 - 曝光校准版")
    print("流程: RAW → 密度 → 解串扰 → 软裁剪 Sigmoid H-D → Cineon → LogC3")
    print("\nv3.6 新特性:")
    print("  - 采样报警分类弹窗（严重欠曝/略欠/过曝/正常）")
    print("  - 删除阴影提亮滑块（根源是曝光）")
    print("  - 修复预览分辨率响应")
    print("  - 基于 KODAK VISION3 官方文档校准的 5207/5219 参数")
    print("=" * 60)
    
    try:
        import rawpy
        print("✓ rawpy已安装")
    except ImportError:
        print("✗ rawpy未安装，请执行: pip install rawpy")
    
    try:
        import tifffile
        print("✓ tifffile已安装")
    except ImportError:
        print("⚠ 建议安装tifffile: pip install tifffile (可选，导出功能需要)")
    
    app = FilmProcessorUI()

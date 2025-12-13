import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import time
import numpy as np
import warnings
from PIL import Image, ImageTk
import math
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import colour
    COLOUR_AVAILABLE = True
except ImportError:
    COLOUR_AVAILABLE = False
    print("警告: colour-science未安装，色彩转换功能受限")

class FilmPipeline:
    CINEON_BLACK_POINT = 95 / 1023
    CINEON_WHITE_POINT = 685 / 1023
    
    def __init__(self):
        self.linear_img_full = None
        self.linear_img = None
        self.cineon_img = None
        self.display_linear_img = None
        self.display_img = None
        
        self.image_loaded = False
        self.current_resolution_scale = 1.0
        
        self.params = {
            'input_gain': 1.0,
            'master_gain': 1.0,
            'master_shift': 0.0,
            'fit_to_cineon': True,
            
            'r_shift': 0.0, 'r_gain': 1.0,
            'g_shift': 0.0, 'g_gain': 1.0,
            'b_shift': 0.0, 'b_gain': 1.0,
            
            'temperature': 0.0,
            'tint': 0.0,
            'saturation': 1.0,
            'contrast': 1.0,
            'exposure': 0.0,
            
            'processing_mode': 'negative',
            'display_gamma': 2.2
        }
        
        self._cache_valid = False
        self._cache_log_img = None
        
    def load_image(self, file_path):
        try:
            import imageio
            img_array = imageio.imread(file_path)
            
            print(f"加载图像: {img_array.shape}, dtype: {img_array.dtype}")
            
            if img_array.dtype == np.uint8:
                img_float = img_array.astype(np.float32) / 255.0
            elif img_array.dtype == np.uint16:
                img_float = img_array.astype(np.float32) / 65535.0
            elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
                img_float = img_array.astype(np.float32)
            else:
                img_float = img_array.astype(np.float32)
                img_min = img_float.min()
                img_max = img_float.max()
                if img_max > img_min:
                    img_float = (img_float - img_min) / (img_max - img_min)
            
            if len(img_float.shape) == 2:
                img_float = np.stack([img_float] * 3, axis=2)
            elif img_float.shape[2] == 1:
                img_float = np.repeat(img_float, 3, axis=2)
            elif img_float.shape[2] == 4:
                img_float = img_float[:, :, :3]
            
            self.linear_img_full = img_float
            print(f"处理后图像: {self.linear_img_full.shape}, range: [{self.linear_img_full.min():.3f}, {self.linear_img_full.max():.3f}]")
            
            self.set_resolution_scale(1.0)
            
            self.image_loaded = True
            self._cache_valid = False
            return True
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_resolution_scale(self, scale):
        if self.linear_img_full is None:
            return False
        
        if scale == self.current_resolution_scale and self.linear_img is not None:
            return True
        
        self.current_resolution_scale = scale
        
        if scale == 1.0:
            self.linear_img = self.linear_img_full.copy()
        else:
            h, w = self.linear_img_full.shape[:2]
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            
            img_8bit = (self.linear_img_full * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_8bit)
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.linear_img = np.array(img_pil).astype(np.float32) / 255.0
        
        print(f"设置分辨率: {scale:.1%}, 新尺寸: {self.linear_img.shape}")
        self._cache_valid = False
        return True
    
    def apply_log_encoding(self):
        if self.linear_img is None:
            return None
        
        mode = self.params['processing_mode']
        
        img_data = self.linear_img.copy()
        if self.params['input_gain'] != 1.0:
            img_data = np.clip(img_data * self.params['input_gain'], 0, 1)
        
        if mode == "log2log":
            log_img = np.clip(img_data, 0, 1)
            print("Log-to-Log模式: 跳过对数编码")
            
        elif mode == "negative":
            positive_linear = 1.0 - img_data
            
            epsilon = 1e-6
            positive_linear = np.maximum(positive_linear, epsilon)
            
            log_img = -np.log10(positive_linear)
            
            log_img = np.clip(log_img / 2.5, 0, 1)
            
            print(f"负片模式: 反转后范围 [{positive_linear.min():.3f}, {positive_linear.max():.3f}], "
                  f"对数后范围 [{log_img.min():.3f}, {log_img.max():.3f}]")
            
        else:
            img_data = np.maximum(img_data, 1e-6)
            log_img = -np.log10(img_data)
            log_img = np.clip(log_img / 3.0, 0, 1)
            
            print(f"反转片模式: 对数后范围 [{log_img.min():.3f}, {log_img.max():.3f}]")
        
        master_gain = self.params['master_gain']
        master_shift = self.params['master_shift']
        
        if master_gain != 1.0:
            log_img = log_img * master_gain
        
        if master_shift != 0.0:
            log_img = log_img + master_shift
        
        if self.params['fit_to_cineon']:
            black = self.CINEON_BLACK_POINT
            white = self.CINEON_WHITE_POINT
            log_img = (log_img * (white - black) + black)
        
        self.cineon_img = np.clip(log_img, 0, 1)
        print(f"Cineon图像范围: [{self.cineon_img.min():.3f}, {self.cineon_img.max():.3f}]")
        return self.cineon_img
    
    def apply_channel_alignment(self, log_img):
        if log_img is None or log_img.ndim != 3:
            return log_img
        
        aligned_img = log_img.copy()
        
        if self.params['r_shift'] != 0:
            aligned_img[:,:,0] = np.clip(aligned_img[:,:,0] + self.params['r_shift'], 0, 1)
        
        if self.params['r_gain'] != 1.0:
            aligned_img[:,:,0] = np.clip(aligned_img[:,:,0] * self.params['r_gain'], 0, 1)
        
        if self.params['g_shift'] != 0:
            aligned_img[:,:,1] = np.clip(aligned_img[:,:,1] + self.params['g_shift'], 0, 1)
        
        if self.params['g_gain'] != 1.0:
            aligned_img[:,:,1] = np.clip(aligned_img[:,:,1] * self.params['g_gain'], 0, 1)
        
        if self.params['b_shift'] != 0:
            aligned_img[:,:,2] = np.clip(aligned_img[:,:,2] + self.params['b_shift'], 0, 1)
        
        if self.params['b_gain'] != 1.0:
            aligned_img[:,:,2] = np.clip(aligned_img[:,:,2] * self.params['b_gain'], 0, 1)
        
        return aligned_img
    
    def log_to_linear(self, log_img):
        if log_img is None:
            return None
        
        if self.params['fit_to_cineon']:
            black = self.CINEON_BLACK_POINT
            white = self.CINEON_WHITE_POINT
            range_inv = 1.0 / (white - black)
            normalized = np.clip((log_img - black) * range_inv, 0, 1)
        else:
            normalized = np.clip(log_img, 0, 1)
        
        mode = self.params['processing_mode']
        
        if mode == "log2log":
            linear_img = normalized
            
        elif mode == "negative":
            density = normalized * 2.5
            transmittance = np.power(10.0, -density)
            linear_img = transmittance
            
        else:
            density = normalized * 3.0
            linear_img = np.power(10.0, -density)
        
        linear_img = np.clip(linear_img, 0, 1)
        print(f"线性化后范围: [{linear_img.min():.3f}, {linear_img.max():.3f}]")
        return linear_img
    
    def apply_color_adjustments(self, linear_img):
        if linear_img is None:
            return None
        
        img = linear_img.copy()
        
        exposure = self.params['exposure']
        if exposure != 0.0:
            exposure_gain = 2 ** exposure
            img = np.clip(img * exposure_gain, 0, 1)
        
        contrast = self.params['contrast']
        if contrast != 1.0:
            img = 0.5 + (img - 0.5) * contrast
            img = np.clip(img, 0, 1)
        
        saturation = self.params['saturation']
        if saturation != 1.0 and img.ndim == 3 and img.shape[2] >= 3:
            luminance = np.mean(img, axis=2, keepdims=True)
            img = luminance + (img - luminance) * saturation
            img = np.clip(img, 0, 1)
        
        temp = self.params['temperature']
        tint = self.params['tint']
        if (temp != 0.0 or tint != 0.0) and img.ndim == 3 and img.shape[2] >= 3:
            r_gain = 1.0 + temp * 0.3 - tint * 0.1
            g_gain = 1.0 - temp * 0.1 + tint * 0.2
            b_gain = 1.0 - temp * 0.3 - tint * 0.1
            
            img[:,:,0] = np.clip(img[:,:,0] * r_gain, 0, 1)
            img[:,:,1] = np.clip(img[:,:,1] * g_gain, 0, 1)
            img[:,:,2] = np.clip(img[:,:,2] * b_gain, 0, 1)
        
        self.display_linear_img = img
        print(f"色彩调整后范围: [{self.display_linear_img.min():.3f}, {self.display_linear_img.max():.3f}]")
        return self.display_linear_img
    
    def linear_to_display(self, linear_img):
        if linear_img is None:
            return None
        
        gamma = 1.0 / self.params['display_gamma']
        display_img = np.power(np.clip(linear_img, 0, 1), gamma)
        
        if display_img.ndim == 2:
            display_img_8bit = (display_img * 255).astype(np.uint8)
        else:
            display_img_8bit = (display_img[:,:,:3] * 255).astype(np.uint8)
        
        print(f"Gamma校正后范围: [{display_img.min():.3f}, {display_img.max():.3f}], "
              f"8位范围: [{display_img_8bit.min()}, {display_img_8bit.max()}]")
        return display_img_8bit
    
    def process_realtime(self, force_update=False):
        if not self.image_loaded or self.linear_img is None:
            return None
        
        if not force_update and self._cache_valid and self.display_img is not None:
            return self.display_img
        
        try:
            print("\n=== 开始处理流程 ===")
            print(f"输入图像范围: [{self.linear_img.min():.3f}, {self.linear_img.max():.3f}]")
            
            log_img = self.apply_log_encoding()
            if log_img is None:
                print("错误: 对数编码失败")
                return None
            
            aligned_img = self.apply_channel_alignment(log_img)
            if aligned_img is None:
                print("错误: 通道对齐失败")
                return None
            
            linear_img = self.log_to_linear(aligned_img)
            if linear_img is None:
                print("错误: 线性化失败")
                return None
            
            adjusted_img = self.apply_color_adjustments(linear_img)
            if adjusted_img is None:
                print("错误: 色彩调整失败")
                return None
            
            self.display_img = self.linear_to_display(adjusted_img)
            if self.display_img is None:
                print("错误: 显示转换失败")
                return None
            
            self._cache_valid = True
            
            print("=== 处理流程完成 ===\n")
            return self.display_img
            
        except Exception as e:
            print(f"处理错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_histogram_data(self):
        if self.display_linear_img is None:
            print("警告: display_linear_img为None")
            return None, None
        
        img = self.display_linear_img
        
        print(f"获取直方图数据: 图像形状 {img.shape}, 范围 [{img.min():.3f}, {img.max():.3f}]")
        
        if img.ndim == 3 and img.shape[2] >= 3:
            r_data = img[:,:,0].flatten()
            g_data = img[:,:,1].flatten()
            b_data = img[:,:,2].flatten()
            
            r_hist, r_bins = np.histogram(r_data, bins=128, range=(0, 1), density=True)
            g_hist, g_bins = np.histogram(g_data, bins=128, range=(0, 1), density=True)
            b_hist, b_bins = np.histogram(b_data, bins=128, range=(0, 1), density=True)
            
            print(f"直方图统计 - R: [{r_data.min():.3f}, {r_data.max():.3f}], "
                  f"G: [{g_data.min():.3f}, {g_data.max():.3f}], "
                  f"B: [{b_data.min():.3f}, {b_data.max():.3f}]")
            print(f"直方图最大值 - R: {r_hist.max():.3f}, G: {g_hist.max():.3f}, B: {b_hist.max():.3f}")
            
            return (r_hist, g_hist, b_hist), (r_bins, g_bins, b_bins)
        
        elif img.ndim == 2:
            gray_data = img.flatten()
            gray_hist, gray_bins = np.histogram(gray_data, bins=128, range=(0, 1), density=True)
            print(f"灰度直方图统计: [{gray_data.min():.3f}, {gray_data.max():.3f}], 最大值: {gray_hist.max():.3f}")
            
            return (gray_hist, gray_hist, gray_hist), (gray_bins, gray_bins, gray_bins)
        
        print("错误: 图像维度不支持")
        return None, None
    
    def update_parameter(self, name, value):
        if name in self.params:
            self.params[name] = value
            self._cache_valid = False
            return True
        return False
    
    def set_processing_mode(self, mode):
        if mode in ['negative', 'reversal', 'log2log']:
            self.params['processing_mode'] = mode
            self._cache_valid = False
            return True
        return False

class FilmProcessorUI:
    # 灵感来源于早期NamiColor概念
    
    def __init__(self):
        self.pipeline = FilmPipeline()
        
        self.current_file = None
        self.display_image = None
        self.display_photo = None
        
        self.update_timer = None
        self.update_delay = 150
        self.param_changed = False
        self.last_waveform_update = 0
        self.waveform_update_interval = 0.3
        
        self.resolution_options = {
            "完整 (100%)": 1.0,
            "高 (75%)": 0.75,
            "中 (50%)": 0.5,
            "低 (25%)": 0.25,
            "最低 (12.5%)": 0.125
        }
        self.current_resolution = "中 (50%)"
        
        self.root = tk.Tk()
        self.root.title("胶片负片处理器 v2.3")
        self.root.geometry("1500x950")
        
        self.setup_styles()
        self.setup_ui()
        self.root.after(100, self.check_realtime_update)
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        self.bg_color = '#f0f0f0'
        self.frame_bg = '#ffffff'
        self.accent_color = '#2c7fb8'
        
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.setup_title_bar(main_frame)
        
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.setup_image_display(content_frame)
        
        if MATPLOTLIB_AVAILABLE:
            self.setup_waveform_display(content_frame)
        else:
            self.setup_no_waveform_warning(content_frame)
        
        self.setup_parameter_notebook(content_frame)
        self.setup_bottom_panel(main_frame)
    
    def setup_title_bar(self, parent):
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="胶片负片处理器", 
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        version_label = ttk.Label(title_frame, text="v2.3", foreground='gray')
        version_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_image_display(self, parent):
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        image_frame = ttk.LabelFrame(left_frame, text="图像预览", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(image_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="选择图像", 
                  command=self.select_image, width=12).pack(side=tk.LEFT, padx=2)
        
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(mode_frame, text="模式:").pack(side=tk.LEFT)
        
        self.mode_var = tk.StringVar(value="negative")
        modes = [("负片", "negative"), ("反转片", "reversal"), ("Log-to-Log", "log2log")]
        
        for text, value in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                           value=value, command=self.on_mode_changed).pack(side=tk.LEFT, padx=2)
        
        res_frame = ttk.Frame(control_frame)
        res_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(res_frame, text="分辨率:").pack(side=tk.LEFT)
        
        self.resolution_var = tk.StringVar(value=self.current_resolution)
        resolution_menu = ttk.OptionMenu(res_frame, self.resolution_var, 
                                        self.current_resolution, *self.resolution_options.keys(),
                                        command=self.on_resolution_changed)
        resolution_menu.config(width=10)
        resolution_menu.pack(side=tk.LEFT, padx=5)
        
        self.realtime_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="实时渲染", 
                       variable=self.realtime_var).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame, text="调试", 
                  command=self.debug_info, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="重置参数", 
                  command=self.reset_parameters, width=12).pack(side=tk.RIGHT, padx=2)
        
        self.image_canvas = tk.Canvas(image_frame, bg='gray20')
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.image_info_label = ttk.Label(image_frame, text="未加载图像", relief='sunken')
        self.image_info_label.pack(fill=tk.X, pady=(10, 0))
    
    def setup_waveform_display(self, parent):
        middle_frame = ttk.Frame(parent, width=400)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(10, 0))
        middle_frame.pack_propagate(False)
        
        waveform_frame = ttk.LabelFrame(middle_frame, text="RGB分量图 (实时更新)", padding=10)
        waveform_frame.pack(fill=tk.BOTH, expand=True)
        
        self.figure = Figure(figsize=(4, 3), dpi=80)
        self.figure.patch.set_facecolor('#f8f8f8')
        
        self.ax_r = self.figure.add_subplot(311)
        self.ax_g = self.figure.add_subplot(312)
        self.ax_b = self.figure.add_subplot(313)
        
        for ax in [self.ax_r, self.ax_g, self.ax_b]:
            ax.set_facecolor('#f8f8f8')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 10)
            ax.grid(True, alpha=0.3)
        
        self.ax_r.set_title("红色通道")
        self.ax_g.set_title("绿色通道")
        self.ax_b.set_title("蓝色通道")
        self.ax_b.set_xlabel("亮度值 (0-1)")
        
        self.line_r, = self.ax_r.plot([0], [0], color='red', linewidth=1.5)
        self.fill_r = self.ax_r.fill_between([0], [0], alpha=0.3, color='red')
        
        self.line_g, = self.ax_g.plot([0], [0], color='green', linewidth=1.5)
        self.fill_g = self.ax_g.fill_between([0], [0], alpha=0.3, color='green')
        
        self.line_b, = self.ax_b.plot([0], [0], color='blue', linewidth=1.5)
        self.fill_b = self.ax_b.fill_between([0], [0], alpha=0.3, color='blue')
        
        self.waveform_canvas = FigureCanvasTkAgg(self.figure, waveform_frame)
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_no_waveform_warning(self, parent):
        middle_frame = ttk.Frame(parent, width=400)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(10, 0))
        middle_frame.pack_propagate(False)
        
        warning_frame = ttk.LabelFrame(middle_frame, text="分量图功能", padding=20)
        warning_frame.pack(fill=tk.BOTH, expand=True)
        
        warning_text = "分量图功能不可用\n\n"
        warning_text += "要启用分量图功能，请安装matplotlib:\n"
        warning_text += "pip install matplotlib"
        
        warning_label = ttk.Label(warning_frame, text=warning_text, justify=tk.LEFT)
        warning_label.pack(fill=tk.BOTH, expand=True)
    
    def setup_parameter_notebook(self, parent):
        right_frame = ttk.Frame(parent, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        core_frame = ttk.Frame(self.notebook)
        self.setup_core_parameters(core_frame)
        self.notebook.add(core_frame, text="核心参数")
        
        channel_frame = ttk.Frame(self.notebook)
        self.setup_channel_alignment(channel_frame)
        self.notebook.add(channel_frame, text="通道对齐")
        
        color_frame = ttk.Frame(self.notebook)
        self.setup_color_adjustments(color_frame)
        self.notebook.add(color_frame, text="色彩调整")
        
        display_frame = ttk.Frame(self.notebook)
        self.setup_display_settings(display_frame)
        self.notebook.add(display_frame, text="显示设置")
    
    def setup_core_parameters(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        input_frame = ttk.LabelFrame(scrollable_frame, text="输入调整", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.input_gain_var = tk.DoubleVar(value=1.0)
        self.create_slider(input_frame, "输入增益:", self.input_gain_var, 
                          0.1, 5.0, 0.1, 'input_gain')
        
        master_frame = ttk.LabelFrame(scrollable_frame, text="对数域调整", padding=10)
        master_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.master_gain_var = tk.DoubleVar(value=1.0)
        self.create_slider(master_frame, "主增益:", self.master_gain_var, 
                          0.5, 2.0, 0.05, 'master_gain')
        
        self.master_shift_var = tk.DoubleVar(value=0.0)
        self.create_slider(master_frame, "主偏移:", self.master_shift_var, 
                          -0.5, 0.5, 0.05, 'master_shift')
        
        cineon_frame = ttk.LabelFrame(scrollable_frame, text="Cineon设置", padding=10)
        cineon_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.fit_cineon_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cineon_frame, text="适应Cineon基准", 
                       variable=self.fit_cineon_var,
                       command=lambda: self.pipeline.update_parameter('fit_to_cineon', 
                                                                     self.fit_cineon_var.get())).pack(anchor=tk.W)
        
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="保存图像", 
                  command=self.save_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(button_frame, text="强制更新分量图", 
                  command=self.force_update_waveform).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    
    def setup_channel_alignment(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        red_frame = ttk.LabelFrame(scrollable_frame, text="红色通道", padding=10)
        red_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.r_shift_var = tk.DoubleVar(value=0.0)
        self.create_slider(red_frame, "偏移:", self.r_shift_var, 
                          -0.5, 0.5, 0.01, 'r_shift')
        
        self.r_gain_var = tk.DoubleVar(value=1.0)
        self.create_slider(red_frame, "增益:", self.r_gain_var, 
                          0.5, 2.0, 0.05, 'r_gain')
        
        green_frame = ttk.LabelFrame(scrollable_frame, text="绿色通道", padding=10)
        green_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.g_shift_var = tk.DoubleVar(value=0.0)
        self.create_slider(green_frame, "偏移:", self.g_shift_var, 
                          -0.5, 0.5, 0.01, 'g_shift')
        
        self.g_gain_var = tk.DoubleVar(value=1.0)
        self.create_slider(green_frame, "增益:", self.g_gain_var, 
                          0.5, 2.0, 0.05, 'g_gain')
        
        blue_frame = ttk.LabelFrame(scrollable_frame, text="蓝色通道", padding=10)
        blue_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.b_shift_var = tk.DoubleVar(value=0.0)
        self.create_slider(blue_frame, "偏移:", self.b_shift_var, 
                          -0.5, 0.5, 0.01, 'b_shift')
        
        self.b_gain_var = tk.DoubleVar(value=1.0)
        self.create_slider(blue_frame, "增益:", self.b_gain_var, 
                          0.5, 2.0, 0.05, 'b_gain')
        
        info_frame = ttk.Frame(scrollable_frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        info_text = "通道对齐说明：\n"
        info_text += "1. 使用偏移对齐黑点\n"
        info_text += "2. 使用增益对齐白点\n"
        info_text += "3. 调整直到三个通道直方图大致对齐"
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(fill=tk.X)
    
    def setup_color_adjustments(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        exp_frame = ttk.LabelFrame(scrollable_frame, text="曝光与对比度", padding=10)
        exp_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.exposure_var = tk.DoubleVar(value=0.0)
        self.create_slider(exp_frame, "曝光:", self.exposure_var, 
                          -3.0, 3.0, 0.1, 'exposure')
        
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.create_slider(exp_frame, "对比度:", self.contrast_var, 
                          0.1, 3.0, 0.05, 'contrast')
        
        sat_frame = ttk.LabelFrame(scrollable_frame, text="饱和度", padding=10)
        sat_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.saturation_var = tk.DoubleVar(value=1.0)
        self.create_slider(sat_frame, "饱和度:", self.saturation_var, 
                          0.0, 3.0, 0.05, 'saturation')
        
        temp_frame = ttk.LabelFrame(scrollable_frame, text="白平衡", padding=10)
        temp_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.temperature_var = tk.DoubleVar(value=0.0)
        self.create_slider(temp_frame, "色温:", self.temperature_var, 
                          -2.0, 2.0, 0.05, 'temperature')
        
        self.tint_var = tk.DoubleVar(value=0.0)
        self.create_slider(temp_frame, "色调:", self.tint_var, 
                          -2.0, 2.0, 0.05, 'tint')
    
    def setup_display_settings(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        gamma_frame = ttk.LabelFrame(scrollable_frame, text="显示Gamma", padding=10)
        gamma_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.gamma_var = tk.DoubleVar(value=2.2)
        
        gamma_frame_inner = ttk.Frame(gamma_frame)
        gamma_frame_inner.pack(fill=tk.X, pady=5)
        
        ttk.Label(gamma_frame_inner, text="Gamma值:").pack(side=tk.LEFT)
        
        gamma_values = [("1.0 (线性)", 1.0), ("1.8 (Mac)", 1.8), 
                       ("2.2 (sRGB/PC)", 2.2), ("2.4 (Rec.709)", 2.4)]
        
        for text, value in gamma_values:
            ttk.Radiobutton(gamma_frame_inner, text=text, variable=self.gamma_var,
                           value=value, command=self.on_gamma_changed).pack(anchor=tk.W)
        
        perf_frame = ttk.LabelFrame(scrollable_frame, text="性能设置", padding=10)
        perf_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.update_delay_var = tk.IntVar(value=150)
        delay_frame = ttk.Frame(perf_frame)
        delay_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(delay_frame, text="更新延迟(ms):").pack(side=tk.LEFT)
        ttk.Label(delay_frame, textvariable=self.update_delay_var, width=6).pack(side=tk.RIGHT)
        
        delay_slider = ttk.Scale(perf_frame, from_=50, to=500, 
                                variable=self.update_delay_var,
                                orient=tk.HORIZONTAL,
                                command=self.on_delay_changed)
        delay_slider.pack(fill=tk.X, pady=(5, 0))
        
        self.waveform_interval_var = tk.DoubleVar(value=0.3)
        interval_frame = ttk.Frame(perf_frame)
        interval_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(interval_frame, text="分量图更新间隔(s):").pack(side=tk.LEFT)
        ttk.Label(interval_frame, textvariable=self.waveform_interval_var, width=6).pack(side=tk.RIGHT)
        
        interval_slider = ttk.Scale(perf_frame, from_=0.1, to=2.0,
                                   variable=self.waveform_interval_var,
                                   orient=tk.HORIZONTAL,
                                   command=self.on_waveform_interval_changed)
        interval_slider.pack(fill=tk.X, pady=(5, 0))
        
        reset_frame = ttk.Frame(scrollable_frame)
        reset_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(reset_frame, text="重置所有参数", 
                  command=self.reset_all_parameters).pack(fill=tk.X)
        
        info_frame = ttk.LabelFrame(scrollable_frame, text="状态信息", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.info_text = tk.StringVar(value="就绪")
        info_label = ttk.Label(info_frame, textvariable=self.info_text, justify=tk.LEFT)
        info_label.pack(fill=tk.X)
    
    def create_slider(self, parent, label, variable, from_, to, resolution, param_name):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text=label).pack(side=tk.LEFT)
        value_label = ttk.Label(top_frame, text=f"{variable.get():.3f}", width=8)
        value_label.pack(side=tk.RIGHT)
        
        slider = ttk.Scale(frame, from_=from_, to=to, variable=variable,
                          orient=tk.HORIZONTAL)
        
        def on_slider_change(val):
            float_val = float(val)
            value_label.config(text=f"{float_val:.3f}")
            self.pipeline.update_parameter(param_name, float_val)
            self.param_changed = True
        
        slider.config(command=lambda v: on_slider_change(v))
        slider.pack(fill=tk.X, pady=(5, 0))
        
        setattr(self, f"{param_name}_label", value_label)
        
        return value_label
    
    def setup_bottom_panel(self, parent):
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        console_frame = ttk.LabelFrame(bottom_frame, text="处理日志", padding=5)
        console_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.console_text = scrolledtext.ScrolledText(console_frame, height=6)
        self.console_text.pack(fill=tk.BOTH, expand=True)
        self.console_text.insert(tk.END, "胶片负片处理器 v2.3\n")
        self.console_text.insert(tk.END, "就绪 - 选择图像开始处理...\n")
        self.console_text.config(state='disabled')
        
        status_frame = ttk.Frame(bottom_frame)
        status_frame.pack(fill=tk.X)
        
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, 
                              relief='sunken', padding=5)
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(status_frame, text="退出", 
                  command=self.root.quit, width=10).pack(side=tk.RIGHT, padx=2)
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.tif;*.tiff;*.jpg;*.jpeg;*.png;*.exr"),
                ("TIFF文件", "*.tif;*.tiff"),
                ("EXR文件", "*.exr"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.current_file = file_path
            self.log_message(f"加载图像: {os.path.basename(file_path)}")
            self.status_var.set(f"加载中: {os.path.basename(file_path)}")
            
            thread = threading.Thread(target=self.load_image_thread, args=(file_path,), daemon=True)
            thread.start()
    
    def load_image_thread(self, file_path):
        try:
            success = self.pipeline.load_image(file_path)
            if success:
                scale = self.resolution_options[self.resolution_var.get()]
                self.pipeline.set_resolution_scale(scale)
                
                self.root.after(0, self.on_image_loaded)
            else:
                self.root.after(0, lambda: messagebox.showerror("错误", "加载图像失败"))
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"加载失败: {e}"))
    
    def on_image_loaded(self):
        self.log_message("图像加载成功，开始实时处理...")
        self.status_var.set(f"已加载: {os.path.basename(self.current_file)}")
        
        self.process_and_update_display(force_waveform=True)
    
    def process_and_update_display(self, force_waveform=False):
        if not self.pipeline.image_loaded:
            return
        
        try:
            display_array = self.pipeline.process_realtime()
            
            if display_array is not None:
                if display_array.ndim == 2:
                    pil_img = Image.fromarray(display_array, mode='L')
                else:
                    pil_img = Image.fromarray(display_array, mode='RGB')
                
                self.update_display_image(pil_img)
                
                current_time = time.time()
                if (force_waveform or 
                    current_time - self.last_waveform_update > self.waveform_update_interval):
                    self.update_waveform()
                    self.last_waveform_update = current_time
                
                self.param_changed = False
                
        except Exception as e:
            self.log_message(f"处理错误: {e}")
    
    def update_display_image(self, pil_img):
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width > 10 and canvas_height > 10:
            img_width, img_height = pil_img.size
            scale = min(canvas_width / img_width, canvas_height / img_height) * 0.9
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            if new_width > 0 and new_height > 0:
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_img)
        
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width//2, canvas_height//2, 
                                     anchor=tk.CENTER, image=photo)
        
        if self.current_file:
            res_text = f"{self.resolution_var.get()}"
            self.image_info_label.config(
                text=f"{os.path.basename(self.current_file)} - {pil_img.width}x{pil_img.height} - {self.mode_var.get()} - {res_text}"
            )
        
        self.display_photo = photo
    
    def update_waveform(self):
        if not MATPLOTLIB_AVAILABLE or not self.pipeline.image_loaded:
            return
        
        try:
            hist_data, bins_data = self.pipeline.get_histogram_data()
            
            if hist_data is None or bins_data is None:
                self.log_message("警告: 无法获取直方图数据")
                return
            
            r_hist, g_hist, b_hist = hist_data
            r_bins, g_bins, b_bins = bins_data
            
            self.ax_r.clear()
            self.ax_r.fill_between(r_bins[:-1], r_hist, alpha=0.3, color='red')
            self.ax_r.plot(r_bins[:-1], r_hist, color='red', linewidth=1.5)
            
            self.ax_g.clear()
            self.ax_g.fill_between(g_bins[:-1], g_hist, alpha=0.3, color='green')
            self.ax_g.plot(g_bins[:-1], g_hist, color='green', linewidth=1.5)
            
            self.ax_b.clear()
            self.ax_b.fill_between(b_bins[:-1], b_hist, alpha=0.3, color='blue')
            self.ax_b.plot(b_bins[:-1], b_hist, color='blue', linewidth=1.5)
            
            max_hist = max(r_hist.max(), g_hist.max(), b_hist.max())
            y_max = max(5, max_hist * 1.2)
            
            for ax in [self.ax_r, self.ax_g, self.ax_b]:
                ax.set_facecolor('#f8f8f8')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, y_max)
                ax.grid(True, alpha=0.3)
            
            self.ax_r.set_title("红色通道")
            self.ax_g.set_title("绿色通道")
            self.ax_b.set_title("蓝色通道")
            self.ax_b.set_xlabel("亮度值 (0-1)")
            
            self.waveform_canvas.draw()
            
            self.log_message(f"分量图更新完成 (最大值: {max_hist:.3f})")
            
        except Exception as e:
            self.log_message(f"更新分量图失败: {e}")
            print(f"更新分量图失败: {e}")
            import traceback
            traceback.print_exc()
    
    def force_update_waveform(self):
        self.last_waveform_update = 0
        self.process_and_update_display(force_waveform=True)
    
    def debug_info(self):
        if not self.pipeline.image_loaded:
            self.log_message("未加载图像")
            return
        
        self.log_message("=== 调试信息 ===")
        self.log_message(f"图像已加载: {self.pipeline.image_loaded}")
        self.log_message(f"图像形状: {self.pipeline.linear_img.shape}")
        self.log_message(f"图像范围: [{self.pipeline.linear_img.min():.3f}, {self.pipeline.linear_img.max():.3f}]")
        self.log_message(f"处理模式: {self.pipeline.params['processing_mode']}")
        self.log_message(f"display_linear_img: {self.pipeline.display_linear_img is not None}")
        if self.pipeline.display_linear_img is not None:
            self.log_message(f"display_linear_img范围: [{self.pipeline.display_linear_img.min():.3f}, {self.pipeline.display_linear_img.max():.3f}]")
        
        self.force_update_waveform()
    
    def check_realtime_update(self):
        if self.realtime_var.get() and self.param_changed and self.pipeline.image_loaded:
            self.process_and_update_display()
        
        self.root.after(self.update_delay, self.check_realtime_update)
    
    def on_mode_changed(self):
        mode = self.mode_var.get()
        self.pipeline.set_processing_mode(mode)
        self.param_changed = True
        
        if self.pipeline.image_loaded:
            self.log_message(f"切换到{mode}模式")
            self.process_and_update_display(force_waveform=True)
    
    def on_resolution_changed(self, value):
        scale = self.resolution_options[value]
        self.log_message(f"切换分辨率到{value} (缩放比例: {scale})")
        
        if self.pipeline.image_loaded:
            self.pipeline.set_resolution_scale(scale)
            self.process_and_update_display(force_waveform=True)
    
    def on_gamma_changed(self):
        gamma = self.gamma_var.get()
        self.pipeline.update_parameter('display_gamma', gamma)
        self.param_changed = True
        
        if self.pipeline.image_loaded:
            self.log_message(f"Gamma值改为{gamma}")
            self.process_and_update_display(force_waveform=True)
    
    def on_delay_changed(self, value):
        self.update_delay = int(float(value))
        self.log_message(f"更新延迟调整为{self.update_delay}ms")
    
    def on_waveform_interval_changed(self, value):
        self.waveform_update_interval = float(value)
        self.log_message(f"分量图更新间隔调整为{self.waveform_update_interval:.1f}秒")
    
    def reset_parameters(self):
        self.input_gain_var.set(1.0)
        self.master_gain_var.set(1.0)
        self.master_shift_var.set(0.0)
        self.fit_cineon_var.set(True)
        
        self.r_shift_var.set(0.0)
        self.r_gain_var.set(1.0)
        
        self.g_shift_var.set(0.0)
        self.g_gain_var.set(1.0)
        
        self.b_shift_var.set(0.0)
        self.b_gain_var.set(1.0)
        
        self.exposure_var.set(0.0)
        self.contrast_var.set(1.0)
        self.saturation_var.set(1.0)
        self.temperature_var.set(0.0)
        self.tint_var.set(0.0)
        
        self.gamma_var.set(2.2)
        
        for param_name in self.pipeline.params:
            var_name = f"{param_name}_var"
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                self.pipeline.update_parameter(param_name, var.get())
        
        self.param_changed = True
        self.log_message("所有参数已重置")
        
        if self.pipeline.image_loaded:
            self.process_and_update_display(force_waveform=True)
    
    def reset_all_parameters(self):
        self.reset_parameters()
        self.update_delay_var.set(150)
        self.update_delay = 150
        self.waveform_interval_var.set(0.3)
        self.waveform_update_interval = 0.3
        self.log_message("所有设置已重置")
    
    def save_image(self):
        if not self.pipeline.image_loaded or self.current_file is None:
            messagebox.showwarning("警告", "没有可保存的图像")
            return
        
        if not messagebox.askyesno("保存图像", "将使用完整分辨率处理并保存图像，可能需要一些时间。是否继续？"):
            return
        
        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
        mode = self.mode_var.get()
        default_name = f"{base_name}_{mode}_processed.png"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg;*.jpeg"),
                ("TIFF文件", "*.tiff;*.tif"),
                ("EXR文件", "*.exr"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                thread = threading.Thread(target=self.save_image_thread, 
                                        args=(file_path,), daemon=True)
                thread.start()
                    
            except Exception as e:
                self.log_message(f"保存失败: {e}")
                messagebox.showerror("错误", f"保存失败: {e}")
    
    def save_image_thread(self, file_path):
        try:
            original_scale = self.pipeline.current_resolution_scale
            self.pipeline.set_resolution_scale(1.0)
            
            self.root.after(0, lambda: self.status_var.set("处理完整分辨率图像中..."))
            display_array = self.pipeline.process_realtime(force_update=True)
            
            if display_array is not None:
                pil_img = Image.fromarray(display_array)
                
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    pil_img.save(file_path, 'JPEG', quality=95)
                elif ext in ['.tif', '.tiff']:
                    pil_img.save(file_path, 'TIFF')
                elif ext == '.exr':
                    self.root.after(0, lambda: self.log_message("EXR保存需要OpenEXR库，已保存为PNG格式"))
                    pil_img.save(file_path, 'PNG')
                else:
                    pil_img.save(file_path, 'PNG')
                
                self.pipeline.set_resolution_scale(original_scale)
                
                self.root.after(0, lambda: self.log_message(f"图像已保存: {os.path.basename(file_path)}"))
                self.root.after(0, lambda: self.status_var.set(f"已保存: {os.path.basename(file_path)}"))
                self.root.after(0, lambda: messagebox.showinfo("成功", "图像保存成功"))
            else:
                self.root.after(0, lambda: messagebox.showerror("错误", "无法获取处理后的图像"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"保存失败: {e}"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"保存失败: {e}"))
    
    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.console_text.config(state='normal')
        self.console_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.console_text.see(tk.END)
        self.console_text.config(state='disabled')
        self.info_text.set(message)
    
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    print("胶片负片处理器 v2.3")
    print("=" * 60)
    print("功能特性:")
    print("1. 修复分量图显示问题")
    print("2. 实时分量图更新")
    print("3. 可选实时渲染分辨率 (12.5% - 100%)")
    print("4. 正确的负片处理流程: 反转 -> 对数编码 -> 通道对齐 -> 线性化")
    print("5. 三种处理模式: 负片/反转片/Log-to-Log")
    print("6. 分页参数面板: 核心参数/通道对齐/色彩调整/显示设置")
    print("=" * 60)
    
    print("\n检查依赖库:")
    if not COLOUR_AVAILABLE:
        print("警告: colour-science库未安装，部分色彩转换功能可能受限")
        print("运行: pip install colour-science")
    else:
        print("colour-science 已安装")
    
    if not MATPLOTLIB_AVAILABLE:
        print("警告: matplotlib未安装，分量图功能将不可用")
        print("运行: pip install matplotlib")
    else:
        print("matplotlib 已安装")
    
    print("\n推荐安装完整依赖:")
    print("pip install numpy pillow imageio matplotlib")
    print("=" * 60)
    print("正在启动主界面...")
    
    app = FilmProcessorUI()
    app.run()

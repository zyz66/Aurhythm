import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import time
import json
import numpy as np
import warnings
from PIL import Image, ImageTk
import math
warnings.filterwarnings('ignore')

# 导入语言配置文件
try:
    import configparser
    CONFIG_AVAILABLE = True
except:
    CONFIG_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('TkAgg')
    matplotlib.rcParams.update({
        'axes.facecolor': '#1e1e1e',
        'figure.facecolor': '#2d2d30',
        'axes.edgecolor': '#404040',
        'axes.labelcolor': '#d4d4d4',
        'xtick.color': '#d4d4d4',
        'ytick.color': '#d4d4d4',
        'grid.color': '#404040',
        'text.color': '#d4d4d4',
        'axes.titlecolor': '#d4d4d4',
        'font.family': 'sans-serif'
    })
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from scipy.interpolate import interp1d
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class LanguageManager:
    """语言管理器"""
    
    def __init__(self):
        self.language = 'zh'  # 默认中文
        self.translations = {
            'zh': {
                # 主界面
                'title': "胶片负片处理器 v3.0",
                'image_management': "图像管理",
                'image_list': "图像列表",
                'add_images': "添加图像",
                'batch_process': "批量处理",
                'filename': "文件名",
                'size': "尺寸",
                'status': "状态",
                'loading': "加载中",
                'pending': "待处理",
                'loaded': "已加载",
                'done': "已完成",
                'failed': "失败",
                
                # 图像预览
                'image_preview': "图像预览",
                'mode': "模式",
                'negative': "负片",
                'reversal': "反转片",
                'log': "Log",
                'preview_resolution': "预览分辨率",
                'full': "完整",
                'high': "高",
                'medium': "中",
                'low': "低",
                'minimum': "最低",
                'real_time': "实时渲染",
                'save_current': "保存当前",
                'save_all': "保存全部",
                'no_image_selected': "未选择图像",
                
                # 参数面板
                'core_parameters': "核心参数",
                'channel_alignment': "通道对齐",
                'color_adjustments': "色彩调整",
                'preset_management': "预设管理",
                'visualization': "图像分析",
                
                # 核心参数
                'input_adjustment': "输入调整",
                'input_gain': "输入增益",
                'log_domain_adjustment': "对数域调整",
                'master_gain': "主增益",
                'master_shift': "主偏移",
                'cineon_settings': "Cineon设置",
                'fit_to_cineon': "适应Cineon基准",
                'display_settings': "显示设置",
                'linear': "线性",
                'mac': "Mac",
                'sRGB': "sRGB",
                'rec709': "Rec.709",
                
                # 通道对齐
                'red_channel': "红色通道",
                'green_channel': "绿色通道",
                'blue_channel': "蓝色通道",
                'shift': "偏移",
                'gain': "增益",
                
                # 色彩调整
                'exposure_contrast': "曝光与对比度",
                'exposure': "曝光",
                'contrast': "对比度",
                'saturation': "饱和度",
                'white_balance': "白平衡",
                'temperature': "色温",
                'tint': "色调",
                
                # 胶片曲线参数
                'film_curve': "胶片曲线参数",
                'toe_strength': "趾部强度",
                'toe_slope': "趾部斜率",
                'shoulder_strength': "肩部强度",
                'shoulder_slope': "肩部斜率",
                'mid_contrast': "中间调对比度",
                'gamma': "中间调Gamma",
                
                # 预设管理
                'preset_explanation': "预设功能允许您保存当前的所有参数设置，\n以便在不同图像之间快速应用相同的调整。",
                'preset_name': "预设名称",
                'export_preset': "导出预设",
                'import_preset': "导入预设",
                'reset_parameters': "重置参数",
                'preset_list': "预设列表",
                
                # 状态栏
                'ready': "就绪",
                'processing': "处理中",
                'saving': "保存中",
                'memory': "内存",
                'images': "图像",
                'cache': "缓存",
                
                # 菜单
                'file': "文件",
                'language': "语言",
                'chinese': "中文",
                'english': "英文",
                'exit': "退出",
                'help': "帮助",
                'about': "关于",
                'restart_required': "重启后生效",
                
                # 消息
                'no_images_warning': "没有可处理的图像",
                'select_save_dir': "选择保存目录",
                'processing_complete': "处理完成",
                'save_success': "保存成功",
                'save_failed': "保存失败",
                'load_failed': "加载失败",
                'export_success': "导出成功",
                'import_success': "导入成功",
                
                # 关于对话框
                'about_title': "关于 Aurhythm",
                'about_text': "胶片负片处理器 v3.0\n\n"
                            "功能特性:\n"
                            "1. 专业的胶片处理流程\n"
                            "2. 实时密度-曝光曲线\n"
                            "3. 多图像批处理\n"
                            "4. 预设保存与导入\n"
                            "5. 中英文界面切换"
            },
            'en': {
                # Main interface
                'title': "Film Negative Processor v3.0",
                'image_management': "Image Management",
                'image_list': "Image List",
                'add_images': "Add Images",
                'batch_process': "Batch Process",
                'filename': "Filename",
                'size': "Size",
                'status': "Status",
                'loading': "Loading",
                'pending': "Pending",
                'loaded': "Loaded",
                'done': "Done",
                'failed': "Failed",
                
                # Image preview
                'image_preview': "Image Preview",
                'mode': "Mode",
                'negative': "Negative",
                'reversal': "Reversal",
                'log': "Log",
                'preview_resolution': "Preview Resolution",
                'full': "Full",
                'high': "High",
                'medium': "Medium",
                'low': "Low",
                'minimum': "Minimum",
                'real_time': "Real-time",
                'save_current': "Save Current",
                'save_all': "Save All",
                'no_image_selected': "No image selected",
                
                # Parameter panel
                'core_parameters': "Core Parameters",
                'channel_alignment': "Channel Alignment",
                'color_adjustments': "Color Adjustments",
                'preset_management': "Preset Management",
                'visualization': "Visualization",
                
                # Core parameters
                'input_adjustment': "Input Adjustment",
                'input_gain': "Input Gain",
                'log_domain_adjustment': "Log Domain Adjustment",
                'master_gain': "Master Gain",
                'master_shift': "Master Shift",
                'cineon_settings': "Cineon Settings",
                'fit_to_cineon': "Fit to Cineon",
                'display_settings': "Display Settings",
                'linear': "Linear",
                'mac': "Mac",
                'sRGB': "sRGB",
                'rec709': "Rec.709",
                
                # Channel alignment
                'red_channel': "Red Channel",
                'green_channel': "Green Channel",
                'blue_channel': "Blue Channel",
                'shift': "Shift",
                'gain': "Gain",
                
                # Color adjustments
                'exposure_contrast': "Exposure & Contrast",
                'exposure': "Exposure",
                'contrast': "Contrast",
                'saturation': "Saturation",
                'white_balance': "White Balance",
                'temperature': "Temperature",
                'tint': "Tint",
                
                # Film curve parameters
                'film_curve': "Film Curve Parameters",
                'toe_strength': "Toe Strength",
                'toe_slope': "Toe Slope",
                'shoulder_strength': "Shoulder Strength",
                'shoulder_slope': "Shoulder Slope",
                'mid_contrast': "Mid-tone Contrast",
                'gamma': "Mid-tone Gamma",
                
                # Preset management
                'preset_explanation': "Presets allow you to save all current parameter settings\nfor quick application to different images.",
                'preset_name': "Preset Name",
                'export_preset': "Export Preset",
                'import_preset': "Import Preset",
                'reset_parameters': "Reset Parameters",
                'preset_list': "Preset List",
                
                # Status bar
                'ready': "Ready",
                'processing': "Processing",
                'saving': "Saving",
                'memory': "Memory",
                'images': "Images",
                'cache': "Cache",
                
                # Menu
                'file': "File",
                'language': "Language",
                'chinese': "Chinese",
                'english': "English",
                'exit': "Exit",
                'help': "Help",
                'about': "About",
                'restart_required': "Requires restart",
                
                # Messages
                'no_images_warning': "No images to process",
                'select_save_dir': "Select Save Directory",
                'processing_complete': "Processing complete",
                'save_success': "Save successful",
                'save_failed': "Save failed",
                'load_failed': "Load failed",
                'export_success': "Export successful",
                'import_success': "Import successful",
                
                # About dialog
                'about_title': "About Aurhythm",
                'about_text': "Film Negative Processor v3.0\n\n"
                            "Features:\n"
                            "1. Professional film processing pipeline\n"
                            "2. Real-time density-exposure curve\n"
                            "3. Multi-image batch processing\n"
                            "4. Preset save and import\n"
                            "5. Chinese/English interface"
            }
        }
        
        # 加载语言配置
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists('aurhythm_config.ini'):
            try:
                config = configparser.ConfigParser()
                config.read('aurhythm_config.ini')
                if 'Settings' in config and 'Language' in config['Settings']:
                    lang = config['Settings']['Language']
                    if lang in self.translations:
                        self.language = lang
            except:
                pass
    
    def save_config(self):
        """保存配置到文件"""
        if CONFIG_AVAILABLE:
            try:
                config = configparser.ConfigParser()
                config['Settings'] = {'Language': self.language}
                with open('aurhythm_config.ini', 'w') as f:
                    config.write(f)
            except:
                pass
    
    def set_language(self, lang):
        """设置语言"""
        if lang in self.translations:
            self.language = lang
            self.save_config()
            return True
        return False
    
    def get(self, key, default=None):
        """获取翻译"""
        if self.language in self.translations:
            lang_dict = self.translations[self.language]
            if key in lang_dict:
                return lang_dict[key]
        
        # 如果当前语言中没有，尝试英文
        if self.language != 'en' and 'en' in self.translations:
            if key in self.translations['en']:
                return self.translations['en'][key]
        
        return default if default is not None else key

class ImageManager:
    """管理多个图像，避免全部加载到内存"""
    
    def __init__(self):
        self.images = {}
        self.current_id = None
        self.processed_cache = {}
        self._next_id = 0
    
    def clear_cache(self):
        """清理缓存"""
        self.processed_cache.clear()
        for img_id in self.images:
            if 'thumbnail' in self.images[img_id]:
                self.images[img_id]['thumbnail'] = None
    
    def add_image(self, file_path):
        img_id = self._next_id
        self._next_id += 1
        
        self.images[img_id] = {
            'path': file_path,
            'name': os.path.basename(file_path),
            'params': {},
            'thumbnail': None,
            'metadata': {}
        }
        
        thread = threading.Thread(target=self._load_metadata, args=(img_id,), daemon=True)
        thread.start()
        
        return img_id
    
    def _load_metadata(self, img_id):
        try:
            with Image.open(self.images[img_id]['path']) as img:
                self.images[img_id]['metadata'] = {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode
                }
                
                thumb_size = (100, 100)
                img.thumbnail(thumb_size)
                self.images[img_id]['thumbnail'] = img.copy()
        except Exception as e:
            print(f"加载图像元数据失败 {img_id}: {e}")
    
    def get_image_data(self, img_id, scale=0.125):
        if img_id not in self.images:
            return None
        
        img_info = self.images[img_id]
        
        cache_key = f"{img_id}_{scale}"
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]
        
        try:
            import imageio
            img_array = imageio.imread(img_info['path'])
            
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
            
            if scale < 1.0:
                h, w = img_float.shape[:2]
                new_h = max(1, int(h * scale))
                new_w = max(1, int(w * scale))
                
                img_8bit = (img_float * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_8bit)
                img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                img_float = np.array(img_pil).astype(np.float32) / 255.0
            
            self.processed_cache[cache_key] = img_float
            
            if len(self.processed_cache) > 5:
                oldest_key = next(iter(self.processed_cache))
                del self.processed_cache[oldest_key]
            
            return img_float
            
        except Exception as e:
            print(f"加载图像数据失败 {img_id}: {e}")
            return None

class FilmPipeline:
    """胶片处理管道"""
    
    def __init__(self):
        self.linear_img = None
        self.cineon_img = None  # 密度域为0-3.0
        self.display_linear_img = None
        self.display_img = None
        
        self.image_loaded = False
        self.current_resolution_scale = 0.125
        
        # 默认参数 - 增加胶片曲线控制参数
        self.default_params = {
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
            'display_gamma': 2.2,
            
            # 胶片S型曲线参数 - 新增和优化
            'toe_strength': 0.8,      # 趾部强度 (0-2)
            'toe_slope': 0.5,         # 趾部斜率 (0-1)
            'shoulder_strength': 0.8,  # 肩部强度 (0-2)
            'shoulder_slope': 0.5,    # 肩部斜率 (0-1)
            'mid_contrast': 1.0,      # 中间调对比度 (0.1-3)
            'gamma': 1.0              # 中间调gamma
        }
        
        self.params = self.default_params.copy()
        self._cache_valid = False
        self._curve_cache = None
    
    def load_image(self, img_array):
        if img_array is None:
            return False
        
        self.linear_img = img_array
        self.image_loaded = True
        self._cache_valid = False
        self._curve_cache = None
        
        return True
    
    def set_resolution_scale(self, scale):
        self.current_resolution_scale = scale
        self._cache_valid = False
        self._curve_cache = None
        return True
    
    def apply_log_encoding(self):
        """应用对数编码，将线性数据转换为密度域(0-3.0)"""
        if self.linear_img is None:
            return None
        
        mode = self.params['processing_mode']
        
        img_data = self.linear_img.copy()
        if self.params['input_gain'] != 1.0:
            img_data = np.clip(img_data * self.params['input_gain'], 0, 1)
        
        epsilon = 1e-6  # 防止对数计算错误
        if mode == "log2log":
            # 直接映射到0-3.0密度域
            log_img = np.clip(img_data * 3.0, 0, 3.0)
            
        elif mode == "negative":
            positive_linear = 1.0 - img_data
            positive_linear = np.maximum(positive_linear, epsilon)
            # 负片密度计算，直接映射到0-3.0范围
            log_img = -np.log10(positive_linear)
            log_img = np.clip(log_img, 0, 3.0)
            
        else:  # reversal
            img_data = np.maximum(img_data, epsilon)
            log_img = -np.log10(img_data)
            log_img = np.clip(log_img, 0, 3.0)
        
        # 应用主增益和偏移
        master_gain = self.params['master_gain']
        master_shift = self.params['master_shift']
        log_img = log_img * master_gain + master_shift
        
        # Cineon标准映射 (密度范围0-3.0对应Cineon的95-685编码值)
        if self.params['fit_to_cineon']:
            # 从Cineon编码值转换为密度
            # Cineon编码范围: 95-685 -> 密度范围: 0-3.0
            cineon_min = 95.0 / 1023.0
            cineon_max = 685.0 / 1023.0
            log_img = (log_img / 3.0) * (cineon_max - cineon_min) + cineon_min
            # 转回密度域0-3.0
            log_img = (log_img - cineon_min) / (cineon_max - cineon_min) * 3.0
        
        self.cineon_img = np.clip(log_img, 0, 3.0)
        return self.cineon_img
    
    def apply_channel_alignment(self, log_img):
        """应用通道对齐，在密度域(0-3.0)上操作"""
        if log_img is None or log_img.ndim != 3:
            return log_img
        
        aligned_img = log_img.copy()
        
        # 对每个通道应用偏移和增益，保持在0-3.0范围内
        if self.params['r_shift'] != 0 or self.params['r_gain'] != 1.0:
            aligned_img[:,:,0] = np.clip(
                aligned_img[:,:,0] * self.params['r_gain'] + self.params['r_shift'], 
                0, 3.0
            )
        
        if self.params['g_shift'] != 0 or self.params['g_gain'] != 1.0:
            aligned_img[:,:,1] = np.clip(
                aligned_img[:,:,1] * self.params['g_gain'] + self.params['g_shift'], 
                0, 3.0
            )
        
        if self.params['b_shift'] != 0 or self.params['b_gain'] != 1.0:
            aligned_img[:,:,2] = np.clip(
                aligned_img[:,:,2] * self.params['b_gain'] + self.params['b_shift'], 
                0, 3.0
            )
        
        return aligned_img
    
    def log_to_linear(self, log_img):
        """将密度域(0-3.0)转换回线性空间"""
        if log_img is None:
            return None
        
        # 移除Cineon映射，回到原始密度域
        if self.params['fit_to_cineon']:
            cineon_min = 95.0 / 1023.0
            cineon_max = 685.0 / 1023.0
            log_img = (log_img - cineon_min) / (cineon_max - cineon_min) * 3.0
        
        # 确保密度在有效范围内
        density = np.clip(log_img, 0, 3.0)
        mode = self.params['processing_mode']
        
        if mode == "log2log":
            # 直接映射回0-1线性空间
            linear_img = density / 3.0
            
        elif mode == "negative":
            # 负片转换: 密度 -> 透射率 -> 线性
            transmittance = np.power(10.0, -density)
            linear_img = transmittance
            
        else:  # reversal
            # 反转片转换
            linear_img = np.power(10.0, -density)
        
        return np.clip(linear_img, 0, 1)
    
    def apply_color_adjustments(self, linear_img):
        """应用色彩调整到线性图像"""
        if linear_img is None:
            return None
        
        img = linear_img.copy()
        
        # 曝光调整
        exposure = self.params['exposure']
        if exposure != 0.0:
            exposure_gain = 2 ** exposure
            img = np.clip(img * exposure_gain, 0, 1)
        
        # 对比度调整
        contrast = self.params['contrast']
        if contrast != 1.0:
            img = 0.5 + (img - 0.5) * contrast
            img = np.clip(img, 0, 1)
        
        # 饱和度调整
        saturation = self.params['saturation']
        if saturation != 1.0 and img.ndim == 3 and img.shape[2] >= 3:
            luminance = np.mean(img, axis=2, keepdims=True)
            img = luminance + (img - luminance) * saturation
            img = np.clip(img, 0, 1)
        
        # 白平衡调整
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
        return self.display_linear_img
    
    def linear_to_display(self, linear_img):
        """将线性图像转换为显示空间"""
        if linear_img is None:
            return None
        
        gamma = 1.0 / self.params['display_gamma']
        display_img = np.power(np.clip(linear_img, 0, 1), gamma)
        
        if display_img.ndim == 2:
            display_img_8bit = (display_img * 255).astype(np.uint8)
        else:
            display_img_8bit = (display_img[:,:,:3] * 255).astype(np.uint8)
        
        return display_img_8bit
    
    def process_realtime(self, force_update=False):
        """实时处理图像"""
        if not self.image_loaded or self.linear_img is None:
            return None
        
        if not force_update and self._cache_valid and self.display_img is not None:
            return self.display_img
        
        try:
            log_img = self.apply_log_encoding()
            if log_img is None:
                return None
            
            aligned_img = self.apply_channel_alignment(log_img)
            if aligned_img is None:
                return None
            
            # 应用胶片S型曲线
            film_curve_img = self.apply_film_curve(aligned_img)
            if film_curve_img is None:
                return None
            
            linear_img = self.log_to_linear(film_curve_img)
            if linear_img is None:
                return None
            
            adjusted_img = self.apply_color_adjustments(linear_img)
            if adjusted_img is None:
                return None
            
            self.display_img = self.linear_to_display(adjusted_img)
            self._cache_valid = True
            self._curve_cache = None  # 处理时清空曲线缓存
            
            return self.display_img
            
        except Exception as e:
            print(f"处理错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def apply_film_curve(self, density_img):
        """应用胶片S型特性曲线到密度图像"""
        if density_img is None:
            return None
            
        # 提取曲线参数
        toe_strength = self.params['toe_strength']
        toe_slope = self.params['toe_slope']
        shoulder_strength = self.params['shoulder_strength']
        shoulder_slope = self.params['shoulder_slope']
        mid_contrast = self.params['mid_contrast']
        gamma = self.params['gamma']
        
        # 对每个通道应用曲线
        result = np.zeros_like(density_img)
        for i in range(min(3, density_img.shape[2])):
            result[:,:,i] = self.film_response_curve(
                density_img[:,:,i],
                toe_strength, toe_slope,
                shoulder_strength, shoulder_slope,
                mid_contrast, gamma
            )
        
        return result
    
    def film_response_curve(self, x, toe_strength, toe_slope, 
                           shoulder_strength, shoulder_slope, 
                           mid_contrast, gamma):
        """
        模拟真实胶片的S型响应曲线
        x: 输入密度值 (0-3.0)
        返回: 处理后的密度值 (0-3.0)
        """
        # 确保输入在有效范围内
        x = np.clip(x, 0, 3.0)
        
        # 归一化到0-1范围进行计算
        x_norm = x / 3.0
        
        # 趾部曲线 - 使用平滑的指数函数，增强EV<0区域控制
        toe = 1 - np.exp(-(x_norm **(1 / (toe_strength + 0.1)))* toe_slope * 5)
        
        # 肩部曲线 - 使用反向指数函数
        shoulder = 1 - np.exp(-(((1.0 - x_norm)** (1 / (shoulder_strength + 0.1))) * 
                               shoulder_slope * 5))
        shoulder = 1 - shoulder  # 反转肩部曲线
        
        # 中间调曲线 - 应用gamma和对比度
        mid = (x_norm) **(1/gamma) * mid_contrast
        mid = np.clip(mid, 0, 1)
        
        # 混合三条曲线，使用平滑的权重函数实现自然过渡
        # 趾部权重：在低区域较高，随x增加而衰减
        toe_weight = np.exp(-(((x_norm - 0.2) / 0.3)** 2))
        
        # 肩部权重：在高区域较高，随x减少而衰减
        shoulder_weight = np.exp(-(((x_norm - 0.8) / 0.3) **2))
        
        # 中间调权重：在中间区域较高
        mid_weight = 1 - np.maximum(toe_weight, shoulder_weight)
        
        # 组合曲线并转换回0-3.0密度范围
        result_norm = (toe * toe_weight + mid * mid_weight + shoulder * shoulder_weight)
        return np.clip(result_norm * 3.0, 0, 3.0)
    
    def get_density_curve_data(self):
        """获取胶片密度-曝光曲线数据"""
        if self.linear_img is None:
            return None
        
        # 处理图像以获取当前状态
        log_img = self.apply_log_encoding()
        if log_img is None:
            return None
        
        aligned_img = self.apply_channel_alignment(log_img)
        if aligned_img is None:
            return None
        
        # 应用胶片曲线
        film_curve_img = self.apply_film_curve(aligned_img)
        if film_curve_img is None:
            return None
        
        # 获取线性数据用于曝光计算
        linear_data = self.linear_img.copy()
        
        # 计算曝光值 (EV)，以18%灰为参考
        epsilon = 1e-7
        linear_data_clipped = np.clip(linear_data, epsilon, 1.0)
        ev_data = np.log2(linear_data_clipped / 0.18)
        
        # 密度数据（已应用胶片曲线）
        density_data = film_curve_img.copy()
        
        # 分箱数据用于绘图
        ev_bins = np.linspace(-8, 8, 97)  # 从-8到+8的96个箱
        
        curve_data = {}
        for i, channel in enumerate(['r', 'g', 'b']):
            ev_flat = ev_data[:,:,i].flatten()
            density_flat = density_data[:,:,i].flatten()
            
            # 移除无效值
            valid_mask = ~np.isnan(ev_flat) & ~np.isnan(density_flat)
            ev_valid = ev_flat[valid_mask]
            density_valid = density_flat[valid_mask]
            
            if len(ev_valid) < 10:  # 需要足够的样本
                continue
            
            # 按EV值分箱
            ev_indices = np.digitize(ev_valid, ev_bins)
            
            # 计算每个箱的平均密度
            bin_densities = []
            bin_centers = []
            
            for bin_idx in range(1, len(ev_bins)):
                mask = (ev_indices == bin_idx)
                if np.sum(mask) > 0:
                    bin_density = np.mean(density_valid[mask])
                    bin_center = (ev_bins[bin_idx-1] + ev_bins[bin_idx]) / 2
                    bin_densities.append(bin_density)
                    bin_centers.append(bin_center)
            
            curve_data[channel] = {
                'ev': np.array(bin_centers),
                'density': np.array(bin_densities)  # 密度范围0-3.0
            }
        
        return curve_data
    
    def get_histogram_data(self):
        """获取直方图数据用于波形显示"""
        if self.display_linear_img is None:
            return None, None
        
        img = self.display_linear_img
        
        if img.ndim == 3 and img.shape[2] >= 3:
            r_data = img[:,:,0].flatten()
            g_data = img[:,:,1].flatten()
            b_data = img[:,:,2].flatten()
            
            r_hist, r_bins = np.histogram(r_data, bins=128, range=(0, 1), density=True)
            g_hist, g_bins = np.histogram(g_data, bins=128, range=(0, 1), density=True)
            b_hist, b_bins = np.histogram(b_data, bins=128, range=(0, 1), density=True)
            
            return (r_hist, g_hist, b_hist), (r_bins, g_bins, b_bins)
        
        elif img.ndim == 2:
            gray_data = img.flatten()
            gray_hist, gray_bins = np.histogram(gray_data, bins=128, range=(0, 1), density=True)
            return (gray_hist, gray_hist, gray_hist), (gray_bins, gray_bins, gray_bins)
        
        return None, None
    
    def update_parameter(self, name, value):
        if name in self.params:
            self.params[name] = value
            self._cache_valid = False
            self._curve_cache = None
            return True
        return False
    
    def set_processing_mode(self, mode):
        if mode in ['negative', 'reversal', 'log2log']:
            self.params['processing_mode'] = mode
            self._cache_valid = False
            self._curve_cache = None
            return True
        return False
    
    def reset_parameters(self):
        self.params = self.default_params.copy()
        self._cache_valid = False
        self._curve_cache = None
    
    def export_preset(self):
        """导出当前参数为预设"""
        preset = {
            'params': self.params.copy(),
            'metadata': {
                'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0'
            }
        }
        return preset
    
    def import_preset(self, preset_data):
        """从预设导入参数"""
        if 'params' in preset_data:
            for key, value in preset_data['params'].items():
                if key in self.params:
                    self.params[key] = value
            self._cache_valid = False
            self._curve_cache = None
            return True
        return False

class FilmProcessorUI:
    def __init__(self):
        self.image_manager = ImageManager()
        self.pipeline = FilmPipeline()
        self.lang = LanguageManager()  # 语言管理器
        
        self.current_file = None
        self.display_image = None
        self.display_photo = None
        self.selected_images = []
        
        self.update_timer = None
        self.update_delay = 150
        self.param_changed = False
        self.last_visual_update = 0
        self.visual_update_interval = 0.3
        
        self.resolution_options = {
            "完整 (100%)": 1.0,
            "高 (75%)": 0.75,
            "中 (50%)": 0.5,
            "低 (25%)": 0.25,
            "最低 (12.5%)": 0.125
        }
        
        # 黑色主题颜色
        self.bg_color = '#1e1e1e'
        self.frame_bg = '#252526'
        self.text_color = '#d4d4d4'
        self.accent_color = '#007acc'
        self.highlight_color = '#2d2d30'
        
        # 曲线显示设置
        self.ev_range = [-6, 6]
        self.zoom_factor = 1.0  # 缩放因子
        
        self.root = tk.Tk()
        self.root.title(self.lang.get('title'))
        self.root.geometry("1800x1000")
        
        # 设置关闭协议
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 设置黑色主题
        self.root.configure(bg=self.bg_color)
        
        # 创建菜单栏
        self.setup_menu()
        
        self.setup_styles()
        self.setup_ui()
        self.root.after(100, self.check_realtime_update)
        
        # 绑定键盘事件
        self.root.bind('<Control-MouseWheel>', self.on_mousewheel)
        self.root.bind('<Control-Button-4>', self.on_mousewheel)  # Linux
        self.root.bind('<Control-Button-5>', self.on_mousewheel)  # Linux
    
    def setup_menu(self):
        """设置菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 语言菜单（独立顶层菜单）
        language_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.lang.get('language'), menu=language_menu)
        language_menu.add_command(label=self.lang.get('chinese'), 
                                 command=lambda: self.change_language('zh'))
        language_menu.add_command(label=self.lang.get('english'), 
                                 command=lambda: self.change_language('en'))
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.lang.get('file'), menu=file_menu)
        file_menu.add_command(label=self.lang.get('exit'), command=self.on_closing)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.lang.get('help'), menu=help_menu)
        help_menu.add_command(label=self.lang.get('about'), command=self.show_about)
    
    def change_language(self, lang):
        """更改语言"""
        if self.lang.set_language(lang):
            messagebox.showinfo(self.lang.get('language'), 
                              f"{self.lang.get('restart_required')}")
    
    def show_about(self):
        """显示关于对话框"""
        messagebox.showinfo(self.lang.get('about_title'), 
                          self.lang.get('about_text'))
    
    def on_mousewheel(self, event):
        """处理Ctrl+滚轮缩放事件"""
        if MATPLOTLIB_AVAILABLE and self.pipeline.image_loaded:
            # Windows和Mac使用event.delta，Linux使用event.num
            if hasattr(event, 'delta'):
                delta = event.delta
            else:
                delta = 120 if event.num == 4 else -120
            
            # 缩放因子变化
            zoom_change = 1.1 if delta > 0 else 0.9
            self.zoom_factor *= zoom_change
            self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))  # 限制缩放范围
            
            # 更新曲线
            self.update_density_curve()
    
    def on_closing(self):
        """关闭窗口时的清理工作"""
        # 停止所有计时器
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        
        # 清理matplotlib资源
        if MATPLOTLIB_AVAILABLE:
            import matplotlib.pyplot as plt
            plt.close('all')
        
        # 清理图像缓存
        if hasattr(self.image_manager, 'clear_cache'):
            self.image_manager.clear_cache()
        
        # 确保所有线程安全退出
        self.pipeline.linear_img = None
        self.pipeline.display_img = None
        
        # 使用destroy()销毁窗口
        self.root.destroy()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置黑色主题
        style.configure('.', 
                       background=self.bg_color,
                       foreground=self.text_color,
                       fieldbackground=self.frame_bg)
        
        style.configure('Title.TLabel', 
                       font=('Microsoft YaHei', 16, 'bold'),
                       foreground=self.text_color)
        
        style.configure('Heading.TLabel', 
                       font=('Microsoft YaHei', 11, 'bold'),
                       foreground=self.text_color)
        
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabelframe', 
                       background=self.frame_bg,
                       foreground=self.text_color,
                       relief='flat',
                       borderwidth=1)
        
        style.configure('TLabelframe.Label', 
                       background=self.frame_bg,
                       foreground=self.text_color)
        
        style.configure('TNotebook', background=self.bg_color)
        style.configure('TNotebook.Tab', 
                       background=self.frame_bg,
                       foreground=self.text_color,
                       padding=[10, 5])
        
        style.map('TNotebook.Tab',
                 background=[('selected', self.accent_color)],
                 foreground=[('selected', 'white')])
        
        style.configure('TButton',
                       background=self.highlight_color,
                       foreground=self.text_color,
                       borderwidth=1,
                       focusthickness=3,
                       focuscolor='none')
        
        style.map('TButton',
                 background=[('active', self.accent_color),
                            ('disabled', '#3c3c3c')],
                 foreground=[('active', 'white'),
                            ('disabled', '#6c6c6c')])
        
        style.configure('TScale',
                       background=self.bg_color,
                       troughcolor=self.highlight_color)
        
        style.configure('TCheckbutton',
                       background=self.frame_bg,
                       foreground=self.text_color)
        
        style.configure('TRadiobutton',
                       background=self.frame_bg,
                       foreground=self.text_color)
    
    def setup_ui(self):
        # 主布局分为三列
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：图像管理面板
        left_panel = ttk.Frame(main_paned, width=300)
        self.setup_image_manager(left_panel)
        main_paned.add(left_panel)
        
        # 中间：图像预览和曲线图
        middle_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        
        # 图像预览
        preview_frame = ttk.LabelFrame(middle_paned, text=self.lang.get('image_preview'), padding=10)
        self.setup_image_preview(preview_frame)
        middle_paned.add(preview_frame)
        
        # 可视化图表（放回预览下方）
        if MATPLOTLIB_AVAILABLE:
            viz_frame = ttk.LabelFrame(middle_paned, text=self.lang.get('visualization'), padding=10)
            self.setup_visualization(viz_frame)
            middle_paned.add(viz_frame)
        
        main_paned.add(middle_paned)
        
        # 右侧：参数控制面板
        right_panel = ttk.Frame(main_paned, width=450)
        self.setup_parameter_panel(right_panel)
        main_paned.add(right_panel)
        
        # 底部状态栏
        self.setup_status_bar()
    
    def setup_image_manager(self, parent):
        # 标题
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text=self.lang.get('image_management'), style='Title.TLabel').pack(side=tk.LEFT)
        
        # 控制按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(btn_frame, text=self.lang.get('add_images'), 
                  command=self.add_images, width=15).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text=self.lang.get('batch_process'), 
                  command=self.batch_process, width=15).pack(side=tk.LEFT, padx=2)
        
        # 图像列表
        list_frame = ttk.LabelFrame(parent, text=self.lang.get('image_list'), padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # 列表控件
        columns = ('name', 'size', 'status')
        self.image_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', height=15)
        
        # 配置列
        self.image_tree.heading('#0', text='', anchor=tk.W)
        self.image_tree.column('#0', width=30, stretch=False)
        
        self.image_tree.heading('name', text=self.lang.get('filename'), anchor=tk.W)
        self.image_tree.column('name', width=150)
        
        self.image_tree.heading('size', text=self.lang.get('size'), anchor=tk.W)
        self.image_tree.column('size', width=80)
        
        self.image_tree.heading('status', text=self.lang.get('status'), anchor=tk.W)
        self.image_tree.column('status', width=60)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_tree.yview)
        self.image_tree.configure(yscrollcommand=scrollbar.set)
        
        self.image_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定选择事件
        self.image_tree.bind('<<TreeviewSelect>>', self.on_image_selected)
    
    def setup_image_preview(self, parent):
        # 控制栏
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 处理模式
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(mode_frame, text=self.lang.get('mode') + ":").pack(side=tk.LEFT)
        
        self.mode_var = tk.StringVar(value="negative")
        modes = [(self.lang.get('negative'), "negative"), 
                (self.lang.get('reversal'), "reversal"), 
                (self.lang.get('log'), "log2log")]
        
        for text, value in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                           value=value, command=self.on_mode_changed).pack(side=tk.LEFT, padx=2)
        
        # 分辨率
        res_frame = ttk.Frame(control_frame)
        res_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(res_frame, text=self.lang.get('preview_resolution') + ":").pack(side=tk.LEFT)
        
        self.resolution_var = tk.StringVar(value=self.lang.get('minimum') + " (12.5%)")
        resolution_menu = ttk.OptionMenu(res_frame, self.resolution_var, 
                                        self.lang.get('minimum') + " (12.5%)", 
                                        self.lang.get('full') + " (100%)",
                                        self.lang.get('high') + " (75%)",
                                        self.lang.get('medium') + " (50%)",
                                        self.lang.get('low') + " (25%)",
                                        self.lang.get('minimum') + " (12.5%)",
                                        command=self.on_resolution_changed)
        resolution_menu.config(width=15)
        resolution_menu.pack(side=tk.LEFT, padx=5)
        
        # 实时渲染
        self.realtime_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text=self.lang.get('real_time'), 
                       variable=self.realtime_var).pack(side=tk.LEFT, padx=10)
        
        # 右侧按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text=self.lang.get('save_current'), 
                  command=self.save_current_image, width=12).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text=self.lang.get('save_all'), 
                  command=self.save_all_images, width=12).pack(side=tk.LEFT, padx=2)
        
        # 图像显示区域 - 简单的Canvas，没有滚动条
        self.image_canvas = tk.Canvas(parent, bg='black')
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 图像信息
        self.image_info_label = ttk.Label(parent, text=self.lang.get('no_image_selected'), relief='sunken')
        self.image_info_label.pack(fill=tk.X, pady=(10, 0))
    
    def setup_visualization(self, parent):
        """设置可视化图表区域（密度曲线和波形图）- 保持英文"""
        # 创建选项卡切换两种图表
        self.viz_notebook = ttk.Notebook(parent)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 选项卡1：密度-曝光曲线
        curve_frame = ttk.Frame(self.viz_notebook)
        self.setup_density_curve(curve_frame)
        self.viz_notebook.add(curve_frame, text="Density-Exposure Curve")
        
        # 选项卡2：RGB分量图
        waveform_frame = ttk.Frame(self.viz_notebook)
        self.setup_waveform_display(waveform_frame)
        self.viz_notebook.add(waveform_frame, text="RGB Waveform")
    
    def setup_density_curve(self, parent):
        """设置密度-曝光曲线图 - 保持英文"""
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(parent, text="Matplotlib not available").pack(expand=True)
            return
        
        # 控制栏
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 提示文本
        ttk.Label(control_frame, text="Ctrl+MouseWheel to zoom, EV Range:").pack(side=tk.LEFT, padx=5)
        
        # EV范围选择
        self.ev_range_var = tk.StringVar(value="-6 to +6")
        ev_range_menu = ttk.OptionMenu(control_frame, self.ev_range_var, 
                                       "-6 to +6", "-4 to +4", "-8 to +8",
                                       command=self.on_ev_range_changed)
        ev_range_menu.pack(side=tk.LEFT, padx=5)
        
        # 重置缩放按钮
        ttk.Button(control_frame, text="Reset Zoom", 
                  command=self.reset_zoom).pack(side=tk.RIGHT, padx=5)
        
        # 创建图形
        self.figure_curve = Figure(figsize=(8, 4), dpi=80)
        self.ax_curve = self.figure_curve.add_subplot(111)
        
        # 设置图表样式
        self.ax_curve.set_facecolor('#1e1e1e')
        self.ax_curve.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.ax_curve.set_xlabel('Exposure (EV)', fontsize=10)
        self.ax_curve.set_ylabel('Density', fontsize=10)
        self.ax_curve.set_title('Film Density-Exposure Curve', fontsize=12, fontweight='bold')
        
        # 设置初始坐标轴范围
        self.ax_curve.set_xlim(-6, 6)
        self.ax_curve.set_ylim(0, 3.0)  # 密度范围改为0-3.0
        
        # 添加参考线
        self.ax_curve.axvline(x=0, color='gray', alpha=0.5, linestyle='--', linewidth=0.5)
        self.ax_curve.axhline(y=1.5, color='gray', alpha=0.3, linestyle=':', linewidth=0.5)
        
        # 初始化曲线
        self.line_curve_r, = self.ax_curve.plot([], [], color='red', linewidth=2, alpha=0.8, label='Red')
        self.line_curve_g, = self.ax_curve.plot([], [], color='green', linewidth=2, alpha=0.8, label='Green')
        self.line_curve_b, = self.ax_curve.plot([], [], color='blue', linewidth=2, alpha=0.8, label='Blue')
        
        # 参考曲线
        self.line_reference, = self.ax_curve.plot([], [], color='white', linewidth=1, alpha=0.5, linestyle='--', label='Reference')
        
        # 添加图例
        self.ax_curve.legend(loc='upper right', fontsize=9)
        
        self.curve_canvas = FigureCanvasTkAgg(self.figure_curve, parent)
        self.curve_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 绑定键盘事件到画布
        self.curve_canvas.mpl_connect('scroll_event', self.on_curve_scroll)
    
    def on_curve_scroll(self, event):
        """处理曲线图上的滚轮事件"""
        if event.key == 'control':
            # 计算缩放因子
            zoom_change = 1.1 if event.step > 0 else 0.9
            self.zoom_factor *= zoom_change
            self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))  # 限制缩放范围
            
            # 更新曲线
            self.update_density_curve()
    
    def reset_zoom(self):
        """重置缩放"""
        self.zoom_factor = 1.0
        self.update_density_curve()
    
    def setup_waveform_display(self, parent):
        """设置RGB波形图 - 保持英文"""
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(parent, text="Matplotlib not available").pack(expand=True)
            return
        
        self.figure_wave = Figure(figsize=(8, 6), dpi=80)
        
        # 创建三个子图
        self.ax_wave_r = self.figure_wave.add_subplot(311)
        self.ax_wave_g = self.figure_wave.add_subplot(312)
        self.ax_wave_b = self.figure_wave.add_subplot(313)
        
        # 设置样式
        for ax in [self.ax_wave_r, self.ax_wave_g, self.ax_wave_b]:
            ax.set_facecolor('#1e1e1e')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(colors='#d4d4d4')
        
        self.ax_wave_r.set_title("Red Channel", color='#d4d4d4')
        self.ax_wave_g.set_title("Green Channel", color='#d4d4d4')
        self.ax_wave_b.set_title("Blue Channel", color='#d4d4d4')
        self.ax_wave_b.set_xlabel("Brightness Value (0-1)", color='#d4d4d4')
        
        # 初始化曲线
        self.line_wave_r, = self.ax_wave_r.plot([0], [0], color='red', linewidth=1.5)
        self.line_wave_g, = self.ax_wave_g.plot([0], [0], color='green', linewidth=1.5)
        self.line_wave_b, = self.ax_wave_b.plot([0], [0], color='blue', linewidth=1.5)
        
        self.waveform_canvas = FigureCanvasTkAgg(self.figure_wave, parent)
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_parameter_panel(self, parent):
        # 使用Notebook组织参数
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 核心参数页
        core_frame = ttk.Frame(self.notebook)
        self.setup_core_parameters(core_frame)
        self.notebook.add(core_frame, text=self.lang.get('core_parameters'))
        
        # 通道对齐页
        channel_frame = ttk.Frame(self.notebook)
        self.setup_channel_alignment(channel_frame)
        self.notebook.add(channel_frame, text=self.lang.get('channel_alignment'))
        
        # 色彩调整页
        color_frame = ttk.Frame(self.notebook)
        self.setup_color_adjustments(color_frame)
        self.notebook.add(color_frame, text=self.lang.get('color_adjustments'))
        
        # 预设管理页
        preset_frame = ttk.Frame(self.notebook)
        self.setup_preset_manager(preset_frame)
        self.notebook.add(preset_frame, text=self.lang.get('preset_management'))
    
    def setup_core_parameters(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0, bg=self.frame_bg)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 输入调整
        input_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('input_adjustment'), padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.input_gain_var = tk.DoubleVar(value=1.0)
        self.create_parameter_row(input_frame, self.lang.get('input_gain') + ":", 
                                 self.input_gain_var, 0.1, 5.0, 0.1, 'input_gain')
        
        # 对数域调整
        master_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('log_domain_adjustment'), padding=10)
        master_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.master_gain_var = tk.DoubleVar(value=1.0)
        self.create_parameter_row(master_frame, self.lang.get('master_gain') + ":", 
                                 self.master_gain_var, 0.5, 2.0, 0.05, 'master_gain')
        
        self.master_shift_var = tk.DoubleVar(value=0.0)
        self.create_parameter_row(master_frame, self.lang.get('master_shift') + ":", 
                                 self.master_shift_var, -0.5, 0.5, 0.01, 'master_shift')
        
        # 胶片曲线参数
        film_curve_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('film_curve'), padding=10)
        film_curve_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.toe_strength_var = tk.DoubleVar(value=0.8)
        self.create_parameter_row(film_curve_frame, self.lang.get('toe_strength') + ":", 
                                 self.toe_strength_var, 0.0, 2.0, 0.05, 'toe_strength')
        
        self.toe_slope_var = tk.DoubleVar(value=0.5)
        self.create_parameter_row(film_curve_frame, self.lang.get('toe_slope') + ":", 
                                 self.toe_slope_var, 0.0, 1.0, 0.01, 'toe_slope')
        
        self.shoulder_strength_var = tk.DoubleVar(value=0.8)
        self.create_parameter_row(film_curve_frame, self.lang.get('shoulder_strength') + ":", 
                                 self.shoulder_strength_var, 0.0, 2.0, 0.05, 'shoulder_strength')
        
        self.shoulder_slope_var = tk.DoubleVar(value=0.5)
        self.create_parameter_row(film_curve_frame, self.lang.get('shoulder_slope') + ":", 
                                 self.shoulder_slope_var, 0.0, 1.0, 0.01, 'shoulder_slope')
        
        self.mid_contrast_var = tk.DoubleVar(value=1.0)
        self.create_parameter_row(film_curve_frame, self.lang.get('mid_contrast') + ":", 
                                 self.mid_contrast_var, 0.1, 3.0, 0.05, 'mid_contrast')
        
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.create_parameter_row(film_curve_frame, self.lang.get('gamma') + ":", 
                                 self.gamma_var, 0.3, 3.0, 0.05, 'gamma')
        
        # Cineon设置
        cineon_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('cineon_settings'), padding=10)
        cineon_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.fit_cineon_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cineon_frame, text=self.lang.get('fit_to_cineon'), 
                       variable=self.fit_cineon_var,
                       command=lambda: self.update_parameter('fit_to_cineon', 
                                                            self.fit_cineon_var.get())).pack(anchor=tk.W)
        
        # 显示设置
        display_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('display_settings'), padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.display_gamma_var = tk.DoubleVar(value=2.2)
        
        gamma_values = [(f"1.0 ({self.lang.get('linear')})", 1.0), 
                       (f"1.8 ({self.lang.get('mac')})", 1.8), 
                       (f"2.2 ({self.lang.get('sRGB')})", 2.2), 
                       (f"2.4 ({self.lang.get('rec709')})", 2.4)]
        
        for text, value in gamma_values:
            ttk.Radiobutton(display_frame, text=text, variable=self.display_gamma_var,
                           value=value, command=self.on_gamma_changed).pack(anchor=tk.W)
    
    def setup_channel_alignment(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0, bg=self.frame_bg)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 红色通道
        red_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('red_channel'), padding=10)
        red_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.r_shift_var = tk.DoubleVar(value=0.0)
        self.create_parameter_row(red_frame, self.lang.get('shift') + ":", 
                                 self.r_shift_var, -0.5, 0.5, 0.01, 'r_shift')
        
        self.r_gain_var = tk.DoubleVar(value=1.0)
        self.create_parameter_row(red_frame, self.lang.get('gain') + ":", 
                                 self.r_gain_var, 0.5, 2.0, 0.05, 'r_gain')
        
        # 绿色通道
        green_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('green_channel'), padding=10)
        green_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.g_shift_var = tk.DoubleVar(value=0.0)
        self.create_parameter_row(green_frame, self.lang.get('shift') + ":", 
                                 self.g_shift_var, -0.5, 0.5, 0.01, 'g_shift')
        
        self.g_gain_var = tk.DoubleVar(value=1.0)
        self.create_parameter_row(green_frame, self.lang.get('gain') + ":", 
                                 self.g_gain_var, 0.5, 2.0, 0.05, 'g_gain')
        
        # 蓝色通道
        blue_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('blue_channel'), padding=10)
        blue_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.b_shift_var = tk.DoubleVar(value=0.0)
        self.create_parameter_row(blue_frame, self.lang.get('shift') + ":", 
                                 self.b_shift_var, -0.5, 0.5, 0.01, 'b_shift')
        
        self.b_gain_var = tk.DoubleVar(value=1.0)
        self.create_parameter_row(blue_frame, self.lang.get('gain') + ":", 
                                 self.b_gain_var, 0.5, 2.0, 0.05, 'b_gain')
    
    def setup_color_adjustments(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0, bg=self.frame_bg)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 曝光与对比度
        exp_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('exposure_contrast'), padding=10)
        exp_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.exposure_var = tk.DoubleVar(value=0.0)
        self.create_parameter_row(exp_frame, self.lang.get('exposure') + ":", 
                                 self.exposure_var, -3.0, 3.0, 0.1, 'exposure')
        
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.create_parameter_row(exp_frame, self.lang.get('contrast') + ":", 
                                 self.contrast_var, 0.1, 3.0, 0.05, 'contrast')
        
        # 饱和度
        sat_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('saturation'), padding=10)
        sat_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.saturation_var = tk.DoubleVar(value=1.0)
        self.create_parameter_row(sat_frame, self.lang.get('saturation') + ":", 
                                 self.saturation_var, 0.0, 3.0, 0.05, 'saturation')
        
        # 白平衡
        temp_frame = ttk.LabelFrame(scrollable_frame, text=self.lang.get('white_balance'), padding=10)
        temp_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.temperature_var = tk.DoubleVar(value=0.0)
        self.create_parameter_row(temp_frame, self.lang.get('temperature') + ":", 
                                 self.temperature_var, -2.0, 2.0, 0.05, 'temperature')
        
        self.tint_var = tk.DoubleVar(value=0.0)
        self.create_parameter_row(temp_frame, self.lang.get('tint') + ":", 
                                 self.tint_var, -2.0, 2.0, 0.05, 'tint')
    
    def setup_preset_manager(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 预设说明
        ttk.Label(frame, text=self.lang.get('preset_management'), style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(frame, text=self.lang.get('preset_explanation'), justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 20))
        
        # 预设名称输入
        name_frame = ttk.Frame(frame)
        name_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(name_frame, text=self.lang.get('preset_name') + ":").pack(side=tk.LEFT)
        self.preset_name_var = tk.StringVar(value="我的预设")
        ttk.Entry(name_frame, textvariable=self.preset_name_var, width=20).pack(side=tk.LEFT, padx=5)
        
        # 按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text=self.lang.get('export_preset'), 
                  command=self.export_preset, width=15).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text=self.lang.get('import_preset'), 
                  command=self.import_preset, width=15).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text=self.lang.get('reset_parameters'), 
                  command=self.reset_parameters, width=15).pack(side=tk.LEFT, padx=2)
        
        # 预设列表
        list_frame = ttk.LabelFrame(frame, text=self.lang.get('preset_list'), padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.preset_listbox = tk.Listbox(list_frame, bg=self.frame_bg, fg=self.text_color, height=8)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.preset_listbox.yview)
        self.preset_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_parameter_row(self, parent, label, variable, from_val, to_val, resolution, param_name):
        """创建带滑块和数值输入的参数行"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 8))
        
        # 标签
        ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
        
        # 滑块
        slider_frame = ttk.Frame(frame)
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        slider = ttk.Scale(slider_frame, from_=from_val, to=to_val, variable=variable,
                          orient=tk.HORIZONTAL, length=150)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 数值输入框
        entry_frame = ttk.Frame(frame)
        entry_frame.pack(side=tk.RIGHT)
        
        entry_var = tk.StringVar(value=f"{variable.get():.3f}")
        entry = ttk.Entry(entry_frame, textvariable=entry_var, width=8)
        entry.pack(side=tk.LEFT, padx=2)
        
        # 更新标签显示
        value_label = ttk.Label(entry_frame, text=f"{variable.get():.3f}", width=8)
        value_label.pack(side=tk.LEFT)
        
        # 绑定事件
        def on_slider_change(val):
            float_val = float(val)
            entry_var.set(f"{float_val:.3f}")
            value_label.config(text=f"{float_val:.3f}")
            self.pipeline.update_parameter(param_name, float_val)
            self.param_changed = True
        
        def on_entry_change(*args):
            try:
                float_val = float(entry_var.get())
                if from_val <= float_val <= to_val:
                    variable.set(float_val)
                    value_label.config(text=f"{float_val:.3f}")
                    self.pipeline.update_parameter(param_name, float_val)
                    self.param_changed = True
            except ValueError:
                pass
        
        slider.config(command=lambda v: on_slider_change(v))
        entry_var.trace('w', on_entry_change)
        
        return value_label
    
    def setup_status_bar(self):
        status_frame = ttk.Frame(self.root, relief='sunken')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value=self.lang.get('ready'))
        status_label = ttk.Label(status_frame, textvariable=self.status_var, padding=5)
        status_label.pack(side=tk.LEFT)
    
    def on_ev_range_changed(self, value):
        """处理EV范围改变"""
        if value == "-4 to +4":
            self.ev_range = [-4, 4]
        elif value == "-8 to +8":
            self.ev_range = [-8, 8]
        else:  # "-6 to +6"
            self.ev_range = [-6, 6]
        
        self.update_density_curve()
    
    def update_density_curve(self):
        """更新密度-曝光曲线"""
        if not MATPLOTLIB_AVAILABLE or not self.pipeline.image_loaded:
            return
        
        try:
            # 获取曲线数据
            curve_data = self.pipeline.get_density_curve_data()
            
            if curve_data is None or len(curve_data) == 0:
                # 绘制空曲线
                self.line_curve_r.set_data([], [])
                self.line_curve_g.set_data([], [])
                self.line_curve_b.set_data([], [])
                self.line_reference.set_data([], [])
                self.curve_canvas.draw()
                return
            
            # 清空之前的数据
            self.line_curve_r.set_data([], [])
            self.line_curve_g.set_data([], [])
            self.line_curve_b.set_data([], [])
            
            # 绘制每个通道
            colors = {'r': 'red', 'g': 'green', 'b': 'blue'}
            lines = {'r': self.line_curve_r, 'g': self.line_curve_g, 'b': self.line_curve_b}
            
            all_ev_points = []
            all_density_points = []
            
            for channel in ['r', 'g', 'b']:
                if channel in curve_data:
                    data = curve_data[channel]
                    if len(data['ev']) > 0 and len(data['density']) > 0:
                        # 应用缩放因子（不改变实际数据，只改变显示）
                        # 缩放是在坐标轴层面处理的，所以我们直接使用原始数据
                        lines[channel].set_data(data['ev'], data['density'])
                        
                        # 收集点用于参考线
                        all_ev_points.extend(data['ev'])
                        all_density_points.extend(data['density'])
            
            # 创建参考曲线（所有通道的平均值）
            if all_ev_points and all_density_points:
                # 按EV排序并创建平滑的参考曲线
                sorted_indices = np.argsort(all_ev_points)
                ev_sorted = np.array(all_ev_points)[sorted_indices]
                density_sorted = np.array(all_density_points)[sorted_indices]
                
                # 使用移动平均进行平滑
                window_size = min(5, len(ev_sorted) // 4)
                if window_size > 1:
                    kernel = np.ones(window_size) / window_size
                    ref_density = np.convolve(density_sorted, kernel, mode='valid')
                    ref_ev = ev_sorted[window_size//2:-(window_size//2)]
                    if len(ref_ev) == len(ref_density):
                        self.line_reference.set_data(ref_ev, ref_density)
            
            # 更新坐标轴范围
            self.ax_curve.set_xlim(self.ev_range[0], self.ev_range[1])
            
            # 根据缩放因子调整Y轴
            if all_density_points:
                # 基础范围
                base_y_min = 0
                base_y_max = 3.0
                
                # 应用缩放因子
                y_center = (base_y_min + base_y_max) / 2
                y_range = (base_y_max - base_y_min) / self.zoom_factor
                
                y_min = max(0, y_center - y_range/2)
                y_max = min(3.0, y_center + y_range/2)
                
                self.ax_curve.set_ylim(y_min, y_max)
            
            # 重绘画布
            self.curve_canvas.draw()
            
        except Exception as e:
            print(f"更新密度曲线失败: {e}")
    
    def update_waveform(self):
        """更新RGB波形图"""
        if not MATPLOTLIB_AVAILABLE or not self.pipeline.image_loaded:
            return
        
        try:
            hist_data, bins_data = self.pipeline.get_histogram_data()
            
            if hist_data is None or bins_data is None:
                return
            
            r_hist, g_hist, b_hist = hist_data
            r_bins, g_bins, b_bins = bins_data
            
            # 清除并重新绘制
            self.ax_wave_r.clear()
            self.ax_wave_g.clear()
            self.ax_wave_b.clear()
            
            # 绘制新的直方图
            self.ax_wave_r.fill_between(r_bins[:-1], r_hist, alpha=0.3, color='red')
            self.ax_wave_r.plot(r_bins[:-1], r_hist, color='red', linewidth=1.5)
            
            self.ax_wave_g.fill_between(g_bins[:-1], g_hist, alpha=0.3, color='green')
            self.ax_wave_g.plot(g_bins[:-1], g_hist, color='green', linewidth=1.5)
            
            self.ax_wave_b.fill_between(b_bins[:-1], b_hist, alpha=0.3, color='blue')
            self.ax_wave_b.plot(b_bins[:-1], b_hist, color='blue', linewidth=1.5)
            
            # 再次设置图形
            max_hist = max(r_hist.max(), g_hist.max(), b_hist.max())
            y_max = max(5, max_hist * 1.2)
            
            for ax in [self.ax_wave_r, self.ax_wave_g, self.ax_wave_b]:
                ax.set_facecolor('#1e1e1e')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, y_max)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.tick_params(colors='#d4d4d4')
            
            self.ax_wave_r.set_title("Red Channel", color='#d4d4d4')
            self.ax_wave_g.set_title("Green Channel", color='#d4d4d4')
            self.ax_wave_b.set_title("Blue Channel", color='#d4d4d4')
            self.ax_wave_b.set_xlabel("Brightness Value (0-1)", color='#d4d4d4')
            
            # 重绘画布
            self.waveform_canvas.draw()
            
        except Exception as e:
            print(f"更新波形图失败: {e}")
    
    def add_images(self):
        """添加多个图像"""
        file_paths = filedialog.askopenfilenames(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.tif;*.tiff;*.jpg;*.jpeg;*.png;*.exr"),
                ("TIFF文件", "*.tif;*.tiff"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_paths:
            for file_path in file_paths:
                img_id = self.image_manager.add_image(file_path)
                
                # 添加到列表
                self.image_tree.insert('', 'end', iid=img_id, 
                                      values=(os.path.basename(file_path), self.lang.get('loading'), self.lang.get('pending')))
            
            self.status_var.set(f"添加了 {len(file_paths)} 个图像")
            
            # 如果这是第一个图像，选择它
            if not self.selected_images and self.image_manager.images:
                first_id = list(self.image_manager.images.keys())[0]
                self.image_tree.selection_set(first_id)
                self.on_image_selected()
    
    def on_image_selected(self, event=None):
        """当图像被选中时"""
        selected_ids = self.image_tree.selection()
        
        if not selected_ids:
            return
        
        self.selected_images = [int(iid) for iid in selected_ids]
        
        # 加载第一个选中的图像进行预览
        if self.selected_images:
            first_id = self.selected_images[0]
            self.load_image_for_preview(first_id)
    
    def load_image_for_preview(self, img_id):
        """加载图像进行预览"""
        if img_id not in self.image_manager.images:
            return
        
        scale = self.resolution_options[self.resolution_var.get()]
        
        def load_thread():
            self.status_var.set(self.lang.get('loading'))
            
            # 加载图像数据
            img_data = self.image_manager.get_image_data(img_id, scale)
            
            if img_data is not None:
                self.root.after(0, lambda: self.on_image_loaded(img_id, img_data))
            else:
                self.root.after(0, lambda: self.status_var.set(self.lang.get('load_failed')))
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def on_image_loaded(self, img_id, img_data):
        """图像加载完成"""
        # 更新管道
        self.pipeline.load_image(img_data)
        self.pipeline.set_resolution_scale(self.resolution_options[self.resolution_var.get()])
        
        # 更新显示
        self.process_and_update_display(force_visual=True)
        
        # 更新列表状态
        img_info = self.image_manager.images[img_id]
        metadata = img_info.get('metadata', {})
        size_str = f"{metadata.get('width', 0)}x{metadata.get('height', 0)}" if metadata else self.lang.get('loaded')
        
        self.image_tree.item(img_id, values=(img_info['name'], size_str, self.lang.get('loaded')))
        self.status_var.set(f"{self.lang.get('loaded')}: {img_info['name']}")
        
        # 更新图像信息
        self.image_info_label.config(
            text=f"{img_info['name']} - {img_data.shape[1]}x{img_data.shape[0]} - {self.mode_var.get()}"
        )
    
    def process_and_update_display(self, force_visual=False):
        """处理并更新显示"""
        if not self.pipeline.image_loaded:
            return
        
        try:
            display_array = self.pipeline.process_realtime()
            
            if display_array is not None:
                # 转换为PIL图像
                if display_array.ndim == 2:
                    pil_img = Image.fromarray(display_array, mode='L')
                else:
                    pil_img = Image.fromarray(display_array, mode='RGB')
                
                self.update_display_image(pil_img)
                
                # 更新可视化图表
                current_time = time.time()
                if (force_visual or 
                    current_time - self.last_visual_update > self.visual_update_interval):
                    self.update_visualization()
                    self.last_visual_update = current_time
                
                self.param_changed = False
                
        except Exception as e:
            print(f"处理错误: {e}")
    
    def update_display_image(self, pil_img):
        """更新显示的图像"""
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width > 10 and canvas_height > 10:
            img_width, img_height = pil_img.size
            scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            if new_width > 0 and new_height > 0:
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_img)
        
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width//2, canvas_height//2, 
                                     anchor=tk.CENTER, image=photo)
        
        self.display_photo = photo
    
    def update_visualization(self):
        """更新所有可视化图表"""
        if not MATPLOTLIB_AVAILABLE or not self.pipeline.image_loaded:
            return
        
        try:
            # 更新密度-曝光曲线
            self.update_density_curve()
            
            # 更新RGB波形图
            self.update_waveform()
            
        except Exception as e:
            print(f"更新可视化图表失败: {e}")
    
    def on_mode_changed(self):
        """处理模式改变"""
        mode = self.mode_var.get()
        self.pipeline.set_processing_mode(mode)
        self.param_changed = True
        
        if self.pipeline.image_loaded:
            self.process_and_update_display(force_visual=True)
    
    def on_resolution_changed(self, value):
        """分辨率改变"""
        if self.selected_images:
            first_id = self.selected_images[0]
            self.load_image_for_preview(first_id)
    
    def on_gamma_changed(self):
        """Gamma值改变"""
        gamma = self.display_gamma_var.get()
        self.pipeline.update_parameter('display_gamma', gamma)
        self.param_changed = True
        
        if self.pipeline.image_loaded:
            self.process_and_update_display(force_visual=True)
    
    def update_parameter(self, param_name, value):
        """更新参数"""
        self.pipeline.update_parameter(param_name, value)
        self.param_changed = True
    
    def reset_parameters(self):
        """重置所有参数"""
        self.pipeline.reset_parameters()
        
        # 更新UI变量
        for param_name, value in self.pipeline.params.items():
            var_name = f"{param_name}_var"
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                var.set(value)
        
        self.param_changed = True
        
        if self.pipeline.image_loaded:
            self.process_and_update_display(force_visual=True)
        
        self.status_var.set("参数已重置")
    
    def export_preset(self):
        """导出预设"""
        preset = self.pipeline.export_preset()
        preset['name'] = self.preset_name_var.get()
        
        # 保存为JSON文件
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=f"{preset['name']}.json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(preset, f, indent=2, ensure_ascii=False)
                
                self.status_var.set(f"预设已导出: {os.path.basename(file_path)}")
                messagebox.showinfo(self.lang.get('export_preset'), self.lang.get('export_success'))
                
                # 添加到预设列表
                self.preset_listbox.insert(tk.END, os.path.basename(file_path))
                
            except Exception as e:
                messagebox.showerror("错误", f"导出预设失败: {e}")
    
    def import_preset(self):
        """导入预设"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                
                if self.pipeline.import_preset(preset_data):
                    # 更新UI变量
                    for param_name, value in self.pipeline.params.items():
                        var_name = f"{param_name}_var"
                        if hasattr(self, var_name):
                            var = getattr(self, var_name)
                            var.set(value)
                    
                    self.param_changed = True
                    
                    if self.pipeline.image_loaded:
                        self.process_and_update_display(force_visual=True)
                    
                    preset_name = preset_data.get('name', os.path.basename(file_path))
                    self.status_var.set(f"预设已导入: {preset_name}")
                    messagebox.showinfo(self.lang.get('import_preset'), self.lang.get('import_success'))
                else:
                    messagebox.showerror("错误", "预设文件格式不正确")
                
            except Exception as e:
                messagebox.showerror("错误", f"导入预设失败: {e}")
    
    def batch_process(self):
        """批量处理选中的图像"""
        if not self.selected_images:
            messagebox.showwarning("警告", self.lang.get('no_images_warning'))
            return
        
        # 询问保存目录
        save_dir = filedialog.askdirectory(title=self.lang.get('select_save_dir'))
        if not save_dir:
            return
        
        # 创建处理线程
        def process_thread():
            total = len(self.selected_images)
            
            for i, img_id in enumerate(self.selected_images):
                try:
                    # 更新状态
                    self.root.after(0, lambda idx=i+1: 
                                  self.status_var.set(f"{self.lang.get('processing')} ({idx}/{total})"))
                    
                    # 使用完整分辨率处理
                    img_data = self.image_manager.get_image_data(img_id, scale=1.0)
                    
                    if img_data is None:
                        continue
                    
                    # 临时加载到管道
                    temp_pipeline = FilmPipeline()
                    temp_pipeline.load_image(img_data)
                    
                    # 应用当前参数
                    for param_name, value in self.pipeline.params.items():
                        temp_pipeline.update_parameter(param_name, value)
                    
                    # 处理图像
                    display_array = temp_pipeline.process_realtime(force_update=True)
                    
                    if display_array is not None:
                        # 保存图像
                        img_info = self.image_manager.images[img_id]
                        base_name = os.path.splitext(img_info['name'])[0]
                        save_path = os.path.join(save_dir, f"{base_name}_processed.png")
                        
                        pil_img = Image.fromarray(display_array)
                        pil_img.save(save_path, 'PNG')
                        
                        # 更新列表状态
                        self.root.after(0, lambda id=img_id: 
                                      self.image_tree.set(id, 'status', self.lang.get('done')))
                    
                except Exception as e:
                    print(f"处理图像 {img_id} 失败: {e}")
                    self.root.after(0, lambda id=img_id: 
                                  self.image_tree.set(id, 'status', self.lang.get('failed')))
            
            self.root.after(0, lambda: self.status_var.set(f"{self.lang.get('processing_complete')} {save_dir}"))
            self.root.after(0, lambda: messagebox.showinfo(self.lang.get('processing_complete'), 
                                                          f"{self.lang.get('processing_complete')}:\n{save_dir}"))
        
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
    
    def save_current_image(self):
        """保存当前图像"""
        if not self.selected_images:
            messagebox.showwarning("警告", self.lang.get('no_images_warning'))
            return
        
        img_id = self.selected_images[0]
        img_info = self.image_manager.images[img_id]
        
        base_name = os.path.splitext(img_info['name'])[0]
        mode = self.mode_var.get()
        default_name = f"{base_name}_{mode}_processed.png"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg;*.jpeg"),
                ("TIFF文件", "*.tiff;*.tif"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            # 使用完整分辨率处理并保存
            def save_thread():
                self.root.after(0, lambda: self.status_var.set(self.lang.get('saving')))
                
                img_data = self.image_manager.get_image_data(img_id, scale=1.0)
                
                if img_data is None:
                    self.root.after(0, lambda: self.status_var.set(self.lang.get('load_failed')))
                    return
                
                # 临时管道
                temp_pipeline = FilmPipeline()
                temp_pipeline.load_image(img_data)
                
                # 应用当前参数
                for param_name, value in self.pipeline.params.items():
                    temp_pipeline.update_parameter(param_name, value)
                
                display_array = temp_pipeline.process_realtime(force_update=True)
                
                if display_array is not None:
                    pil_img = Image.fromarray(display_array)
                    
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in ['.jpg', '.jpeg']:
                        pil_img.save(file_path, 'JPEG', quality=95)
                    elif ext in ['.tif', '.tiff']:
                        pil_img.save(file_path, 'TIFF')
                    else:
                        pil_img.save(file_path, 'PNG')
                    
                    self.root.after(0, lambda: self.status_var.set(f"{self.lang.get('save_success')}: {os.path.basename(file_path)}"))
                    self.root.after(0, lambda: messagebox.showinfo(self.lang.get('save_success'), self.lang.get('save_success')))
                else:
                    self.root.after(0, lambda: messagebox.showerror(self.lang.get('save_failed'), self.lang.get('save_failed')))
            
            thread = threading.Thread(target=save_thread, daemon=True)
            thread.start()
    
    def save_all_images(self):
        """保存所有图像（批量保存）"""
        if not self.image_manager.images:
            messagebox.showwarning("警告", self.lang.get('no_images_warning'))
            return
        
        # 使用批量处理功能
        self.selected_images = list(self.image_manager.images.keys())
        self.batch_process()
    
    def check_realtime_update(self):
        """检查并执行实时更新"""
        if self.realtime_var.get() and self.param_changed and self.pipeline.image_loaded:
            self.process_and_update_display()
        
        self.root.after(self.update_delay, self.check_realtime_update)
    
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    print("=" * 70)
    print("Aurhythm - 胶片负片处理器 v3.0")
    print("=" * 70)
    print("主要特性:")
    print("1. 专业黑色界面主题")
    print("2. 胶片密度-曝光S型曲线")
    print("3. RGB波形图")
    print("4. 多图像管理（支持批量处理）")
    print("5. 预设保存与导入")
    print("6. 中英文界面切换")
    print("7. Ctrl+滚轮缩放密度轴")
    print("8. 内存优化（按需加载图像）")
    print("=" * 70)
    
    if not MATPLOTLIB_AVAILABLE:
        print("警告: matplotlib未安装，图表功能将不可用")
        print("安装命令: pip install matplotlib scipy")
    
    if not CONFIG_AVAILABLE:
        print("提示: configparser未安装，语言设置可能无法保存")
        print("安装命令: pip install 标准库已包含")
    
    print("正在启动主界面...")
    
    app = FilmProcessorUI()
    app.run()

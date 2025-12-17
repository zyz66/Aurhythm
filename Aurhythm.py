import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import time
import numpy as np
from PIL import Image, ImageTk
import rawpy
import warnings
warnings.filterwarnings('ignore')

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
    
    def wait_for_ready(self):
        self.ready_event.wait()
        self.ready_event.clear()

class ScientificFilmPipeline:
    def __init__(self):
        self.linear_img = None
        self.image_loaded = False
        
        self.black_point = None
        self.white_point = None
        self.sampled = False
        
        self.params = {
            'r_offset': 0.0,
            'r_gain': 1.0,
            'g_offset': 0.0,
            'g_gain': 1.0,
            'b_offset': 0.0,
            'b_gain': 1.0,
            
            'cineon_black': 95.0,
            'cineon_white': 685.0,
        }
        
        self.target_space = 'cineon'
        
    def load_linear_image(self, img_array):
        if img_array is None:
            return False
            
        self.linear_img = img_array.copy()
        self.image_loaded = True
        return True
    
    def set_black_point(self, rgb_values):
        if len(rgb_values) == 3:
            self.black_point = np.array(rgb_values, dtype=np.float32)
            return True
        return False
    
    def set_white_point(self, rgb_values):
        if len(rgb_values) == 3:
            self.white_point = np.array(rgb_values, dtype=np.float32)
            return True
        return False
    
    def complete_sampling(self):
        if self.black_point is not None and self.white_point is not None:
            self.sampled = True
            return True
        return False
    
    def apply_black_white_mapping(self, linear_img):
        if linear_img is None or not self.sampled:
            return None
        
        img = linear_img.copy()
        
        img = 1.0 - img
        
        black_linear = 1.0 - self.black_point
        white_linear = 1.0 - self.white_point
        
        for i in range(3):
            channel = img[:,:,i]
            black_val = black_linear[i]
            white_val = white_linear[i]
            
            if abs(white_val - black_val) > 1e-6:
                channel = (channel - black_val) / (white_val - black_val)
            else:
                channel = channel - black_val
            
            img[:,:,i] = np.clip(channel, 0.0, 1.0)
        
        epsilon = 1e-7
        img = np.maximum(img, epsilon)
        density = -np.log10(img)
        
        cineon_range = self.params['cineon_white'] - self.params['cineon_black']
        density_range = 2.048
        
        density_normalized = np.clip(density, 0, density_range) / density_range
        cineon = self.params['cineon_white'] - (density_normalized * cineon_range)
        
        cineon_normalized = (cineon - self.params['cineon_black']) / cineon_range
        
        return np.clip(cineon_normalized, 0.0, 1.0)
    
    def apply_channel_alignment(self, cineon_img):
        if cineon_img is None:
            return None
        
        aligned = cineon_img.copy()
        
        if self.params['r_offset'] != 0 or self.params['r_gain'] != 1.0:
            aligned[:,:,0] = aligned[:,:,0] * self.params['r_gain'] + self.params['r_offset']
        
        if self.params['g_offset'] != 0 or self.params['g_gain'] != 1.0:
            aligned[:,:,1] = aligned[:,:,1] * self.params['g_gain'] + self.params['g_offset']
        
        if self.params['b_offset'] != 0 or self.params['b_gain'] != 1.0:
            aligned[:,:,2] = aligned[:,:,2] * self.params['b_gain'] + self.params['b_offset']
        
        return np.clip(aligned, 0.0, 1.0)
    
    def convert_to_target_space(self, cineon_img):
        if cineon_img is None:
            return None
        
        cineon_range = self.params['cineon_white'] - self.params['cineon_black']
        cineon_code = cineon_img * cineon_range + self.params['cineon_black']
        
        if self.target_space == 'cineon':
            return cineon_img
        
        elif self.target_space == 'logc3':
            cineon_normalized = cineon_code / 1023.0
            logc3 = cineon_normalized
            return np.clip(logc3, 0.0, 1.0)
            
        elif self.target_space == 'logc4':
            cineon_normalized = cineon_code / 1023.0
            logc4 = cineon_normalized
            return np.clip(logc4, 0.0, 1.0)
            
        elif self.target_space == 'slog3':
            cineon_normalized = cineon_code / 1023.0
            slog3 = cineon_normalized
            return np.clip(slog3, 0.0, 1.0)
            
        else:
            return cineon_img
    
    def cineon_to_display(self, cineon_img):
        if cineon_img is None:
            return None
        
        cineon_range = self.params['cineon_white'] - self.params['cineon_black']
        cineon_code = cineon_img * cineon_range + self.params['cineon_black']
        
        density_normalized = (self.params['cineon_white'] - cineon_code) / cineon_range
        density = density_normalized * 2.048
        
        epsilon = 1e-7
        density = np.maximum(density, epsilon)
        linear = np.power(10.0, -density)
        
        result = np.where(
            linear < 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1.0/2.4) - 0.055
        )
        
        return np.clip(result, 0.0, 1.0)
    
    def process_for_preview(self):
        if not self.image_loaded or self.linear_img is None or not self.sampled:
            return None
        
        try:
            cineon = self.apply_black_white_mapping(self.linear_img)
            
            if cineon is None:
                return None
            
            aligned = self.apply_channel_alignment(cineon)
            
            display_img = self.cineon_to_display(aligned)
            
            if display_img is not None:
                display_array = (display_img * 255).astype(np.uint8)
                return display_array
            
        except Exception as exc:
            print(f"预览处理错误: {exc}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def process_for_output(self):
        if not self.image_loaded or self.linear_img is None or not self.sampled:
            print(f"处理失败: image_loaded={self.image_loaded}, linear_img={self.linear_img is not None}, sampled={self.sampled}")
            return None
        
        try:
            cineon = self.apply_black_white_mapping(self.linear_img)
            
            if cineon is None:
                print("Cineon映射失败")
                return None
            
            aligned = self.apply_channel_alignment(cineon)
            
            target_img = self.convert_to_target_space(aligned)
            
            print(f"输出处理完成，形状: {target_img.shape if target_img is not None else 'None'}, "
                  f"范围: [{target_img.min() if target_img is not None else 'N/A'}, "
                  f"{target_img.max() if target_img is not None else 'N/A'}]")
            
            return target_img
            
        except Exception as exc:
            print(f"输出处理错误: {exc}")
            import traceback
            traceback.print_exc()
            return None

class ImageManager:
    def __init__(self):
        self.images = {}
        self.current_id = None
        self._next_id = 0
        
    def add_image(self, file_path, is_dng=False, color_space=None):
        img_id = self._next_id
        self._next_id += 1
        
        self.images[img_id] = {
            'path': file_path,
            'name': os.path.basename(file_path),
            'is_dng': is_dng,
            'color_space': color_space,
            'thumbnail': None,
            'metadata': {},
            'pipeline': ScientificFilmPipeline()
        }
        
        thread = threading.Thread(target=self._load_metadata, args=(img_id,), daemon=True)
        thread.start()
        
        return img_id
    
    def _load_metadata(self, img_id):
        try:
            img_info = self.images[img_id]
            
            if img_info['is_dng']:
                with rawpy.imread(img_info['path']) as raw:
                    self.images[img_id]['metadata'] = {
                        'width': raw.sizes.width,
                        'height': raw.sizes.height,
                        'is_dng': True
                    }
            else:
                with Image.open(img_info['path']) as img:
                    self.images[img_id]['metadata'] = {
                        'width': img.width,
                        'height': img.height,
                        'format': img.format,
                        'mode': img.mode,
                        'is_dng': False
                    }
                    
                    try:
                        exif = img._getexif()
                        if exif and 40961 in exif:
                            color_space = exif[40961]
                            if color_space == 1:
                                img_info['color_space'] = 'sRGB'
                            elif color_space == 2:
                                img_info['color_space'] = 'Adobe RGB'
                            else:
                                img_info['color_space'] = '未知'
                    except:
                        img_info['color_space'] = '未知'
        except Exception as e:
            print(f"加载图像元数据失败 {img_id}: {e}")
    
    def get_image_data(self, img_id, scale=0.125):
        if img_id not in self.images:
            return None
        
        img_info = self.images[img_id]
        
        try:
            if img_info['is_dng']:
                with rawpy.imread(img_info['path']) as raw:
                    rgb_linear = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
                    img_float = rgb_linear.astype(np.float32) / 65535.0
            else:
                with Image.open(img_info['path']) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img_array = np.array(img)
                    
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
            
            return img_float
            
        except Exception as e:
            print(f"加载图像数据失败 {img_id}: {e}")
            return None

class ColorPicker:
    def __init__(self, canvas, on_pick_callback=None):
        self.canvas = canvas
        self.on_pick_callback = on_pick_callback
        
        self.current_mode = 'normal'
        self.cursor_cross = None
        self.cursor_text = None
        self.image_data = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_click)
    
    def set_mode(self, mode):
        valid_modes = ['normal', 'black_point', 'white_point']
        if mode in valid_modes:
            self.current_mode = mode
            return True
        return False
    
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
            if len(self.image_data.shape) == 3:
                r = self.image_data[img_y, img_x, 0]
                g = self.image_data[img_y, img_x, 1]
                b = self.image_data[img_y, img_x, 2]
                
                rgb_text = f"R: {r:.3f}  G: {g:.3f}  B: {b:.3f}"
                
                self.update_crosshair(canvas_x, canvas_y, rgb_text)
    
    def on_mouse_click(self, event):
        if self.image_data is None or self.current_mode == 'normal':
            return
        
        canvas_x, canvas_y = event.x, event.y
        img_x, img_y = self.canvas_to_image(canvas_x, canvas_y)
        
        if img_x is not None and img_y is not None:
            if len(self.image_data.shape) == 3:
                r = self.image_data[img_y, img_x, 0]
                g = self.image_data[img_y, img_x, 1]
                b = self.image_data[img_y, img_x, 2]
                
                rgb_values = [r, g, b]
                
                if self.on_pick_callback:
                    self.on_pick_callback(self.current_mode, rgb_values)
    
    def canvas_to_image(self, canvas_x, canvas_y):
        if self.image_data is None:
            return None, None
        
        img_height, img_width = self.image_data.shape[:2]
        display_width = int(img_width * self.display_scale)
        display_height = int(img_height * self.display_scale)
        
        offset_x, offset_y = self.display_offset
        
        if (offset_x <= canvas_x < offset_x + display_width and 
            offset_y <= canvas_y < offset_y + display_height):
            
            img_x = int((canvas_x - offset_x) / self.display_scale)
            img_y = int((canvas_y - offset_y) / self.display_scale)
            
            img_x = max(0, min(img_width - 1, img_x))
            img_y = max(0, min(img_height - 1, img_y))
            
            return img_x, img_y
        
        return None, None
    
    def update_crosshair(self, x, y, text):
        if self.cursor_cross:
            self.canvas.delete(self.cursor_cross)
        if self.cursor_text:
            self.canvas.delete(self.cursor_text)
        
        cross_size = 15
        self.cursor_cross = [
            self.canvas.create_line(x-cross_size, y, x+cross_size, y, fill="white", width=1),
            self.canvas.create_line(x, y-cross_size, x, y+cross_size, fill="white", width=1)
        ]
        
        text_x = x + 20
        text_y = y - 30
        
        canvas_width = self.canvas.winfo_width()
        if text_x + 100 > canvas_width:
            text_x = x - 120
        
        self.cursor_text = self.canvas.create_text(
            text_x, text_y,
            text=text,
            fill="white",
            font=("Arial", 10),
            anchor="w"
        )
    
    def clear_crosshair(self):
        if self.cursor_cross:
            for item in self.cursor_cross:
                self.canvas.delete(item)
            self.cursor_cross = None
        
        if self.cursor_text:
            self.canvas.delete(self.cursor_text)
            self.cursor_text = None

class PrecisionSlider:
    def __init__(self, parent, label, from_val, to_val, resolution, param_name, 
                 callback=None, width=200):
        self.parent = parent
        self.label = label
        self.from_val = from_val
        self.to_val = to_val
        self.resolution = resolution
        self.param_name = param_name
        self.callback = callback
        
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(self.frame, text=label, width=15).pack(side=tk.LEFT)
        
        self.slider_frame = ttk.Frame(self.frame)
        self.slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.value_var = tk.DoubleVar(value=(from_val + to_val) / 2)
        self.slider = ttk.Scale(self.slider_frame, from_=from_val, to=to_val,
                               variable=self.value_var, orient=tk.HORIZONTAL,
                               length=width)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.display_frame = ttk.Frame(self.frame)
        self.display_frame.pack(side=tk.RIGHT)
        
        self.entry_var = tk.StringVar(value=f"{self.value_var.get():.3f}")
        self.entry = ttk.Entry(self.display_frame, textvariable=self.entry_var, width=8)
        self.entry.pack(side=tk.LEFT, padx=2)
        
        self.slider.bind('<B1-Motion>', self.on_slider_drag)
        self.slider.bind('<ButtonRelease-1>', self.on_slider_release)
        self.entry_var.trace('w', self.on_entry_change)
    
    def on_slider_drag(self, event):
        value = self.value_var.get()
        quantized = round(value / self.resolution) * self.resolution
        quantized = max(self.from_val, min(self.to_val, quantized))
        
        self.value_var.set(quantized)
        self.entry_var.set(f"{quantized:.3f}")
        
        if self.callback:
            self.callback(self.param_name, quantized)
    
    def on_slider_release(self, event):
        value = self.value_var.get()
        quantized = round(value / self.resolution) * self.resolution
        quantized = max(self.from_val, min(self.to_val, quantized))
        
        if self.callback:
            self.callback(self.param_name, quantized)
    
    def on_entry_change(self, *args):
        try:
            float_val = float(self.entry_var.get())
            if self.from_val <= float_val <= self.to_val:
                self.value_var.set(float_val)
                
                if self.callback:
                    self.callback(self.param_name, float_val)
        except ValueError:
            pass
    
    def get_value(self):
        return self.value_var.get()
    
    def set_value(self, value):
        self.value_var.set(value)
        self.entry_var.set(f"{value:.3f}")

class FilmProcessorUI:
    def __init__(self):
        self.image_manager = ImageManager()
        
        self.param_queue = ParameterQueue(maxsize=50)
        self.render_buffer = RenderingBuffer()
        
        self.render_thread = None
        self.render_running = False
        
        self.current_image_id = None
        self.current_image_data = None
        self.display_photo = None
        self.color_picker = None
        self.preview_scale = 0.125
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        
        self.rgb_channel_photos = [None, None, None]
        
        self.bg_color = '#1e1e1e'
        self.frame_bg = '#252526'
        self.text_color = '#d4d4d4'
        self.accent_color = '#007acc'
        
        self.root = tk.Tk()
        self.root.title("Aurhythm 胶片负片处理器 v1.0.0")
        self.root.geometry("1400x1000")
        self.root.configure(bg=self.bg_color)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_styles()
        self.setup_ui()
        self.start_render_thread()
        
        self.root.mainloop()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('.', 
                       background=self.bg_color,
                       foreground=self.text_color,
                       fieldbackground=self.frame_bg)
        
        style.configure('Title.TLabel', 
                       font=('Microsoft YaHei', 14, 'bold'),
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
    
    def setup_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_panel = ttk.Frame(main_paned, width=300)
        self.setup_image_panel(left_panel)
        main_paned.add(left_panel)
        
        middle_panel = ttk.Frame(main_paned)
        self.setup_preview_panel(middle_panel)
        main_paned.add(middle_panel)
        
        right_panel = ttk.Frame(main_paned, width=400)
        self.setup_parameter_panel(right_panel)
        main_paned.add(right_panel)
    
    def setup_image_panel(self, parent):
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="图像管理", style='Title.TLabel').pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(btn_frame, text="添加DNG图像", 
                  command=lambda: self.add_images(True), width=15).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text="添加TIFF图像", 
                  command=lambda: self.add_images(False), width=15).pack(side=tk.LEFT, padx=2)
        
        list_frame = ttk.LabelFrame(parent, text="图像列表", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('name', 'size', 'status')
        self.image_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', height=15)
        
        self.image_tree.heading('#0', text='', anchor=tk.W)
        self.image_tree.column('#0', width=30, stretch=False)
        
        self.image_tree.heading('name', text="文件名", anchor=tk.W)
        self.image_tree.column('name', width=150)
        
        self.image_tree.heading('size', text="尺寸", anchor=tk.W)
        self.image_tree.column('size', width=80)
        
        self.image_tree.heading('status', text="状态", anchor=tk.W)
        self.image_tree.column('status', width=60)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_tree.yview)
        self.image_tree.configure(yscrollcommand=scrollbar.set)
        
        self.image_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_tree.bind('<<TreeviewSelect>>', self.on_image_selected)
        
        self.root.bind('<Delete>', self.delete_selected_image)
        
        tip_label = ttk.Label(parent, text="点击选择图像，Delete键删除", 
                             font=('Microsoft YaHei', 8), foreground='gray')
        tip_label.pack(side=tk.BOTTOM, pady=(5, 0))
    
    def setup_preview_panel(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        picker_frame = ttk.Frame(control_frame)
        picker_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(picker_frame, text="取色工具:").pack(side=tk.LEFT)
        
        self.picker_mode_var = tk.StringVar(value='normal')
        
        ttk.Radiobutton(picker_frame, text="正常吸管",
                       variable=self.picker_mode_var, value='normal',
                       command=self.on_picker_mode_changed).pack(side=tk.LEFT, padx=2)
        
        res_frame = ttk.Frame(control_frame)
        res_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(res_frame, text="预览分辨率:").pack(side=tk.LEFT)
        
        self.resolution_var = tk.StringVar(value="12.5%")
        resolution_menu = ttk.OptionMenu(res_frame, self.resolution_var, "12.5%", 
                                        "100%", "75%", "50%", "25%", "12.5%",
                                        command=self.on_resolution_changed)
        resolution_menu.config(width=10)
        resolution_menu.pack(side=tk.LEFT, padx=5)
        
        self.image_canvas = tk.Canvas(parent, bg='black', height=500)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.color_picker = ColorPicker(self.image_canvas, self.on_color_picked)
        
        rgb_frame = ttk.LabelFrame(parent, text="RGB分量图", padding=5)
        rgb_frame.pack(fill=tk.X, pady=(10, 0))
        
        channel_frame = ttk.Frame(rgb_frame)
        channel_frame.pack(fill=tk.BOTH, expand=True)
        
        self.r_canvas = tk.Canvas(channel_frame, bg='black', height=100)
        self.g_canvas = tk.Canvas(channel_frame, bg='black', height=100)
        self.b_canvas = tk.Canvas(channel_frame, bg='black', height=100)
        
        self.r_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        self.g_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.b_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        
        label_frame = ttk.Frame(rgb_frame)
        label_frame.pack(fill=tk.X)
        
        ttk.Label(label_frame, text="R", foreground='red', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, expand=True)
        ttk.Label(label_frame, text="G", foreground='green', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, expand=True)
        ttk.Label(label_frame, text="B", foreground='blue', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, expand=True)
        
        self.image_info_label = ttk.Label(parent, text="未选择图像", relief='sunken')
        self.image_info_label.pack(fill=tk.X, pady=(10, 0))
    
    def update_rgb_channels(self, rgb_array):
        if rgb_array is None or len(rgb_array.shape) != 3:
            return
        
        height, width = rgb_array.shape[:2]
        channel_height = 100
        
        r_channel = rgb_array[:,:,0]
        g_channel = rgb_array[:,:,1]
        b_channel = rgb_array[:,:,2]
        
        scale = channel_height / height
        new_width = int(width * scale)
        
        def resize_channel(channel):
            channel_img = Image.fromarray(channel, mode='L')
            channel_img = channel_img.resize((new_width, channel_height), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(channel_img)
        
        r_photo = resize_channel(r_channel)
        g_photo = resize_channel(g_channel)
        b_photo = resize_channel(b_channel)
        
        self.r_canvas.delete("all")
        self.g_canvas.delete("all")
        self.b_canvas.delete("all")
        
        self.r_canvas.create_image(new_width//2, channel_height//2, anchor=tk.CENTER, image=r_photo)
        self.g_canvas.create_image(new_width//2, channel_height//2, anchor=tk.CENTER, image=g_photo)
        self.b_canvas.create_image(new_width//2, channel_height//2, anchor=tk.CENTER, image=b_photo)
        
        self.rgb_channel_photos = [r_photo, g_photo, b_photo]
    
    def setup_parameter_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        input_frame = ttk.Frame(notebook)
        self.setup_input_settings(input_frame)
        notebook.add(input_frame, text="输入设置")
        
        channel_frame = ttk.Frame(notebook)
        self.setup_channel_alignment(channel_frame)
        notebook.add(channel_frame, text="通道对齐")
        
        output_frame = ttk.Frame(notebook)
        self.setup_output_settings(output_frame)
        notebook.add(output_frame, text="输出设置")
    
    def setup_input_settings(self, parent):
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
        
        info_label = ttk.Label(scrollable_frame, 
                              text="首先使用黑点吸管取样片基最暗处，\n然后使用白点吸管取样最亮处。\n完成后点击'完成采样'按钮。",
                              justify=tk.LEFT)
        info_label.pack(fill=tk.X, pady=(0, 20))
        
        black_frame = ttk.LabelFrame(scrollable_frame, text="片基黑点取样", padding=10)
        black_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.black_point_var = tk.StringVar(value="未取样")
        ttk.Label(black_frame, textvariable=self.black_point_var, 
                 font=("Courier", 10)).pack(anchor=tk.W, pady=5)
        
        ttk.Button(black_frame, text="黑点吸管", 
                  command=lambda: self.set_picker_mode('black_point'), width=15).pack()
        
        white_frame = ttk.LabelFrame(scrollable_frame, text="最亮白点取样", padding=10)
        white_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.white_point_var = tk.StringVar(value="未取样")
        ttk.Label(white_frame, textvariable=self.white_point_var, 
                 font=("Courier", 10)).pack(anchor=tk.W, pady=5)
        
        ttk.Button(white_frame, text="白点吸管", 
                  command=lambda: self.set_picker_mode('white_point'), width=15).pack()
        
        complete_frame = ttk.Frame(scrollable_frame)
        complete_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.complete_button = ttk.Button(complete_frame, text="完成采样", 
                                         command=self.complete_sampling, width=20, state='disabled')
        self.complete_button.pack()
    
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
        
        red_frame = ttk.LabelFrame(scrollable_frame, text="红色通道", padding=10)
        red_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.r_offset_slider = PrecisionSlider(
            red_frame,
            "偏移:",
            -0.5, 0.5, 0.001,
            'r_offset',
            self.on_parameter_changed,
            width=150
        )
        self.r_offset_slider.set_value(0.0)
        
        self.r_gain_slider = PrecisionSlider(
            red_frame,
            "增益:",
            0.5, 2.0, 0.001,
            'r_gain',
            self.on_parameter_changed,
            width=150
        )
        self.r_gain_slider.set_value(1.0)
        
        green_frame = ttk.LabelFrame(scrollable_frame, text="绿色通道", padding=10)
        green_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.g_offset_slider = PrecisionSlider(
            green_frame,
            "偏移:",
            -0.5, 0.5, 0.001,
            'g_offset',
            self.on_parameter_changed,
            width=150
        )
        self.g_offset_slider.set_value(0.0)
        
        self.g_gain_slider = PrecisionSlider(
            green_frame,
            "增益:",
            0.5, 2.0, 0.001,
            'g_gain',
            self.on_parameter_changed,
            width=150
        )
        self.g_gain_slider.set_value(1.0)
        
        blue_frame = ttk.LabelFrame(scrollable_frame, text="蓝色通道", padding=10)
        blue_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.b_offset_slider = PrecisionSlider(
            blue_frame,
            "偏移:",
            -0.5, 0.5, 0.001,
            'b_offset',
            self.on_parameter_changed,
            width=150
        )
        self.b_offset_slider.set_value(0.0)
        
        self.b_gain_slider = PrecisionSlider(
            blue_frame,
            "增益:",
            0.5, 2.0, 0.001,
            'b_gain',
            self.on_parameter_changed,
            width=150
        )
        self.b_gain_slider.set_value(1.0)
        
        reset_frame = ttk.Frame(scrollable_frame)
        reset_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(reset_frame, text="重置所有通道", 
                  command=self.reset_channel_params, width=20).pack(side=tk.LEFT, padx=2)
    
    def setup_output_settings(self, parent):
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
        
        target_frame = ttk.LabelFrame(scrollable_frame, text="目标对数空间", padding=10)
        target_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.target_space_var = tk.StringVar(value='cineon')
        
        ttk.Radiobutton(target_frame, text="保持Cineon", variable=self.target_space_var,
                       value='cineon', command=self.on_target_space_changed).pack(anchor=tk.W)
        
        ttk.Radiobutton(target_frame, text="ARRI LogC3", variable=self.target_space_var,
                       value='logc3', command=self.on_target_space_changed).pack(anchor=tk.W)
        
        ttk.Radiobutton(target_frame, text="ARRI LogC4", variable=self.target_space_var,
                       value='logc4', command=self.on_target_space_changed).pack(anchor=tk.W)
        
        ttk.Radiobutton(target_frame, text="Sony S-Log3", variable=self.target_space_var,
                       value='slog3', command=self.on_target_space_changed).pack(anchor=tk.W)
        
        export_frame = ttk.Frame(scrollable_frame)
        export_frame.pack(fill=tk.X, pady=(20, 10))
        
        ttk.Button(export_frame, text="导出当前图像", 
                  command=self.export_current_image, width=20).pack(side=tk.LEFT, padx=2)
    
    def start_render_thread(self):
        self.render_running = True
        
        def render_worker():
            while self.render_running:
                try:
                    params = self.param_queue.get_latest()
                    
                    if self.current_image_id is not None and params:
                        img_info = self.image_manager.images[self.current_image_id]
                        pipeline = img_info['pipeline']
                        
                        if not pipeline.sampled:
                            time.sleep(0.01)
                            continue
                        
                        for param_name, value in params.items():
                            if param_name in pipeline.params:
                                pipeline.params[param_name] = value
                        
                        display_array = pipeline.process_for_preview()
                        
                        if display_array is not None:
                            if display_array.ndim == 2:
                                pil_img = Image.fromarray(display_array, mode='L')
                            else:
                                pil_img = Image.fromarray(display_array, mode='RGB')
                            
                            self.render_buffer.update_back_buffer(pil_img)
                            self.render_buffer.swap_buffers()
                    
                    time.sleep(0.01)
                    
                except Exception as exc:
                    print(f"渲染线程错误: {exc}")
        
        self.render_thread = threading.Thread(target=render_worker, daemon=True)
        self.render_thread.start()
        
        self.start_update_timer()
    
    def start_update_timer(self):
        def update_display():
            buffer_data = self.render_buffer.get_front_buffer()
            if buffer_data is not None:
                self.update_display_image(buffer_data)
            
            self.root.after(33, update_display)
        
        self.root.after(33, update_display)
    
    def set_picker_mode(self, mode):
        self.picker_mode_var.set(mode)
        self.color_picker.set_mode(mode)
    
    def on_picker_mode_changed(self):
        mode = self.picker_mode_var.get()
        self.color_picker.set_mode(mode)
    
    def on_color_picked(self, mode, rgb_values):
        if self.current_image_id is None:
            return
        
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        
        if mode == 'black_point':
            pipeline.set_black_point(rgb_values)
            self.black_point_var.set(f"R: {rgb_values[0]:.3f}  G: {rgb_values[1]:.3f}  B: {rgb_values[2]:.3f}")
        elif mode == 'white_point':
            pipeline.set_white_point(rgb_values)
            self.white_point_var.set(f"R: {rgb_values[0]:.3f}  G: {rgb_values[1]:.3f}  B: {rgb_values[2]:.3f}")
        
        if pipeline.black_point is not None and pipeline.white_point is not None:
            self.complete_button.config(state='normal')
    
    def complete_sampling(self):
        if self.current_image_id is None:
            return
        
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        
        if pipeline.complete_sampling():
            self.picker_mode_var.set('normal')
            self.color_picker.set_mode('normal')
            
            self.complete_button.config(state='disabled')
            
            params = self.get_current_params()
            self.param_queue.put(params)
            
            messagebox.showinfo("采样完成", "黑点白点采样已完成，可以开始通道对齐调整。")
    
    def on_parameter_changed(self, param_name, value):
        params = self.param_queue.get_latest() or {}
        params[param_name] = value
        self.param_queue.put(params)
    
    def on_target_space_changed(self):
        if self.current_image_id is not None:
            pipeline = self.image_manager.images[self.current_image_id]['pipeline']
            pipeline.target_space = self.target_space_var.get()
    
    def on_resolution_changed(self, value):
        resolution_map = {
            "100%": 1.0,
            "75%": 0.75,
            "50%": 0.5,
            "25%": 0.25,
            "12.5%": 0.125
        }
        
        if value in resolution_map:
            self.preview_scale = resolution_map[value]
            if self.current_image_id is not None:
                self.load_image_for_preview(self.current_image_id)
    
    def add_images(self, is_dng=False):
        if is_dng:
            filetypes = [("DNG文件", "*.dng"), ("所有文件", "*.*")]
        else:
            filetypes = [("TIFF文件", "*.tif;*.tiff"), ("所有文件", "*.*")]
        
        file_paths = filedialog.askopenfilenames(
            title="选择DNG文件" if is_dng else "选择TIFF文件",
            filetypes=filetypes
        )
        
        if file_paths:
            for file_path in file_paths:
                if not is_dng:
                    color_space = self.check_tiff_color_space(file_path)
                    if color_space not in [None, 'RGB', 'sRGB', 'Adobe RGB']:
                        messagebox.showerror("错误", f"不支持的色彩空间: {color_space}\n请使用RGB或Adobe RGB的线性TIFF")
                        continue
                
                img_id = self.image_manager.add_image(file_path, is_dng, color_space if not is_dng else None)
                
                self.image_tree.insert('', 'end', iid=img_id, 
                                      values=(os.path.basename(file_path), "加载中", "待处理"))
            
            if self.current_image_id is None and file_paths:
                self.image_tree.selection_set(img_id)
                self.on_image_selected()
    
    def check_tiff_color_space(self, file_path):
        try:
            with Image.open(file_path) as img:
                if img.mode not in ['RGB', 'RGBA', 'L', 'LA']:
                    return img.mode
                
                try:
                    exif = img._getexif()
                    if exif:
                        if 40961 in exif:
                            color_space = exif[40961]
                            if color_space == 1:
                                return 'sRGB'
                            elif color_space == 2:
                                return 'Adobe RGB'
                            else:
                                return f"未知 ({color_space})"
                except:
                    pass
                
                return None
        except Exception as e:
            return f"错误: {str(e)}"
    
    def delete_selected_image(self, event=None):
        selected_ids = self.image_tree.selection()
        if not selected_ids:
            return
        
        response = messagebox.askyesno("确认删除", f"确定要删除选中的 {len(selected_ids)} 张图像吗？")
        if response:
            for img_id in selected_ids:
                img_id_int = int(img_id)
                
                if img_id_int == self.current_image_id:
                    self.current_image_id = None
                    self.clear_preview()
                
                if img_id_int in self.image_manager.images:
                    del self.image_manager.images[img_id_int]
                
                self.image_tree.delete(img_id)
    
    def on_image_selected(self, event=None):
        selected_ids = self.image_tree.selection()
        
        if not selected_ids:
            self.current_image_id = None
            self.clear_preview()
            return
        
        img_id = int(selected_ids[0])
        
        if img_id == self.current_image_id:
            return
        
        self.current_image_id = img_id
        
        self.black_point_var.set("未取样")
        self.white_point_var.set("未取样")
        self.complete_button.config(state='disabled')
        
        self.picker_mode_var.set('normal')
        self.color_picker.set_mode('normal')
        
        self.load_image_for_preview(img_id)
    
    def load_image_for_preview(self, img_id):
        if img_id not in self.image_manager.images:
            return
        
        def load_thread():
            img_data = self.image_manager.get_image_data(img_id, self.preview_scale)
            
            if img_data is not None:
                self.root.after(0, lambda: self.on_image_loaded(img_id, img_data))
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def on_image_loaded(self, img_id, img_data):
        if img_id != self.current_image_id:
            return
        
        img_info = self.image_manager.images[img_id]
        
        metadata = img_info.get('metadata', {})
        size_str = f"{metadata.get('width', 0)}x{metadata.get('height', 0)}" if metadata else "已加载"
        self.image_tree.item(img_id, values=(img_info['name'], size_str, "已加载"))
        
        self.current_image_data = img_data
        
        pipeline = img_info['pipeline']
        pipeline.load_linear_image(img_data)
        
        pipeline.sampled = False
        pipeline.black_point = None
        pipeline.white_point = None
        
        self.color_picker.clear_crosshair()
        
        self.color_picker.update_image_info(img_data, self.display_scale, self.display_offset)
        
        self.image_info_label.config(
            text=f"{img_info['name']} - {img_data.shape[1]}x{img_data.shape[0]}"
        )
        
        while self.param_queue.get() is not None:
            pass
        
        self.display_raw_image(img_data)
    
    def display_raw_image(self, img_data):
        if img_data is None:
            return
        
        if img_data.dtype == np.float32:
            img_8bit = (img_data * 255).astype(np.uint8)
        else:
            img_8bit = img_data
        
        if img_8bit.ndim == 2:
            pil_img = Image.fromarray(img_8bit, mode='L')
        else:
            pil_img = Image.fromarray(img_8bit, mode='RGB')
        
        self.update_display_image(pil_img)
        
        if img_8bit.ndim == 3:
            self.update_rgb_channels(img_8bit)
    
    def update_display_image(self, pil_img):
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width > 10 and canvas_height > 10:
            img_width, img_height = pil_img.size
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y) * 0.95
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            if new_width > 0 and new_height > 0:
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                self.display_scale = scale
                self.display_offset = ((canvas_width - new_width) // 2, (canvas_height - new_height) // 2)
        
        photo = ImageTk.PhotoImage(pil_img)
        
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width//2, canvas_height//2, 
                                     anchor=tk.CENTER, image=photo)
        
        self.display_photo = photo
        
        if self.current_image_data is not None:
            self.color_picker.update_image_info(self.current_image_data, self.display_scale, self.display_offset)
        
        if pil_img.mode == 'RGB':
            rgb_array = np.array(pil_img)
            self.update_rgb_channels(rgb_array)
    
    def get_current_params(self):
        params = {
            'r_offset': self.r_offset_slider.get_value(),
            'r_gain': self.r_gain_slider.get_value(),
            'g_offset': self.g_offset_slider.get_value(),
            'g_gain': self.g_gain_slider.get_value(),
            'b_offset': self.b_offset_slider.get_value(),
            'b_gain': self.b_gain_slider.get_value(),
        }
        return params
    
    def reset_channel_params(self):
        self.r_offset_slider.set_value(0.0)
        self.r_gain_slider.set_value(1.0)
        self.g_offset_slider.set_value(0.0)
        self.g_gain_slider.set_value(1.0)
        self.b_offset_slider.set_value(0.0)
        self.b_gain_slider.set_value(1.0)
        
        params = self.get_current_params()
        self.param_queue.put(params)
    
    def export_current_image(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "没有选择图像")
            return
        
        img_info = self.image_manager.images[self.current_image_id]
        pipeline = img_info['pipeline']
        
        if not pipeline.sampled:
            messagebox.showwarning("警告", "请先完成黑点白点采样")
            return
        
        base_name = os.path.splitext(img_info['name'])[0]
        default_name = f"{base_name}_processed.tif"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            initialfile=default_name,
            filetypes=[("TIFF文件", "*.tif;*.tiff"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
        
        def export_thread():
            try:
                print("开始导出处理...")
                
                img_data = self.image_manager.get_image_data(self.current_image_id, scale=1.0)
                
                if img_data is None:
                    self.root.after(0, lambda: messagebox.showerror("错误", "加载图像失败"))
                    return
                
                print(f"加载图像成功，形状: {img_data.shape}")
                
                export_pipeline = ScientificFilmPipeline()
                
                export_pipeline.black_point = pipeline.black_point.copy() if pipeline.black_point is not None else None
                export_pipeline.white_point = pipeline.white_point.copy() if pipeline.white_point is not None else None
                export_pipeline.sampled = pipeline.sampled
                export_pipeline.params = pipeline.params.copy()
                export_pipeline.target_space = pipeline.target_space
                
                print(f"复制状态: sampled={export_pipeline.sampled}, "
                      f"black_point={export_pipeline.black_point is not None}, "
                      f"white_point={export_pipeline.white_point is not None}")
                
                if not export_pipeline.load_linear_image(img_data):
                    self.root.after(0, lambda: messagebox.showerror("错误", "加载图像到管道失败"))
                    return
                
                print("图像加载到管道成功")
                
                export_pipeline.sampled = True
                
                output_img = export_pipeline.process_for_output()
                
                if output_img is not None:
                    print(f"输出图像处理成功，形状: {output_img.shape}, 范围: [{output_img.min()}, {output_img.max()}]")
                    
                    output_float32 = output_img.astype(np.float32)
                    
                    if output_float32.ndim != 3 or output_float32.shape[2] != 3:
                        print(f"错误的数据形状: {output_float32.shape}")
                        self.root.after(0, lambda: messagebox.showerror("错误", f"错误的数据形状: {output_float32.shape}"))
                        return
                    
                    try:
                        import tifffile
                        
                        tifffile.imwrite(file_path, output_float32, photometric='rgb')
                        
                        print("图像保存成功")
                        self.root.after(0, lambda: messagebox.showinfo("成功", f"32位浮点TIFF已导出: {os.path.basename(file_path)}"))
                        
                    except ImportError:
                        try:
                            import imageio
                            
                            imageio.imwrite(file_path, output_float32, format='TIFF')
                            
                            print("图像保存成功（使用imageio）")
                            self.root.after(0, lambda: messagebox.showinfo("成功", f"32位浮点TIFF已导出: {os.path.basename(file_path)}"))
                            
                        except ImportError:
                            self.root.after(0, lambda: messagebox.showerror("错误", 
                                "需要安装tifffile或imageio库来保存32位浮点TIFF\n"
                                "请安装: pip install tifffile 或 pip install imageio"))
                            
                else:
                    print("输出图像处理失败")
                    self.root.after(0, lambda: messagebox.showerror("错误", "处理图像失败"))
                    
            except Exception as exc:
                print(f"导出失败: {exc}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("错误", f"导出失败: {str(exc)}"))
            
        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()
    
    def clear_preview(self):
        self.image_canvas.delete("all")
        self.color_picker.clear_crosshair()
        self.image_info_label.config(text="未选择图像")
        self.current_image_data = None
        
        self.r_canvas.delete("all")
        self.g_canvas.delete("all")
        self.b_canvas.delete("all")
        
        self.black_point_var.set("未取样")
        self.white_point_var.set("未取样")
        self.complete_button.config(state='disabled')
    
    def on_closing(self):
        self.render_running = False
        self.root.destroy()

if __name__ == '__main__':
    print("=" * 70)
    print("Aurhythm 胶片负片处理器 v1.0.0")
    print("=" * 70)
    print("早期参考了namicolor的原理")
    print("功能特性:")
    print("1. 支持DNG和线性TIFF输入")
    print("2. TIFF文件自动检查色彩空间")
    print("3. 三阶段工作流程：输入设置 → 通道对齐 → 输出设置")
    print("4. 取色吸管工具：正常吸管、黑点吸管、白点吸管")
    print("5. RGB分量图显示")
    print("6. 通道对齐：每个通道的偏移和增益（精度0.001）")
    print("7. 生产者-消费者模式保证UI响应")
    print("8. 输出选项：Cineon、LogC3、LogC4、S-Log3")
    print("9. 导出32位浮点TIFF")
    print("=" * 70)
    print("使用说明:")
    print("1. 添加图像文件")
    print("2. 在输入设置页面取样黑点和白点")
    print("3. 点击'完成采样'")
    print("4. 在通道对齐页面调整参数")
    print("5. 在输出设置页面选择目标空间并导出")
    print("=" * 70)
    
    try:
        import rawpy
        print("✓ rawpy已安装，支持DNG文件")
    except ImportError:
        print("✗ rawpy未安装，DNG支持不可用")
        print("请安装: pip install rawpy")
    
    try:
        import tifffile
        print("✓ tifffile已安装，支持32位浮点TIFF导出")
    except ImportError:
        try:
            import imageio
            print("✓ imageio已安装，支持TIFF导出")
        except ImportError:
            print("⚠ 建议安装tifffile或imageio库以获得更好的TIFF导出支持")
            print("请安装: pip install tifffile 或 pip install imageio")
    
    app = FilmProcessorUI()

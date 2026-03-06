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
        self.linear_img = None          # 原始线性数据 (已减黑电平)
        self.image_loaded = False
        self.base_val_rgb = None         # 片基线性值 [R, G, B]
        self.channel_gains = [1.0, 1.0, 1.0]   # 红绿蓝增益
        self.target_space = 'cineon'     # 保留，但实际只输出Cineon
        
    def load_linear_image(self, img_array):
        """加载线性图像数据 (浮点型0~1)"""
        if img_array is None:
            return False
        self.linear_img = img_array.copy().astype(np.float32)
        self.image_loaded = True
        return True
    
    def set_base_val(self, rgb_values):
        """设置片基采样值 (已减黑电平的线性值)"""
        self.base_val_rgb = np.array(rgb_values, dtype=np.float32)
    
    def set_channel_gains(self, gains):
        """设置红绿蓝增益 (用于模拟光源调节)"""
        self.channel_gains = gains
    
    def _linear_to_cineon_norm(self, linear):
        """
        线性值 -> 归一化Cineon (0~1)
        CineonCode = 95 + 500 * log10(base / L)
        结果归一化到0~1 (除以1023)
        """
        if self.base_val_rgb is None:
            return linear  # 无法转换，返回原值
        base = self.base_val_rgb.reshape(1,1,3)
        L = np.maximum(linear, 1e-6)
        cineon_code = 95.0 + 500.0 * np.log10(base / L)
        cineon_code = np.clip(cineon_code, 0.0, 1023.0)
        return cineon_code / 1023.0
    
    def process_for_preview(self):
        """生成预览图像 (8位RGB，正像)"""
        if not self.image_loaded or self.base_val_rgb is None:
            return None
        # 应用通道增益 (模拟物理光源调节)
        linear_gained = self.linear_img * np.array(self.channel_gains).reshape(1,1,3)
        # 转为Cineon归一化
        cineon_norm = self._linear_to_cineon_norm(linear_gained)
        # 反相并应用sRGB伽马 (粗略预览)
        inv = 1.0 - cineon_norm
        # sRGB近似
        preview = np.where(inv <= 0.0031308,
                          inv * 12.92,
                          1.055 * (inv ** (1/2.4)) - 0.055)
        preview = np.clip(preview, 0, 1)
        return (preview * 255).astype(np.uint8)
    
    def process_for_output(self):
        """输出归一化Cineon图像 (0~1) 用于保存"""
        if not self.image_loaded or self.base_val_rgb is None:
            return None
        linear_gained = self.linear_img * np.array(self.channel_gains).reshape(1,1,3)
        cineon_norm = self._linear_to_cineon_norm(linear_gained)
        return cineon_norm  # 直接返回0~1的Cineon值

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
            with rawpy.imread(img_info['path']) as raw:
                self.images[img_id]['metadata'] = {
                    'width': raw.sizes.width,
                    'height': raw.sizes.height,
                }
        except Exception as e:
            print(f"加载图像元数据失败 {img_id}: {e}")
            self.images[img_id]['metadata'] = {'error': str(e)}
    
    def get_image_data(self, img_id, scale=0.125):
        """返回线性浮点图像 (0~1)，已减黑电平"""
        if img_id not in self.images:
            return None
        
        img_info = self.images[img_id]
        
        try:
            with rawpy.imread(img_info['path']) as raw:
                # 输出线性16位数据 (gamma=1, 无自动亮度)
                rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True,
                                      output_bps=16, use_camera_wb=False,
                                      output_color=rawpy.ColorSpace.raw)
                # 转换为浮点0~1
                img_float = rgb.astype(np.float32) / 65535.0
                
                # rawpy已经减去了黑电平，所以这里直接使用
                
                if len(img_float.shape) == 2:
                    img_float = np.stack([img_float]*3, axis=2)
                elif img_float.shape[2] == 4:
                    img_float = img_float[:,:,:3]
                
                # 缩放预览
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
    def __init__(self, canvas, on_pick_callback=None, on_move_callback=None):
        self.canvas = canvas
        self.on_pick_callback = on_pick_callback
        self.on_move_callback = on_move_callback
        
        self.current_mode = 'normal'   # 始终为normal，但保留以备扩展
        self.cursor_cross = None
        self.cursor_text = None
        self.image_data = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_click)
    
    def set_mode(self, mode):
        self.current_mode = mode
    
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
                
                # 调用移动回调传递线性值
                if self.on_move_callback:
                    self.on_move_callback([r, g, b])
    
    def on_mouse_click(self, event):
        if self.image_data is None:
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
                    self.on_pick_callback(rgb_values)
    
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
            img_x = max(0, min(img_width-1, img_x))
            img_y = max(0, min(img_height-1, img_y))
            return img_x, img_y
        return None, None
    
    def update_crosshair(self, x, y, text):
        if self.cursor_cross:
            for item in self.cursor_cross:
                self.canvas.delete(item)
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
        
        self.bg_color = '#1e1e1e'
        self.frame_bg = '#252526'
        self.text_color = '#d4d4d4'
        self.accent_color = '#007acc'
        
        self.root = tk.Tk()
        self.root.title("Aurhythm 胶片Cineon校准器 v1.0.0")
        self.root.geometry("1400x900")
        self.root.configure(bg=self.bg_color)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_styles()
        self.setup_ui()
        self.start_render_thread()
        self.root.mainloop()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=self.bg_color, foreground=self.text_color,
                       fieldbackground=self.frame_bg)
        style.configure('Title.TLabel', font=('Microsoft YaHei', 14, 'bold'),
                       foreground=self.text_color)
        style.configure('Heading.TLabel', font=('Microsoft YaHei', 11, 'bold'),
                       foreground=self.text_color)
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabelframe', background=self.frame_bg,
                       foreground=self.text_color, relief='flat', borderwidth=1)
        style.configure('TLabelframe.Label', background=self.frame_bg,
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
        
        right_panel = ttk.Frame(main_paned, width=350)
        self.setup_parameter_panel(right_panel)
        main_paned.add(right_panel)
    
    def setup_image_panel(self, parent):
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Label(title_frame, text="图像管理", style='Title.TLabel').pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Button(btn_frame, text="添加RAW图像",
                  command=self.add_raw_images, width=15).pack(side=tk.LEFT, padx=2)
        
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
        tip_label.pack(side=tk.BOTTOM, pady=(5,0))
    
    def setup_preview_panel(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(control_frame, text="预览分辨率:").pack(side=tk.LEFT)
        self.resolution_var = tk.StringVar(value="12.5%")
        resolution_menu = ttk.OptionMenu(control_frame, self.resolution_var, "12.5%",
                                        "100%", "75%", "50%", "25%", "12.5%",
                                        command=self.on_resolution_changed)
        resolution_menu.config(width=10)
        resolution_menu.pack(side=tk.LEFT, padx=5)
        
        self.image_canvas = tk.Canvas(parent, bg='black', height=500)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.cursor_info_label = ttk.Label(parent, text="", relief='sunken')
        self.cursor_info_label.pack(fill=tk.X, pady=(5,0))
        
        self.color_picker = ColorPicker(self.image_canvas,
                                        on_pick_callback=self.on_color_picked,
                                        on_move_callback=self.on_mouse_move)
        
        self.image_info_label = ttk.Label(parent, text="未选择图像", relief='sunken')
        self.image_info_label.pack(fill=tk.X, pady=(10,0))
    
    def setup_parameter_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=(0,5))
        
        calib_frame = ttk.Frame(notebook)
        self.setup_calibration_tab(calib_frame)
        notebook.add(calib_frame, text="Cineon校准")
        
        output_frame = ttk.Frame(notebook)
        self.setup_output_tab(output_frame)
        notebook.add(output_frame, text="输出设置")
    
    def setup_calibration_tab(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0, bg=self.frame_bg)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(5,0))
        scrollbar.pack(side="right", fill="y")
        
        info_label = ttk.Label(scrollable_frame,
                              text="1. 点击'片基采样'按钮，然后在图像片基区域点击取样。\n"
                                   "2. 调整通道增益使片基区域三通道Cineon代码接近95。\n"
                                   "3. 观察实时信息确认校准。",
                              justify=tk.LEFT)
        info_label.pack(fill=tk.X, pady=(0,20))
        
        base_frame = ttk.LabelFrame(scrollable_frame, text="片基采样", padding=10)
        base_frame.pack(fill=tk.X, pady=(0,10))
        
        self.base_point_var = tk.StringVar(value="未采样")
        ttk.Label(base_frame, textvariable=self.base_point_var,
                 font=("Courier",10)).pack(anchor=tk.W, pady=5)
        ttk.Button(base_frame, text="片基采样",
                  command=self.activate_base_sampler, width=15).pack()
        
        gain_frame = ttk.LabelFrame(scrollable_frame, text="通道增益 (模拟光源调节)", padding=10)
        gain_frame.pack(fill=tk.X, pady=(0,10))
        
        self.r_gain_slider = PrecisionSlider(gain_frame, "红增益:", 0.5, 2.0, 0.001,
                                             'r_gain', self.on_parameter_changed, width=150)
        self.r_gain_slider.set_value(1.0)
        self.g_gain_slider = PrecisionSlider(gain_frame, "绿增益:", 0.5, 2.0, 0.001,
                                             'g_gain', self.on_parameter_changed, width=150)
        self.g_gain_slider.set_value(1.0)
        self.b_gain_slider = PrecisionSlider(gain_frame, "蓝增益:", 0.5, 2.0, 0.001,
                                             'b_gain', self.on_parameter_changed, width=150)
        self.b_gain_slider.set_value(1.0)
        
        reset_frame = ttk.Frame(scrollable_frame)
        reset_frame.pack(fill=tk.X, pady=(10,0))
        ttk.Button(reset_frame, text="重置增益", command=self.reset_gains, width=15).pack()
    
    def setup_output_tab(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0, bg=self.frame_bg)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(5,0))
        scrollbar.pack(side="right", fill="y")
        
        target_frame = ttk.LabelFrame(scrollable_frame, text="目标对数空间", padding=10)
        target_frame.pack(fill=tk.X, pady=(0,10))
        self.target_space_var = tk.StringVar(value='cineon')
        ttk.Radiobutton(target_frame, text="Cineon (标准)", variable=self.target_space_var,
                       value='cineon', command=self.on_target_space_changed).pack(anchor=tk.W)
        ttk.Radiobutton(target_frame, text="ARRI LogC3 (保留)", variable=self.target_space_var,
                       value='logc3', command=self.on_target_space_changed).pack(anchor=tk.W)
        ttk.Radiobutton(target_frame, text="Sony S-Log3 (保留)", variable=self.target_space_var,
                       value='slog3', command=self.on_target_space_changed).pack(anchor=tk.W)
        
        export_frame = ttk.Frame(scrollable_frame)
        export_frame.pack(fill=tk.X, pady=(20,10))
        ttk.Button(export_frame, text="导出32位浮点TIFF",
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
                        if pipeline.base_val_rgb is None:
                            time.sleep(0.01)
                            continue
                        # 更新通道增益
                        gains = [params.get('r_gain',1.0),
                                 params.get('g_gain',1.0),
                                 params.get('b_gain',1.0)]
                        pipeline.set_channel_gains(gains)
                        
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
        self.root.after(33, self.update_display)
    
    def update_display(self):
        buffer_data = self.render_buffer.get_front_buffer()
        if buffer_data is not None:
            self.display_image(buffer_data)
        self.root.after(33, self.update_display)
    
    def display_image(self, pil_img):
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
                self.display_offset = ((canvas_width - new_width)//2, (canvas_height - new_height)//2)
        
        photo = ImageTk.PhotoImage(pil_img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width//2, canvas_height//2,
                                     anchor=tk.CENTER, image=photo)
        self.display_photo = photo
        
        if self.current_image_data is not None:
            self.color_picker.update_image_info(self.current_image_data,
                                               self.display_scale, self.display_offset)
    
    def add_raw_images(self):
        filetypes = [("RAW图像", "*.nef *.dng *.cr2 *.arw *.raf *.orf *.rw2"), ("所有文件", "*.*")]
        file_paths = filedialog.askopenfilenames(title="选择RAW文件", filetypes=filetypes)
        if not file_paths:
            return
        for file_path in file_paths:
            img_id = self.image_manager.add_image(file_path)
            self.image_tree.insert('', 'end', iid=img_id,
                                  values=(os.path.basename(file_path), "加载中", "待处理"))
        if self.current_image_id is None and file_paths:
            self.image_tree.selection_set(img_id)
            self.on_image_selected()
    
    def delete_selected_image(self, event=None):
        selected = self.image_tree.selection()
        if not selected:
            return
        if messagebox.askyesno("确认", f"删除选中的 {len(selected)} 张图像？"):
            for img_id in selected:
                img_id_int = int(img_id)
                if img_id_int == self.current_image_id:
                    self.current_image_id = None
                    self.clear_preview()
                if img_id_int in self.image_manager.images:
                    del self.image_manager.images[img_id_int]
                self.image_tree.delete(img_id)
    
    def on_image_selected(self, event=None):
        selected = self.image_tree.selection()
        if not selected:
            self.current_image_id = None
            self.clear_preview()
            return
        img_id = int(selected[0])
        if img_id == self.current_image_id:
            return
        self.current_image_id = img_id
        self.base_point_var.set("未采样")
        self.reset_gains()
        self.load_image_for_preview(img_id)
    
    def load_image_for_preview(self, img_id):
        def load_thread():
            img_data = self.image_manager.get_image_data(img_id, self.preview_scale)
            if img_data is not None:
                self.root.after(0, lambda: self.on_image_loaded(img_id, img_data))
        threading.Thread(target=load_thread, daemon=True).start()
    
    def on_image_loaded(self, img_id, img_data):
        if img_id != self.current_image_id:
            return
        img_info = self.image_manager.images[img_id]
        metadata = img_info.get('metadata', {})
        size_str = f"{metadata.get('width',0)}x{metadata.get('height',0)}" if metadata else "已加载"
        self.image_tree.item(img_id, values=(img_info['name'], size_str, "已加载"))
        self.current_image_data = img_data
        pipeline = img_info['pipeline']
        pipeline.load_linear_image(img_data)
        self.color_picker.clear_crosshair()
        self.color_picker.update_image_info(img_data, self.display_scale, self.display_offset)
        self.image_info_label.config(text=f"{img_info['name']} - {img_data.shape[1]}x{img_data.shape[0]}")
        while self.param_queue.get() is not None:
            pass
        self.display_raw_image(img_data)
    
    def display_raw_image(self, img_data):
        if img_data is None:
            return
        img_8bit = (img_data * 255).astype(np.uint8)
        if img_8bit.ndim == 2:
            pil_img = Image.fromarray(img_8bit, mode='L')
        else:
            pil_img = Image.fromarray(img_8bit, mode='RGB')
        self.display_image(pil_img)
    
    def activate_base_sampler(self):
        self.color_picker.set_mode('base_sampler')
    
    def on_color_picked(self, rgb_values):
        if self.current_image_id is None:
            return
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        pipeline.set_base_val(rgb_values)
        self.base_point_var.set(f"R: {rgb_values[0]:.4f}  G: {rgb_values[1]:.4f}  B: {rgb_values[2]:.4f}")
        # 立即触发预览更新
        params = self.get_current_params()
        self.param_queue.put(params)
    
    def on_mouse_move(self, rgb_values):
        if self.current_image_id is None:
            return
        pipeline = self.image_manager.images[self.current_image_id]['pipeline']
        if pipeline.base_val_rgb is not None:
            # 计算当前像素的Cineon代码
            base = pipeline.base_val_rgb
            gains = pipeline.channel_gains
            linear_gained = [rgb_values[i] * gains[i] for i in range(3)]
            codes = []
            for i in range(3):
                L = max(linear_gained[i], 1e-6)
                code = 95 + 500 * np.log10(base[i] / L)
                codes.append(code)
            self.cursor_info_label.config(
                text=f"线性(增益后): R={linear_gained[0]:.4f} G={linear_gained[1]:.4f} B={linear_gained[2]:.4f} | "
                     f"Cineon代码: R={codes[0]:.1f} G={codes[1]:.1f} B={codes[2]:.1f}"
            )
        else:
            self.cursor_info_label.config(
                text=f"线性原始: R={rgb_values[0]:.4f} G={rgb_values[1]:.4f} B={rgb_values[2]:.4f}"
            )
    
    def on_parameter_changed(self, param_name, value):
        params = self.param_queue.get_latest() or {}
        params[param_name] = value
        self.param_queue.put(params)
    
    def get_current_params(self):
        return {
            'r_gain': self.r_gain_slider.get_value(),
            'g_gain': self.g_gain_slider.get_value(),
            'b_gain': self.b_gain_slider.get_value(),
        }
    
    def reset_gains(self):
        self.r_gain_slider.set_value(1.0)
        self.g_gain_slider.set_value(1.0)
        self.b_gain_slider.set_value(1.0)
        params = self.get_current_params()
        self.param_queue.put(params)
    
    def on_resolution_changed(self, value):
        res_map = {"100%":1.0, "75%":0.75, "50%":0.5, "25%":0.25, "12.5%":0.125}
        if value in res_map:
            self.preview_scale = res_map[value]
            if self.current_image_id is not None:
                self.load_image_for_preview(self.current_image_id)
    
    def on_target_space_changed(self):
        if self.current_image_id is not None:
            pipeline = self.image_manager.images[self.current_image_id]['pipeline']
            pipeline.target_space = self.target_space_var.get()
    
    def export_current_image(self):
        if self.current_image_id is None:
            messagebox.showwarning("警告", "没有选择图像")
            return
        img_info = self.image_manager.images[self.current_image_id]
        pipeline = img_info['pipeline']
        if pipeline.base_val_rgb is None:
            messagebox.showwarning("警告", "请先进行片基采样")
            return
        
        base_name = os.path.splitext(img_info['name'])[0]
        default_name = f"{base_name}_cineon.tif"
        file_path = filedialog.asksaveasfilename(defaultextension=".tif",
                                                initialfile=default_name,
                                                filetypes=[("TIFF文件", "*.tif"), ("所有文件", "*.*")])
        if not file_path:
            return
        
        def export_thread():
            try:
                img_data = self.image_manager.get_image_data(self.current_image_id, scale=1.0)
                if img_data is None:
                    self.root.after(0, lambda: messagebox.showerror("错误", "加载图像失败"))
                    return
                export_pipeline = ScientificFilmPipeline()
                export_pipeline.load_linear_image(img_data)
                export_pipeline.base_val_rgb = pipeline.base_val_rgb.copy()
                export_pipeline.channel_gains = pipeline.channel_gains.copy()
                export_pipeline.target_space = pipeline.target_space
                
                output = export_pipeline.process_for_output()
                if output is None:
                    self.root.after(0, lambda: messagebox.showerror("错误", "处理图像失败"))
                    return
                
                # 保存为32位浮点TIFF
                try:
                    import tifffile
                    tifffile.imwrite(file_path, output.astype(np.float32), photometric='rgb')
                    self.root.after(0, lambda: messagebox.showinfo("成功", f"已导出: {os.path.basename(file_path)}"))
                except ImportError:
                    try:
                        import imageio
                        imageio.imwrite(file_path, output.astype(np.float32), format='TIFF')
                        self.root.after(0, lambda: messagebox.showinfo("成功", f"已导出: {os.path.basename(file_path)}"))
                    except:
                        self.root.after(0, lambda: messagebox.showerror("错误", "需要安装tifffile或imageio"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        
        threading.Thread(target=export_thread, daemon=True).start()
    
    def clear_preview(self):
        self.image_canvas.delete("all")
        self.color_picker.clear_crosshair()
        self.image_info_label.config(text="未选择图像")
        self.cursor_info_label.config(text="")
        self.current_image_data = None
        self.base_point_var.set("未采样")
    
    def on_closing(self):
        self.render_running = False
        self.root.destroy()

if __name__ == '__main__':
    print("=" * 70)
    print("Aurhythm 胶片Cineon校准器 v1.0.0")
    print("=" * 70)
    print("功能: 直接读取RAW (NEF/DNG等) -> 片基采样 -> 增益调节 -> 导出Cineon 32位浮点TIFF")
    print("校准公式: CineonCode = 95 + 500 * log10(base / L)")
    print("=" * 70)
    
    try:
        import rawpy
        print("✓ rawpy已安装")
    except ImportError:
        print("✗ rawpy未安装，请执行: pip install rawpy")
    
    try:
        import tifffile
        print("✓ tifffile已安装")
    except ImportError:
        try:
            import imageio
            print("✓ imageio已安装")
        except ImportError:
            print("⚠ 建议安装tifffile或imageio: pip install tifffile")
    
    app = FilmProcessorUI()

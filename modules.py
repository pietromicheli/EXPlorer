# modules.py
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QLabel, QSlider, QHBoxLayout, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QVector3D

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from matplotlib import cm
import matplotlib.colors as mpl_colors
import time


# ---------------------- Base Module ---------------------- #
class BaseModule(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def update_frame(self, frame_idx: int):
        pass

# ---------------------- Video Module ---------------------- #
class VideoModule(BaseModule,):
    def __init__(self, video_path, fps=15.6, max_size=(800,500), add_sliders=False, parent=None):
        super().__init__(parent)
        # check and manage multiple videos
        if not isinstance(video_path, list):
            video_path = [video_path]

        video_unique_id = {p:i for i,p in enumerate(np.unique(video_path))}
        self.videos_caps = [cv2.VideoCapture(path) for path in video_unique_id]

        # map rec indices to videos indices
        frames_videos = []
        all_frames = 0
        for stim_name in video_path:
            stim_id = video_unique_id[stim_name]

            nframes = int(self.videos_caps[stim_id].get(cv2.CAP_PROP_FRAME_COUNT))
            # get resampling factor
            true_fps = self.videos_caps[stim_id].get(cv2.CAP_PROP_FPS)
            factor = true_fps/fps
            if factor < 1:
                # upsample
                factor = int(round(1/factor))
                frames_videos.extend((stim_id,f) for f in np.repeat(np.arange(0,nframes), factor))
                nframes = nframes*factor
            else:
                # downsample
                factor = int(round(factor))
                frames_videos.extend((stim_id,f) for f in range(0,nframes,factor))

            all_frames += nframes

        self.stim_map = {}
        for f,s in zip(range(all_frames),frames_videos):
            self.stim_map[f] = {'id':s[0],'frame':s[1]}

        # Container widget
        self.widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # QLabel for video display
        self.label = QLabel()
        self.label.setMaximumSize(max_size[0], max_size[1])
        self.label.setScaledContents(True) 
        layout.addWidget(self.label, 1)

        # sliders container
        slider_container = QWidget()
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_container.setLayout(slider_layout)

        # brightness slider (beta)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-200, 200)   # offset range
        self.brightness_slider.setValue(0)           # default = 0 offset
        self.brightness_slider.setFixedWidth(200)

        # contrast slider (alpha)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 400)       # 0.1x to 3.0x
        self.contrast_slider.setValue(100)           # default = 1.0x
        self.contrast_slider.setFixedWidth(200)

        if add_sliders:
            slider_layout.addWidget(self.brightness_slider)
            slider_layout.addWidget(self.contrast_slider)
            layout.addWidget(slider_container, 0, Qt.AlignmentFlag.AlignHCenter)

        self.widget.setLayout(layout)

        self.t0 = time.time()

    def update_frame(self, frame_idx: int):
        
        id,frame = self.stim_map[frame_idx].values()

        cap = self.videos_caps[id]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()

        if not ret:
            return
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # contrast & brightness adjustments
        alpha = self.contrast_slider.value() / 100.0   # contrast multiplier
        beta = self.brightness_slider.value()          # brightness offset
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        h, w = frame.shape
        img = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.label.setPixmap(QPixmap.fromImage(img).scaled(
            self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        
        print(f'fps: {1/(time.time()-self.t0)}', end='\r')
        self.t0 = time.time()

# ---------------------- Heatmap Module ---------------------- #
class HeatmapModule(BaseModule):
    def __init__(self, neural_data, win=600, parent=None):
        super().__init__(parent)
        self.data = neural_data
        self.win = win
        self.hm_min, self.hm_max = -3, 3

        # pad the end of the matrix
        self.data = np.pad(self.data, ((0,0),(0,int(self.win//2))), constant_values=np.inf)
        self.nframes = self.data.shape[1]

        # Main layout
        self.widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.widget.setLayout(layout)

        # Heatmap display
        self.hm_widget = pg.GraphicsLayoutWidget()
        self.view = self.hm_widget.addViewBox()
        self.view.setAspectLocked(False)
        self.img = pg.ImageItem()
        cmap = pg.colormap.get('Greys', source='matplotlib')
        self.img.setColorMap(cmap)
        self.view.addItem(self.img)
        self.cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.view.addItem(self.cursor)

        # Slider container
        slider_container = QWidget()
        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_container.setLayout(slider_layout)

        # Max slider (top)
        self.max_slider = QSlider(Qt.Orientation.Vertical)
        self.max_slider.setFixedHeight(100)
        self.max_slider.setFixedWidth(30)
        self.max_slider.setMinimum(int(self.hm_min * 1000))
        self.max_slider.setMaximum(int(self.hm_max * 1000))
        self.max_slider.setValue(int(self.hm_max * 1000))
        self.max_slider.valueChanged.connect(self.update_levels)
        slider_layout.addWidget(self.max_slider)

        # Min slider (bottom)
        self.min_slider = QSlider(Qt.Orientation.Vertical)
        self.min_slider.setFixedHeight(100)
        self.min_slider.setFixedWidth(30)
        self.min_slider.setMinimum(int(self.hm_min * 1000))
        self.min_slider.setMaximum(int(self.hm_max * 1000))
        self.min_slider.setValue(int(self.hm_min * 1000))
        self.min_slider.valueChanged.connect(self.update_levels)
        slider_layout.addWidget(self.min_slider)

        layout.addWidget(slider_container, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.hm_widget, 1)

    def update_frame(self, frame_idx: int):
        half_win = self.win // 2
        start = max(0, frame_idx - half_win)
        end = min(self.nframes, frame_idx + half_win)
        if end - start < self.win:
            if start == 0:
                end = min(self.win, self.nframes)
            elif end == self.nframes:
                start = max(0, self.nframes - self.win)

        heatmap_slice = self.data[:, start:end]
        self.cursor.setValue(frame_idx if frame_idx < half_win else half_win)
        self.img.setImage(heatmap_slice.T, levels=(self.hm_min, self.hm_max))

    def update_levels(self):
        min_val = self.min_slider.value() / 1000
        max_val = self.max_slider.value() / 1000
        if min_val >= max_val:
            return  # ignore invalid ranges
        self.hm_min = min_val
        self.hm_max = max_val

# ---------------------- Trace Module ---------------------- #
class TraceModule(BaseModule):
    def __init__(self, trace_data, label="Trace", color="w", win=600, parent=None):
        super().__init__(parent)
        self.data = trace_data
        self.win = win
        self.widget = pg.GraphicsLayoutWidget()
        self.plot = self.widget.addPlot()
        self.plot.setLabel('left', label, color='white')
        self.plot.getAxis('left').setPen('w')
        self.plot.getAxis('bottom').setPen('w')
        self.plot.setYRange(np.min(trace_data), np.max(trace_data))
        self.plot.setXRange(0, self.win)
        self.trace_item = self.plot.plot(pen=pg.mkPen(color, width=2))
        self.cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.plot.addItem(self.cursor)
        layout = QHBoxLayout()
        layout.addWidget(self.widget)
        self.setLayout(layout)

    def update_frame(self, frame_idx: int):
        half_win = self.win // 2
        start = max(0, frame_idx - half_win)
        end = min(len(self.data), frame_idx + half_win)
        if frame_idx < half_win:
            self.cursor.setValue(frame_idx)
            trace_slice = self.data[:self.win]
            xdata = np.arange(self.win)
        else:
            self.cursor.setValue(half_win)
            trace_slice = self.data[start:end]
            xdata = np.arange(end - start)
        self.trace_item.setData(x=xdata, y=trace_slice)

# ---------------------- 3D Projection Module ---------------------- #
class Projection3DModule(BaseModule):
    def __init__(self, neural_proj, trial_segments=None, colormaps=None,
                 default_cmap='viridis', default_alpha=0.4, min_size=(500, 300), parent=None):

        super().__init__(parent)
        self.data = neural_proj
        self.trial_segments = trial_segments if trial_segments is not None else {}
        self.colormaps = colormaps if colormaps is not None else {}
        self.default_cmap = default_cmap
        self.default_alpha = default_alpha

        # Create GL widget
        self.widget = gl.GLViewWidget()
        self.widget.setCameraPosition(distance=1.25)

        # Scatter plot item
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 0, 1, 1), size=0.1)
        self.widget.addItem(self.scatter)

        # Layout wrapper
        layout = QHBoxLayout()
        layout.addWidget(self.widget)
        self.setLayout(layout)
        self.widget.setMinimumSize(min_size[0], min_size[1])

        # self.init_camera()

    def update_frame(self, frame_idx: int):
        if frame_idx == 0:
            return

        positions = self.data[:frame_idx]
        colors = np.ones((len(positions), 4))  # default RGBA

        # Reset all colors to default colormap first
        default_cmap_obj = cm.get_cmap(self.default_cmap)
        colors[:, :] = default_cmap_obj(np.linspace(0, 1, len(positions)))
        colors[:, 3] = self.default_alpha

        # Overwrite colors for trial segments
        for trial_name, segments in self.trial_segments.items():
            cmap_name = self.colormaps.get(trial_name, 'cool')
            cmap = cm.get_cmap(cmap_name)
            for start, end in segments:
                # Clip to current frame_idx
                start_clip = max(0, min(start, frame_idx))
                end_clip = max(0, min(end, frame_idx))
                if end_clip <= start_clip:
                    continue
                seg_len = end_clip - start_clip
                # Generate colors spanning full colormap
                seg_colors = cmap(np.linspace(0, 1, seg_len))
                seg_colors[:, 3] = 0.08  # alpha for trial segments
                colors[start_clip:end_clip] = seg_colors

        self.scatter.setData(pos=positions, color=colors, size=6)

    def init_camera(self):
        if self.data is None or len(self.data) == 0:
            return

        # Compute min and max of data along each axis
        mins = self.data.min(axis=0)
        maxs = self.data.max(axis=0)
        center_np = (mins + maxs) / 2  # numpy array (3,)
        center = QVector3D(*center_np)  # convert to QVector3D

        # Compute the maximum range for all axes to set distance
        ranges = maxs - mins
        max_range = np.max(ranges)
        
        # Set camera position to be along z-axis at distance proportional to data range
        self.widget.setCameraPosition(
            pos=center,           # target position
            distance=max_range*2, # distance from target
            elevation=20,         # degrees above xy-plane
            azimuth=45            # rotation around z-axis
        )


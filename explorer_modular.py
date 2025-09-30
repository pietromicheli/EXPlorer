import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from matplotlib import cm

FPS = 15.6

# ---------------------- Base Module ---------------------- #
class BaseModule(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def update_frame(self, frame_idx: int):
        pass

# ---------------------- Video Module ---------------------- #
class VideoModule(BaseModule,):
    def __init__(self, video_path, fps=15, min_size=(500,300), parent=None):
        super().__init__(parent)
        self.label = QLabel()
        self.label.setMinimumSize(min_size[0], min_size[1])
        self.cap = cv2.VideoCapture(video_path)
        true_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.factor = int(round(true_fps/fps))
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        

    def update_frame(self, frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx*self.factor)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape
            img = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
            self.label.setPixmap(QPixmap.fromImage(img).scaled(
                self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

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
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Heatmap display
        self.widget = pg.GraphicsLayoutWidget()
        self.view = self.widget.addViewBox()
        self.view.setAspectLocked(False)
        self.img = pg.ImageItem()
        cmap = pg.colormap.get('Greys', source='matplotlib')
        self.img.setColorMap(cmap)
        self.view.addItem(self.img)
        self.cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.view.addItem(self.cursor)
        layout.addWidget(self.widget)

        # Slider container
        slider_container = QWidget()
        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_container.setLayout(slider_layout)

        # Max slider (top)
        self.max_slider = QSlider(Qt.Orientation.Vertical)
        self.max_slider.setMinimum(int(self.hm_min * 1000))
        self.max_slider.setMaximum(int(self.hm_max * 1000))
        self.max_slider.setValue(int(self.hm_max * 1000))
        self.max_slider.valueChanged.connect(self.update_levels)
        slider_layout.addWidget(self.max_slider)

        # Min slider (bottom)
        self.min_slider = QSlider(Qt.Orientation.Vertical)
        self.min_slider.setMinimum(int(self.hm_min * 1000))
        self.min_slider.setMaximum(int(self.hm_max * 1000))
        self.min_slider.setValue(int(self.hm_min * 1000))
        self.min_slider.valueChanged.connect(self.update_levels)
        slider_layout.addWidget(self.min_slider)

        layout.addWidget(slider_container)

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
    def __init__(self, neural_proj, parent=None):
        super().__init__(parent)
        self.data = neural_proj
        self.widget = gl.GLViewWidget()
        self.widget.setCameraPosition(distance=1.25)
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 0, 1, 1), size=3)
        self.widget.addItem(self.scatter)
        layout = QHBoxLayout()
        layout.addWidget(self.widget)
        self.setLayout(layout)

    def update_frame(self, frame_idx: int):
        if frame_idx > 0:
            pos = self.data[:frame_idx]
            col = cm.cool(np.linspace(0, 1, len(pos)))
            col[..., 3] = 0.4
            self.scatter.setData(pos=pos, color=col, size=5)

# ---------------------- Main Viewer ---------------------- #
# ... keep all previous imports and module classes ...

# ---------------------- Main Viewer ---------------------- #
class TimeSeriesViewer(QWidget):
    def __init__(self, layout_config, data_dict, win=600):
        super().__init__()
        self.setStyleSheet("background-color: black; color: white;")  # make entire app black

        self.modules = []
        self.frame_idx = 0
        self.win = win
        self.nframes = max([v.shape[0] if isinstance(v, np.ndarray) else 1000 for v in data_dict.values()])  # fallback
        layout = QGridLayout()
        self.setLayout(layout)

        for elem in layout_config['elements']:
            r, c = elem['row'], elem['col']
            t = elem['type']
            key = elem.get('data_key')
            cfg = elem.get('cfg', {})
            if t == 'video':
                module = VideoModule(data_dict[key], **cfg)
            elif t == 'heatmap':
                module = HeatmapModule(data_dict[key], win=self.win)
            elif t == 'trace':
                module = TraceModule(data_dict[key], **cfg, win=self.win)
            elif t == 'projection3d':
                module = Projection3DModule(data_dict[key])
                module.widget.setMinimumSize(500, 300)  # ensure visible
            else:
                continue
            layout.addWidget(module, r, c)
            self.modules.append(module)

        # Slider + Play/Pause buttons
        controls_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.nframes - 1)
        self.slider.setValue(0)
        self.slider.sliderMoved.connect(self.slider_moved)
        controls_layout.addWidget(self.slider)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play)
        self.play_btn.setStyleSheet("background-color: #333; color: white;")
        controls_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause)
        self.pause_btn.setStyleSheet("background-color: #333; color: white;")
        controls_layout.addWidget(self.pause_btn)

        layout.addLayout(controls_layout, layout_config['rows'], 0, 1, layout_config['cols'])

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 // FPS))

    def update_frame(self):
        for module in self.modules:
            module.update_frame(self.frame_idx)
        self.frame_idx = (self.frame_idx + 1) % self.nframes
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_idx)
        self.slider.blockSignals(False)

    def slider_moved(self, value):
        self.frame_idx = value
        self.update_frame()

    def play(self):
        if not self.timer.isActive():
            self.timer.start(int(1000 // FPS))

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()


# ---------------------- Run Function with Example ---------------------- #
def run_example():
    # Dummy data
    neural_matrix = np.load(r'example_data\neural_data.npy')
    pupil_area = np.load(r'example_data\pupilArea_mm2.npy')
    pupil_movement = np.load(r'example_data\pupilMotion_mm.npy')
    velocity = np.load(r'example_data\velocity.npy')
    neural_proj = np.load(r'example_data\data_GECO_3D.npy')
    video = r'example_data\TOE.mp4'

    
    
    # Layout config
    layout_config = {
        'rows': 4,
        'cols': 2,
        'elements': [
            {'row': 0, 'col': 0, 'type': 'heatmap', 'data_key': 'neural_matrix'},
            {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'pupil_area', 'cfg': {'label': 'Pupil Size', 'color': 'g'}},
            {'row': 2, 'col': 0, 'type': 'trace', 'data_key': 'velocity', 'cfg': {'label': 'Velocity', 'color': 'b'}},
            {'row': 3, 'col': 0, 'type': 'trace', 'data_key': 'pupil_movement', 'cfg': {'label': 'Pupil Movement', 'color': 'r'}},
            {'row': 0, 'col': 1, 'type': 'projection3d', 'data_key': 'neural_proj'},
            {'row': 1, 'col': 1, 'type': 'video', 'data_key': 'video', 'cfg': {'fps': 15.6, 'min_size':(400,300)}}
        ]
    }

    data_dict = {
        'neural_matrix': neural_matrix,
        'pupil_area': pupil_area,
        'pupil_movement': pupil_movement,
        'velocity': velocity,
        'neural_proj': neural_proj,
        'video': video
    }

    app = QApplication(sys.argv)
    viewer = TimeSeriesViewer(layout_config, data_dict, win=200)
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    run_example()

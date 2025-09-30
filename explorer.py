import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel
from PyQt6.QtWidgets import QPushButton, QSlider, QHBoxLayout, QVBoxLayout
from pyqtgraph import HistogramLUTItem
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, pyqtSlot, QObject, QMutex, QMutexLocker
from PyQt6.QtGui import QPixmap, QImage
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time


FPS = 15.6

def run(video_path_mouse,
        video_paths_stim,
        neural_data,
        pupil_area,
        pupil_xcoords,
        velocity,
        neural_proj,
        win = 600):

    '''
    Run the data explorer:

    - video_path_mouse (str):
        path to the mp4 behavioral video
    - video_paths_stim (str or list of str):
        path to the .npy file(s) contatining the textures of the stimuli
    - neural_data (array-like):
        neural matrix with shape (neurons, time)
    - pupil_area (array-like):
        pupil size data, 1D
    - pupil_xcoords (array-like):
        pupil motion data, 1D
    - velocity (array-like):
        locomotion velocity data, 1D
    - neural_proj (array-like):
        low-dimensional projection of the neural activity (like PCA), can be 2D OR 3D
    - win (int):
        size of the window 

    '''

    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { background-color: black; color: white; }")
    viewer = TimeSeriesViewer(video_path_mouse,video_paths_stim,
                            neural_data,
                            pupil_area,
                            pupil_xcoords,
                            velocity,
                            neural_proj,
                            win)
    viewer.show()
    app.aboutToQuit.connect(viewer.close)
    sys.exit(app.exec())

    return

class FrameDecoder(QObject):
    frame_ready = pyqtSignal(np.ndarray, int)  # Frame, index
    request_frame_signal = pyqtSignal(int)

    def __init__(self, path):
        super().__init__()
        self.cap = cv2.VideoCapture(path)
        self.lock = QMutex()
        self.current_idx = -1
        self.latest_frame = None
        self.request_frame_signal.connect(self.get_frame)

    @pyqtSlot(int)
    def get_frame(self, idx):
        with QMutexLocker(self.lock):
            if self.current_idx == idx and self.latest_frame is not None:
                self.frame_ready.emit(self.latest_frame, idx)
                return

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_idx = idx
                self.latest_frame = frame
                self.frame_ready.emit(frame, idx)

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()


class TimeSeriesViewer(QWidget):
    def __init__(self, video_path_mouse, video_paths_stim, neural_data, pupil_area, pupil_movement, velocity, neural_proj, win=600):
        super().__init__()

        self.neural_data = np.flip(neural_data,axis=0)
        self.pupil_area = pupil_area
        self.pupil_movement = pupil_movement
        self.velocity = velocity

        stim_unique_id = {p:i for i,p in enumerate(np.unique(video_paths_stim))}
        print(stim_unique_id)
        self.videos_stim = [np.load(path, mmap_mode='r')[::2] for path in stim_unique_id]#downsample
        # map rec indices to videos indices
        frames_stims = []
        for stim_name in video_paths_stim:
            stim_id = stim_unique_id[stim_name]
            frames_stims.extend((stim_id,f) for f in range(self.videos_stim[stim_id].shape[0]))
        
        self.stim_map = {}
        for f,s in zip(range(neural_data.shape[1]),frames_stims):
            self.stim_map[f] = {'id':s[0],'frame':s[1]}

        self.frame_idx = 0
        self.win = win
        
        print(f'neural data: {neural_data.shape} \npupil data: {self.pupil_area.shape} \nvelocity data: {self.velocity.shape}')

        self.neural_proj = neural_proj
        self.nframes = neural_data.shape[1]
        self.data_min = neural_data.min()
        self.data_max = neural_data.max()
        self.hm_min = -3 #self.data_min
        self.hm_max = 3 #self.data_max

        # Set-up the multithread video decoders
        # mouse
        self.decoder_thread_mouse = QThread()
        self.decoder_worker_mouse = FrameDecoder(video_path_mouse)
        self.decoder_worker_mouse.moveToThread(self.decoder_thread_mouse)
        # stim
        # self.decoder_thread_stim = QThread()
        # self.decoder_worker_stim = FrameDecoder(video_path_stim)
        # self.decoder_worker_stim.moveToThread(self.decoder_thread_stim)
        # Connect signals
        self.decoder_worker_mouse.frame_ready.connect(self.on_frame_ready_mouse)
        self.decoder_thread_mouse.start()
        # self.decoder_worker_stim.frame_ready.connect(self.on_frame_ready_stim)
        # self.decoder_thread_stim.start()

        ### Set up the layout of the app ###

        self.setWindowTitle("Timeseries Viewer (PyQtGraph)")
        self.resize(1600, 900)
        self.setStyleSheet("background-color: black;")

        layout = QGridLayout()
        self.setLayout(layout)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

        pg.setConfigOption('background', 'k')  # 'k' means black
        pg.setConfigOption('foreground', 'w')  # white text, axes, etc.

        # Add a horizontal layout for controls at the bottom
        self.controls_layout = QHBoxLayout()
        layout.addLayout(self.controls_layout, 2, 0, 1, 2)  # row=2, col=0, span 2 columns
        # Play button
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play)
        self.controls_layout.addWidget(self.play_btn)
        # Pause button
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause)
        self.controls_layout.addWidget(self.pause_btn)
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.nframes - 1)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.controls_layout.addWidget(self.slider)
        # speed buttons
        self.x1_btn = QPushButton("1x")
        self.x1_btn.clicked.connect(self.x1)
        self.controls_layout.addWidget(self.x1_btn)
        self.x2_btn = QPushButton("2x")
        self.x2_btn.clicked.connect(self.x2)
        self.controls_layout.addWidget(self.x2_btn)

        # Videos Display
        self.video_label_mouse = QLabel()
        self.video_label_mouse.setStyleSheet("background-color: black;")
        self.video_label_stim = QLabel()
        self.video_label_stim.setStyleSheet("background-color: black;")
        # self.stim_widget = pg.GraphicsLayoutWidget()
        # self.image_item_stim = pg.ImageItem()
        # self.view_box = self.stim_widget.addViewBox()
        # self.view_box.addItem(self.image_item_stim)
        
        # container layout for videos (side-by-side or stacked)
        video_container = QWidget()
        video_layout = QHBoxLayout()
        video_container.setLayout(video_layout)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(5)
        video_layout.addWidget(self.video_label_mouse, stretch=2)
        video_layout.addWidget(self.video_label_stim, stretch=1)
        layout.addWidget(video_container, 0, 0)

        # Container widget to hold heatmap + sliders
        heatmap_container = QWidget()
        heatmap_layout = QHBoxLayout()
        heatmap_container.setLayout(heatmap_layout)
        # Neural Matrix Heatmap
        # Title
        # heatmap_title = QLabel("Neural Activity")
        # heatmap_title.setStyleSheet("color: white; font-weight: bold;")
        # heatmap_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.heatmap_widget = pg.GraphicsLayoutWidget()
        self.heatmap_view = self.heatmap_widget.addViewBox()
        self.heatmap_view.setAspectLocked(False)
        self.heatmap_view.setBackgroundColor('k')
        self.heatmap_img = pg.ImageItem()
        cmap = pg.colormap.get('Greys', source='matplotlib')
        self.heatmap_img.setColorMap(cmap)
        self.heatmap_view.addItem(self.heatmap_img)
        self.heatmap_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.heatmap_view.addItem(self.heatmap_cursor)        
        # Create vertical sliders container
        slider_container = QWidget()
        slider_layout = QVBoxLayout()
        slider_container.setLayout(slider_layout)
        # Min slider (vertical)
        self.min_slider = QSlider(Qt.Orientation.Vertical)
        self.min_slider.setMinimum(int(self.hm_min*1000))
        self.min_slider.setMaximum(int(self.hm_max*1000))  # or adapt to your data range scale
        self.min_slider.setValue(int(self.hm_min * 1000))  # scale float to int
        self.min_slider.valueChanged.connect(self.update_levels)
        # Max slider (vertical)
        self.max_slider = QSlider(Qt.Orientation.Vertical)
        self.max_slider.setMinimum(int(self.hm_min*1000))
        self.max_slider.setMaximum(int(self.hm_max*1000))
        self.max_slider.setValue(int(self.hm_max*1000))
        self.max_slider.valueChanged.connect(self.update_levels)
        slider_layout.addWidget(self.max_slider)  # max on top (higher value)
        slider_layout.addWidget(self.min_slider)  # min on bottom (lower value)

        heatmap_layout.addWidget(self.heatmap_widget)
        heatmap_layout.addWidget(slider_container)
        # heatmap_layout.addWidget(heatmap_title)
        layout.addWidget(heatmap_container, 0, 1)

        # Traces
        self.trace_widgets = []
        self.trace_cursor_widget = []
        self.trace_data = [self.pupil_area, self.pupil_movement, self.velocity]
        self.trace_colors = ['g', 'r', 'b']
        self.trace_labels = ['Pupil Size', 'Pupil Movement', 'Velocity']
        self.trace_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self.trace_layout, 1, 0)
        self.trace_plots = []

        for i in range(3):
            p = self.trace_layout.addPlot(row=i, col=0)
            p.setLabel('left', self.trace_labels[i], color='white')
            p.showAxis('bottom', i == 2)
            p.getAxis('left').setPen('w')
            p.getAxis('left').setWidth(50)
            p.getAxis('bottom').setPen('w')
            p.setYRange(np.min(self.trace_data[i]), np.max(self.trace_data[i]))
            p.setXRange(0, self.win)
            trace = p.plot(pen=pg.mkPen(self.trace_colors[i], width=2))
            self.trace_widgets.append(trace)
            trace_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
            self.trace_cursor_widget.append(trace_cursor)
            p.addItem(trace_cursor)
            self.trace_plots.append(p)

        # 3D Projection
        gl_container = QWidget()
        gl_layout = QHBoxLayout()
        gl_container.setLayout(gl_layout)
        # Title
        # gl_title = QLabel("Neural Geometrical Embedding")
        # gl_title.setStyleSheet("color: white; font-weight: bold;")
        # gl_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.gl_view = gl.GLViewWidget()
        gl_layout.addWidget(self.gl_view)
        # gl_layout.addWidget(gl_title)
        layout.addWidget(gl_container, 1,1)
        self.gl_view.setMinimumSize(600, 400) 
        self.gl_view.setCameraPosition(distance=1.25) 
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 0, 1, 1), size=3)
        self.arrow = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=(1, 0, 0, 1), width=4, antialias=True)
        self.gl_view.addItem(self.scatter)
        self.gl_view.addItem(self.arrow)
        # axes lines
        # maxs = self.neural_proj.max(axis=0)
        # mins = self.neural_proj.min(axis=0)
        # axis_x = gl.GLLinePlotItem(pos=np.array([[mins[0], 0, 0], [maxs[0], 0, 0]]), color=(1, 1, 1, 1), width=2)
        # axis_y = gl.GLLinePlotItem(pos=np.array([[0, mins[1], 0], [0, maxs[1], 0]]), color=(1, 1, 1, 1), width=2)
        # axis_z = gl.GLLinePlotItem(pos=np.array([[0, 0, mins[2]], [0, 0, maxs[2]]]), color=(1, 1, 1, 1), width=2)
        # self.gl_view.addItem(axis_x)
        # self.gl_view.addItem(axis_y)
        # self.gl_view.addItem(axis_z)

        cmaps = ['spring', 'summer', 'autumn', 'winter', 'cool']
        self.custom_cmap = []
        for path in video_paths_stim:
            id = stim_unique_id[path]
            stim = self.videos_stim[id]
            nframes = stim.shape[0]
            c = cm.get_cmap(cmaps[id])(np.linspace(0, 1, nframes))
            c[..., 3] = 0.4 #alpha
            self.custom_cmap.append(c)
        self.custom_cmap = np.vstack(self.custom_cmap)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.t0 = time()
        self.timer.start(int(1000 // FPS))


    def update_frame(self):

        print('fps : {}'.format(1/(time()-self.t0)), end='\r')
        self.t0 = time()

        # Videos
        # face
        self.decoder_worker_mouse.request_frame_signal.emit(self.frame_idx*2)
        # stim
        id,frame = self.stim_map[self.frame_idx].values()
        stim_frame = self.videos_stim[id][frame]
        self.on_frame_ready_stim(stim_frame)

        start = max(0, self.frame_idx - self.win)
        end = self.frame_idx

        half_win = self.win // 2
        start_hm = max(0, self.frame_idx - half_win)
        end_hm = min(self.nframes, self.frame_idx + half_win)
        heatmap_slice = self.neural_data[:,start_hm:end_hm]
        # Ensure window size is fixed
        if end_hm - start_hm < self.win:
            if start_hm == 0:
                end_hm = min(self.win, self.nframes)
            elif end_hm == self.nframes:
                start_hm = max(0, self.nframes - self.win)

        # Traces
        if end > 0:
            for i, (trace,cursor) in enumerate(zip(self.trace_widgets,self.trace_cursor_widget)):
                if self.frame_idx < half_win:
                    cursor.setValue(self.frame_idx)
                    trace_slice = self.trace_data[i][:self.win]
                    xdata = np.arange(self.win)
                    
                else:
                    cursor.setValue(half_win)
                    trace_slice = self.trace_data[i][start_hm:end_hm]
                    xdata = np.arange(end_hm - start_hm)
                # Update plot
                trace.setData(x=xdata,y=trace_slice)

        if end > 0:
            # Update heatmap
            if self.frame_idx < half_win:
                self.heatmap_cursor.setValue(self.frame_idx)
                heatmap_slice = self.neural_data[:,:self.win].copy()
            else:
                self.heatmap_cursor.setValue(half_win)
                heatmap_slice = self.neural_data[:,start_hm:end_hm].copy()

            # Update image
            self.heatmap_img.setImage(heatmap_slice.T, levels=(self.hm_min, self.hm_max))

        # Update 3D plot
        if end > 0:
            # col = cm.cool(np.linspace(0, 1, self.nframes))
            # col[..., 3] = 0.4 #alpha
            pos = self.neural_proj[:end]
            self.scatter.setData(pos=pos, color=self.custom_cmap[:end], size=5)
            self.get_direction_vec()

        self.frame_idx += 1
        # prevent recursive signal
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_idx)
        self.slider.blockSignals(False)

    def play(self):
        if not self.timer.isActive():
            self.timer.start(int(1000 // FPS))

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()

    def slider_moved(self, value):
        self.frame_idx = value
        self.update_frame()

    def x1(self):
        if self.timer.isActive():
            self.timer.stop()

        self.timer.start(int(1000 // FPS))

    def x2(self):
        if self.timer.isActive():
            self.timer.stop()

        self.timer.start(int(1000 // (FPS*2)))

    def update_levels(self):
        min_val = self.min_slider.value() / 1000
        max_val = self.max_slider.value() / 1000
        if min_val >= max_val:
            # Avoid invalid range - just ignore or fix
            return
        # self.heatmap_img.setLevels((min_val, max_val))
        self.hm_min = min_val
        self.hm_max = max_val

    @pyqtSlot(np.ndarray)
    def on_frame_ready_mouse(self, frame):
        if frame is None:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape
        bytes_per_line = w
        img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        self.video_label_mouse.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label_mouse.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
  
    def on_frame_ready_stim(self, frame):
        if frame is None:
            return

        h, w = frame.shape
        bytes_per_line = w
        img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        self.video_label_stim.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label_stim.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def get_direction_vec(self, npoints=50):
        
        if self.frame_idx>15:
            if self.frame_idx < npoints:
                npoints = self.frame_idx

            # Select the last n points
            last_points = self.neural_proj[self.frame_idx-npoints:self.frame_idx]
            # Compute average direction last_points (tip - tail) per chunk of 15 frames
            directions = []
            s = 0
            for i in range(15,npoints,15):
                direction = last_points[i] - last_points[s] 
                directions.append(direction)
                s = i
            avg_direction = np.mean(directions, axis=0)
            avg_direction /= np.linalg.norm(avg_direction)  # Normalize
            # Scale the arrow for visibility
            arrow_length = np.linalg.norm(last_points[-1]-last_points[0]) 
            arrow_vector = avg_direction * arrow_length
            # Start point of arrow: last point of trajectory
            start_point = self.neural_proj[self.frame_idx-npoints:self.frame_idx].mean(axis=0)
            end_point = start_point + arrow_vector
            arrow_pts = np.array([start_point, end_point])

            self.arrow.setData(pos=arrow_pts)

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        
        if self.decoder_thread_mouse.isRunning():
            self.decoder_worker_mouse.stop()
            self.decoder_thread_mouse.quit()
            self.decoder_thread_mouse.wait()

        # if self.decoder_thread_stim.isRunning():
        #     self.decoder_worker_stim.stop()
        #     self.decoder_thread_stim.quit()
        #     self.decoder_thread_stim.wait()

        # Cleanup OpenGL
        if hasattr(self, 'gl_view'):
            try:
                self.gl_view.clear()
                self.gl_view.setParent(None)  # Detach from layout
                self.gl_view.deleteLater()
            except Exception as e:
                print("Error while clearing GLViewWidget:", e)

        event.accept()


if __name__ == '__main__':

    import sys
    args = sys.argv[1:]
    print('\nParsed args: {}\n'.format(args))
    args_data = [np.load(arg) for arg in args[1:6]]
    mouse_video = args[0]
    stim_videos = args[6:]
    win = 600
    
    run(mouse_video, stim_videos, *args_data, win)
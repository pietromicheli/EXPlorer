from collections import defaultdict
import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QInputDialog, QMainWindow, QSizePolicy
from PyQt6.QtCore import QTimer, Qt
from .modules import *

# ---------------------- Main Viewer ---------------------- #
class TimeSeriesViewer(QMainWindow):
    def __init__(self, layout_config, data_dict, fps, win=600):
        super().__init__()
        self.setStyleSheet("background-color: black; color: white;")  # make entire app black
        self.fps = fps
        self.modules = []
        self.frame_idx = 0
        self.win = win
        self.nframes = max([v.shape[0] if isinstance(v, np.ndarray) else 1000 for v in data_dict.values()])  # fallback

        self.module_map = {
                        'video': VideoModule,
                        'heatmap': HeatmapModule,
                        'trace': TraceModule,
                        'projection3d': Projection3DModule      
            }

        layout = QGridLayout()

        # Group elements by grid position
        grouped = defaultdict(list)
        
        for elem in layout_config['elements']:
            grouped[(elem['row'], elem['col'])].append(elem)

        for (r, c), elems in grouped.items():
            if len(elems) == 1:
                # Just one module, add directly
                elem = elems[0]
                module = self.make_module(elem, data_dict)
                layout.addWidget(module.widget, r, c)
                self.modules.append(module)
            else:
                # Multiple modules → put inside a container
                container = QWidget()
                vbox = QVBoxLayout()
                vbox.setContentsMargins(0, 0, 0, 0)
                for elem in elems:
                    module = self.make_module(elem, data_dict)
                    vbox.addWidget(module.widget)
                    self.modules.append(module)
                container.setLayout(vbox)
                layout.addWidget(container, r, c)

        # make all rows and columns equally resizable
        for r in range(layout_config['rows']):
            layout.setRowStretch(r, 1)
        for c in range(layout_config['cols']):
            layout.setColumnStretch(c, 1)

        # Wrap into QWidget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Slider + timer + speed + Play/Pause buttons
        controls_layout = QHBoxLayout()

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.nframes - 1)
        self.slider.setValue(0)
        self.slider.sliderMoved.connect(self.slider_moved)
        controls_layout.addWidget(self.slider)

        self.time_label = QLabel("0.0 s")
        self.time_label.setStyleSheet("color: white; font-weight: bold;")
        controls_layout.addWidget(self.time_label)

        self.speed_btn = QPushButton("Set Speed")
        self.speed_btn.clicked.connect(self.set_speed)
        controls_layout.addWidget(self.speed_btn)
        # Default speed factor
        self.speed_factor = 1

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
        self.timer.start(int(1000//self.fps))

    def make_module(self, elem, data_dict):
        t = elem['type']
        key = elem.get('data_key')
        cfg = elem.get('cfg', {})

        module = self.module_map[t](data_dict[key], **cfg)
        return module

    def update_frame(self):
        for module in self.modules:
            module.update_frame(self.frame_idx)

        self.frame_idx = (self.frame_idx + self.speed_factor) % self.nframes

        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_idx)
        self.slider.blockSignals(False)
        time_sec = self.frame_idx / self.fps
        mins, secs = divmod(int(time_sec), 60)
        self.time_label.setText(f"{mins:02}:{secs:02}")


    def slider_moved(self, value):
        self.frame_idx = value
        self.update_frame()

    def set_speed(self):
        # Prompt the user for an integer
        speed, ok = QInputDialog.getInt(self, "Playback Speed", "Read every n frames (integer >=1):", 
                                    value=self.speed_factor, min=1)
        if ok:
            self.speed_factor = speed

    def play(self):
        if not self.timer.isActive():
            self.timer.start(int(1000//FPS))

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()


# ---------------------- Run Function with Example ---------------------- #
def run_example():

    # this example uses real data from a recording session, stored in the example_data folder with the following structure:
    # -example_data/
    #    -neural_data.npy
    #    -pupilArea_mm2.npy
    #    -pupilMotion_mm.npy
    #    -velocity.npy
    #    -data_GECO_3D.npy
    #    -TOE.mp4
    #    -textures/
    #        -TouchOfEvil.mp4
    #        -interval_10s.mp4
    
    neural_matrix = np.load(r'example_data\neural_data.npy')
    data_len = neural_matrix.shape[1]
    # trim other data stream according to the neural data
    pupil_area = np.load(r'example_data\pupilArea_mm2.npy')[:data_len]
    pupil_movement = np.load(r'example_data\pupilMotion_mm.npy')[:data_len]
    velocity = np.load(r'example_data\velocity.npy')[:data_len]
    neural_proj = np.load(r'example_data\data_GECO_3D.npy')[:data_len]
    video_behav = r'example_data\TOE.mp4'
    # the dataset was recorded during the presentation of the following stimulation protocol:
    # ISI(10s)-Trial1(TOE,~4 min) X3
    textures_stim = [r'example_data\textures\interval_10s.mp4',r'example_data\textures\TouchOfEvil.mp4']*3+[r'example_data\textures\interval_10s.mp4']
    
    # Layout config
    layout_config = {
        'rows': 2,
        'cols': 2,
        'elements': [
            {'row': 0, 'col': 0, 'type': 'heatmap', 'data_key': 'neural_matrix'},
            {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'pupil_area', 'cfg': {'label': 'Pupil Size', 'color': 'g'}},
            {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'velocity', 'cfg': {'label': 'Velocity', 'color': 'b'}},
            {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'pupil_movement', 'cfg': {'label': 'Pupil Movement', 'color': 'r'}},
            {'row': 0, 'col': 1, 'type': 'video', 'data_key': 'video', 'cfg': {'min_size':(400,300)}},
            {'row': 0, 'col': 1, 'type': 'video', 'data_key': 'textures_stim', 'cfg': {'min_size':(400,300)}},
            {'row': 1, 'col': 1, 'type': 'projection3d', 'data_key': 'neural_proj', 'cfg': {'min_size':(400,300)}},
        ]
    }

    data_dict = {
        'neural_matrix': neural_matrix,
        'pupil_area': pupil_area,
        'pupil_movement': pupil_movement,
        'velocity': velocity,
        'neural_proj': neural_proj,
        'video': video_behav,
        'textures_stim': textures_stim
    }

    app = QApplication(sys.argv)
    viewer = TimeSeriesViewer(layout_config, data_dict, win=200, fps=15.6)
    viewer.show()
    sys.exit(app.exec())

def run_example_dummy():

    # --- Dummy data generation ---
    n_neurons = 50
    n_frames = 1000

    # Heatmap (neural data)
    neural_matrix = np.random.randn(n_neurons, n_frames)

    # Traces
    pupil_area = np.random.rand(n_frames) * 5 + 2      # example pupil size
    velocity = np.random.rand(n_frames) * 2            # example locomotion velocity
    pupil_movement = np.random.randn(n_frames)        # example pupil movement

    # 3D projection
    neural_proj = np.cumsum(np.random.randn(n_frames, 3), axis=0)  # random walk in 3D
    trial_segments = {
        'trial1': [(0, 100), (300, 400)],
        'trial2': [(100, 300)]
    }
    colormaps = {
        'trial1': 'cool',
        'trial2': 'autumn'
    }

    # Layout config
    layout_config = {
        'rows': 2,
        'cols': 2,
        'elements': [
            {'row': 0, 'col': 0, 'type': 'heatmap', 'data_key': 'neural_matrix'},
            {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'pupil_area', 'cfg': {'label': 'Pupil Size', 'color': 'g'}},
            {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'velocity', 'cfg': {'label': 'Velocity', 'color': 'b'}},
            {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'pupil_movement', 'cfg': {'label': 'Pupil Movement', 'color': 'r'}},
            {'row': 1, 'col': 1, 'type': 'projection3d', 'data_key': 'neural_proj', 'cfg': {'trial_segments': trial_segments, 'colormaps': colormaps, 'min_size':(400,300)}},
        ]
    }

    data_dict = {
        'neural_matrix': neural_matrix,
        'pupil_area': pupil_area,
        'pupil_movement': pupil_movement,
        'velocity': velocity,
        'neural_proj': neural_proj,
    }

    app = QApplication(sys.argv)
    viewer = TimeSeriesViewer(layout_config, data_dict, win=200, fps=15.6)
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run_example_dummy()

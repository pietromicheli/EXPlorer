# Modular Neuroscience Data Viewer

A **modular PyQt6/PyQtGraph application** for visualizing aligned neuroscience data, including behavioral videos, stimulus textures, neural activity, and other time series. The application allows flexible layout specification and supports multiple synchronized data streams.

---

## Features

- **Video playback**
  - Display multiple videos (e.g. behavior, stimulus, FOV ...), side by side.
  - Adjustable video size.
  
- **Heatmap visualization**
  - Neural activity displayed as a heatmap.
  - Adjustable color scale with vertical min/max sliders.
  
- **Trace plotting**
  - Plot individual 1D signals (e.g., traces of pupil size, velocity, movement).
  - Each trace is modular and can be placed anywhere in the layout.
  - Moving cursor shows current frame position.
  
- **3D projection**
  - Visualize low-dimensional embeddings of neural data in 3D.
  
- **Flexible layout**
  - Specify arbitrary rows and columns.
  - Multiple modules in the same cell are stacked vertically automatically.
  
- **Playback controls**
  - Play / Pause buttons.
  - Slider to jump to any frame.
  - Time display in minutes and seconds.
  - Adjustable speed to skip frames for faster playback.

---

## Usage Example

Below is a **complete example** showing **generic file names, variable names, and dictionary keys**, ready to adapt.

```python
import numpy as np
from PyQt6.QtWidgets import QApplication
from viewer import TimeSeriesViewer  # your main app file

# --- Load your data ---
heatmap_data = np.load('heatmap_data.npy')
trace_a = np.load('trace_a.npy')
trace_b = np.load('trace_b.npy')
trace_c = np.load('trace_c.npy')
projection_3d_data = np.load('projection_3d.npy')
video_main = 'video_main.mp4'
video_stimuli = ['video_stim1.mp4', 'video_stim2.mp4']

# --- Pack data in a dictionary ---
data_dict = {
    'heatmap_data': heatmap_data,
    'trace_a': trace_a,
    'trace_b': trace_b,
    'trace_c': trace_c,
    'projection_3d': projection_3d_data,
    'video_main': video_main,
    'video_stimuli': video_stimuli
}

# --- Define the layout ---
layout_config = {
    'rows': 2,
    'cols': 2,
    'elements': [
        {'row': 0, 'col': 0, 'type': 'heatmap', 'data_key': 'heatmap_data'},
        {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'trace_a', 'cfg': {'label': 'Trace A', 'color': 'g'}},
        {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'trace_b', 'cfg': {'label': 'Trace B', 'color': 'b'}},
        {'row': 1, 'col': 0, 'type': 'trace', 'data_key': 'trace_c', 'cfg': {'label': 'Trace C', 'color': 'r'}},
        {'row': 0, 'col': 1, 'type': 'video', 'data_key': 'video_main', 'cfg': {'min_size': (400,300)}},
        {'row': 0, 'col': 1, 'type': 'video', 'data_key': 'video_stimuli', 'cfg': {'min_size': (400,300)}},
        {'row': 1, 'col': 1, 'type': 'projection3d', 'data_key': 'projection_3d', 'cfg': {'min_size': (400,300)}},
    ]
}

# --- Launch the viewer ---
app = QApplication([])
viewer = TimeSeriesViewer(layout_config, data_dict, fps=15)
viewer.show()
app.exec()

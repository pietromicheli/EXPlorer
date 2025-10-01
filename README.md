# Modular Neuroscience Data Viewer

This is a **modular PyQt6/PyQtGraph application** for visualizing aligned neuroscience data, including behavioral videos, stimulus textures, neural activity, and other time series data. The application allows flexible layout specification and supports multiple synchronized data streams.

---

## Features

- **Video playback**
  - Display behavioral videos and stimulus videos side by side.
  - Supports multiple videos concatenated along the timeline.
  - Adjustable video size.
  
- **Heatmap visualization**
  - Neural activity displayed as a heatmap.
  - Adjustable color scale with vertical min/max sliders.
  
- **Trace plotting**
  - Plot individual 1D signals (e.g., pupil size, velocity, pupil movement).
  - Each trace is modular and can be placed anywhere in the layout.
  - Moving cursor shows current frame position.
  
- **3D projection**
  - Visualize low-dimensional neural embeddings (like PCA) in 3D.
  
- **Flexible layout**
  - Specify arbitrary rows and columns.
  - Multiple modules in the same cell are stacked vertically in a container.
  
- **Playback controls**
  - Play / Pause buttons.
  - Slider to jump to any frame.
  - Time display in seconds.
  - Speed control to skip frames for faster playback.

---

## Installation

Requires Python 3.8+ and the following packages:

```bash
pip install numpy opencv-python PyQt6 pyqtgraph matplotlib


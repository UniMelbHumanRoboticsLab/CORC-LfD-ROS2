import sys

import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)
from spatialmath import SO3

from PySide6 import QtWidgets, QtCore # use 6.9.0
from PySide6.QtCore import QTimer,Qt
from PySide6.QtGui import QFont
from vispy import scene
from vispy.app import use_app
print(use_app('pyside6'))

import pyqtgraph as pg # use dev pyqtgraph
pg.setConfigOptions(antialias=False)     # lines render faster
pg.setConfigOptions(useOpenGL=True)
pg.setConfigOptions(useCupy=True)
pg.setConfigOptions(useNumba=True)
pg.setConfigOptions(crashWarning=True)

import debugpy
def init_debug():
    debugpy.listen(("localhost", 5678))
    print("Waiting for debugger…")
    debugpy.wait_for_client()

def debug_break():
    import debugpy
    debugpy.breakpoint()
    
N_buffer = 50
# Disable mouse wheel zooming in plots for pyqtgraph
class NoWheelViewBox(pg.ViewBox):
    def __init__(self):
        super().__init__()
        self.setMouseEnabled(x=False, y=False)  
    def wheelEvent(self, ev):
        ev.ignore()   # don't zoom; let parent (e.g., QScrollArea) handle scrolling

class pycorc_gui(QtWidgets.QMainWindow):
    def __init__(self,freq=100,gui_3d=True):
        super().__init__()
        self.setWindowTitle("Live Sensor Plot")
        
        # Get all available screens
        app = QtWidgets.QApplication.instance()
        screens = app.screens()
        # set to the second screen dimension (index 1) if available
        if len(screens) > 1:
            second_screen = screens[1]
            screen_geometry = second_screen.geometry()
            (tl_x,tl_y,br_x,br_y) = screen_geometry.getCoords()
            screen_geometry.setCoords(int(br_x/2),0,br_x,br_y)
            self.setScreen(second_screen)
            self.setGeometry(screen_geometry)
        # self.showFullScreen()

        # self.container and main layout boxes, this is the main box: DONT TOUCH
        self.container = QtWidgets.QWidget()
        self.setCentralWidget(self.container)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.container.setLayout(self.main_layout)

        # add your stuff here to the main layout
        self.labels_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.labels_layout)
        
        # Setup 3D view
        if gui_3d:
            self.init_gui_3d()

        # timer to update the main GUI
        self.gui_timer = QTimer()
        self.gui_timer.setTimerType(Qt.PreciseTimer)
        self.gui_timer.timeout.connect(self.update_gui)
        self.gui_timer.start(int(1/freq*1000))
        
        # FPS variable to calc gui FPS
        self.gui_frame_count = 0
        self.gui_fps = 0
        self.gui_fps_timer = QtCore.QElapsedTimer() # use the system clock
        self.gui_fps_timer.start()
        self.gui_cur_time = self.gui_fps_timer.elapsed()
        self.gui_last_time = self.gui_fps_timer.elapsed()
        self.gui_start_time = self.gui_fps_timer.elapsed()
        self.gui_elapsed_time = 0
        self.gui_label = self.init_response_label(size=[200,50])

        self.plt_items = []

    """
    Init GUI Helper Functions
    """
    def init_live_stream(self,num_columns=3):
        widget = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(widget)        
        live_stream = {
            "time_buffer":np.linspace(-1, 0.0, N_buffer).astype(np.float32),
            "widget":widget,
            "num_of_plots":0,
            "num_columns":num_columns
        }
        return live_stream
    def add_live_stream_plot(self,live_stream,sensor_name="FSR",unit="N",dim=1,autoscale=True,max_range=10):
        r,c = divmod(live_stream["num_of_plots"],live_stream["num_columns"])
        vb = NoWheelViewBox()
        plt = live_stream["widget"].addPlot(row=r, col=c,viewBox=vb)
        plt.setTitle(f'{sensor_name}_{live_stream["num_of_plots"]}', color='b', size='20pt')
        plt.setLabel('left', unit)
        plt.setLabel('bottom', 'Time (s)')
        plt.addLegend(offset=(10, 10))
        plt.showGrid(x=False, y=True)
        # Disable autoscale
        if not autoscale:
            plt.enableAutoRange(True, False)  # Disable for both x and y axes
            # plt.setXRange(-1, 0)  # Match the time_buffer range
            plt.setYRange(0, max_range*1.25)  # Set an appropriate Y range based on your data
        self.plt_items.append(plt)
        live_stream["num_of_plots"] += 1
        
        col = ["r","g","b"]
        if dim == 1:
            name = ["norm"]
            col = ["r","g","b"]
            width = 1
        elif dim ==3:
            name = ["x", "y", "z"]
            col = ["r","g","b"]
            width = 1
        elif dim == 2:
            name = ["instructed","measured"]
            col = ["#fff4e6","#00fd0d"]
            width = 5

        data_buffer = np.zeros((N_buffer,dim), dtype=np.float32)
        lines = []
        for i in range(dim):
            lines.append(plt.plot(live_stream["time_buffer"], 
                                  data_buffer[:,i]+i+1, 
                                  pen=pg.mkPen(col[i], width=width), 
                                  name=name[i]))
            lines[i].setDownsampling(auto=True, method='subsample')
            lines[i].setClipToView(state=True)
            lines[i].setSkipFiniteCheck(skipFiniteCheck=True)

        live_stream_plot = {
            "lines": lines,
            "data": data_buffer 
        }
        return live_stream_plot
    def init_gui_3d(self):
        self.canvas_gui_3d = scene.SceneCanvas(keys='interactive', show=True,size=(800, 400))
        self.canvas_gui_3d
        self.canvas_gui_3d.create_native()
        self.main_layout.addWidget(self.canvas_gui_3d.native)
        self.view_gui_3d = self.canvas_gui_3d.central_widget.add_view()
        self.view_gui_3d.camera = scene.TurntableCamera(up='z',fov=30, distance=2,azimuth=135,elevation=20)

        # Grid for reference
        # axis = scene.visuals.XYZAxis(parent=self.view_gui_3d.scene)
        grid = scene.visuals.GridLines(grid_bounds=(-5, 5, -5, 5),border_width=1)
        self.view_gui_3d.add(grid)

        # create inertial coordinate system
        self.inert_frame = SO3().R
        self.inert_point = np.array([[0,0,0]])
        self.init_point(pos=self.inert_point)
        self.init_frame(pos=self.inert_point,rot=self.inert_frame*0.1)
    def init_line(self,points,color):
        plt = scene.visuals.Line(pos=points, color=color, width=4, method='gl',antialias=False)
        self.view_gui_3d.add(plt)
        return plt
    def init_frame(self, pos, rot):
        # Store axis colors
        self.triaxis_colors = np.array([
            [1, 0, 0, 1],  # X red
            [1, 0, 0, 1],
            [0, 1, 0, 1],  # Y green
            [0, 1, 0, 1],
            [0, 0, 1, 1],  # Z blue
            [0, 0, 1, 1]
        ], dtype=np.float32)

        # Create initial vertex array (6 points = 3 axes * 2 points each)
        verts = np.array([
            pos, pos + rot[:, 0],  # X axis
            pos, pos + rot[:, 1],  # Y axis
            pos, pos + rot[:, 2]   # Z axis
        ], dtype=np.float32)

        # One Line visual for all three axes
        plt = scene.Line(width=4,pos=verts,color=self.triaxis_colors,connect='segments',method='gl',antialias=False)
        self.view_gui_3d.add(plt)
        return plt
    def init_point(self,pos,col=(0,1,1,1)):
        plt = scene.visuals.Markers(pos=pos, edge_color=None, face_color=(0.5, 0.5, 0.5, 1),size=12)
        self.view_gui_3d.add(plt)
        return plt
    def init_response_label(self,size=[200,200],fontsize=10):
        response_label = QtWidgets.QLabel("WAITING RESPONSE")
        response_label.setStyleSheet("background-color: black; color: white;")
        response_label.setFixedSize(size[0],size[1])
        response_label.setFrameStyle(QtWidgets.QFrame.Shape.Box | QtWidgets.QFrame.Shadow.Plain)
        font = QFont("Courier New")
        font.setPixelSize(fontsize)
        response_label.setFont(font)  # monospace
        response_label.setTextFormat(Qt.TextFormat.PlainText)  # keep spaces as-is
        self.labels_layout.addWidget(response_label)
        return response_label
    """
    Update GUI Helper Functions
    """
    def update_live_stream_buffer(self,live_stream):
        # update shared time buffer
        live_stream["time_buffer"][:-1] = live_stream["time_buffer"][1:]
        live_stream["time_buffer"][-1] = self.gui_elapsed_time
    def update_live_stream_plot(self,live_stream,live_stream_plot,new_data,dim=1):
        live_stream_plot["data"][:-1,:] = live_stream_plot["data"][1:,:]
        live_stream_plot["data"][-1,:] = new_data[:dim]
        for i,line in enumerate(live_stream_plot["lines"]):
            line.setData(live_stream["time_buffer"],live_stream_plot["data"][:,i])           
    def update_line(self,plt,points):
        plt.set_data(pos=points)
    def update_frame(self, plt,pos, rot):
        # Update all 6 points at once
        verts = np.array([
            pos, pos + rot[:, 0],
            pos, pos + rot[:, 1],
            pos, pos + rot[:, 2]
        ], dtype=np.float32)
        # Update the line visual in one call
        plt.set_data(pos=verts, color=self.triaxis_colors)
    def update_point(self,plt,pos):
        if len(pos.shape) != 2:
            pos = np.array([pos.tolist()])
        plt.set_data(pos=pos)
    def update_response_label(self,response_label,info):
        response_label.setText(info)
    def update_gui(self):
        self.gui_frame_count += 1
        self.gui_cur_time = self.gui_fps_timer.elapsed()
        self.gui_elapsed_time = (self.gui_cur_time-self.gui_start_time)/1000.0
        if self.gui_cur_time-self.gui_last_time >= 1000:
            self.gui_fps = self.gui_frame_count * 1000 / (self.gui_cur_time-self.gui_last_time)
            self.gui_last_time = self.gui_cur_time
            self.gui_frame_count = 0
            self.update_response_label(self.gui_label,f"FPS:{self.gui_fps}\nElapsed Time:{self.gui_elapsed_time:.2f}s")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = pycorc_gui(freq=100)
    w.setWindowTitle("pycorc")
    w.show()
    sys.exit(app.exec())
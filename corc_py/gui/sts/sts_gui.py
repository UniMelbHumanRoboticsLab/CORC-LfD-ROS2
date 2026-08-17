print("Importing Paths")
import os, sys,json
for mod in [m for m in sys.modules if 'PyQt5' in m]:
    del sys.modules[mod]
    
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)
print("Importing Logger")
from sts_logger import sts_logger
print("Importing GUI")
from base.pycorc_gui import pycorc_gui
print("Importing IOs")
from pycorc_io.corc.corc_FLNL_client import corc_FLNL_client
from pycorc_io.xsens.xsens_CoM_server import xsens_server
print("Importing PySide6")
from PySide6 import QtWidgets
from PySide6.QtGui import QShortcut
from PySide6.QtCore import QThread,Qt,QMetaObject,Slot
from spatialmath import SO3, SE3

class sts_gui(pycorc_gui):
    def __init__(self,init_args):
        self.init_args = init_args
        self.corc_args = self.init_args["init_flags"]["corc"]
        self.xsens_args = self.init_args["init_flags"]["xsens"]
        self.gui_args = self.init_args["init_flags"]["gui"]
        self.log_args = self.init_args["init_flags"]["log"]
        self.session_data = self.init_args["session_data"]

        self.num_closed_threads = 0
        self.num_opened_threads = 0
        self.requested_threads = 2

        super().__init__(freq=self.gui_args["freq"],gui_3d=self.gui_args["3d"])
        
        # initialize live stream
        if self.gui_args["vel"]:
            self.sts_vel_live_stream = self.init_live_stream(num_columns=3)
            
        """
        Initilization
        """
        self.init_IOs()
        self.init_shortcuts()

    """
    Init Sensor Worker / Thread / GUI Helper Functions
    """
    def add_opened_threads(self):
        self.num_opened_threads += 1
        print(f"Threads Opened:{self.num_opened_threads}")
        
        if self.num_opened_threads == self.requested_threads:
            QMetaObject.invokeMethod(self.logger_worker , "sensors_ready", Qt.ConnectionType.QueuedConnection)
    def init_corc(self):
        # init response label
        self.corc_label = self.init_response_label(size=[525,200])
        # init corc thread and worker
        self.corc_thread = QThread()
        self.corc_worker = corc_FLNL_client(ip=self.corc_args["ip"],port=self.corc_args["port"])
    def init_xsens(self):
        # init response label
        self.xsens_label = self.init_response_label(size=[300,300])
        # init corc thread and worker
        self.xsens_thread = QThread()
        self.xsens_worker = xsens_server(ip=self.xsens_args["ip"],port=self.xsens_args["port"])
        
        # initialize live stream plots
        self.sts_vel_live_stream_plots = []
        if self.gui_args["on"] and self.gui_args["vel"]:
            self.sts_vel_live_stream_plots.append(self.add_live_stream_plot(live_stream=self.sts_vel_live_stream,sensor_name= f"CoM Vel",unit="m/s",dim=3))
            
        # init CoM point and frame
        if self.gui_args["on"] and self.gui_args["3d"]:
            self.com_frame = self.init_frame(pos=np.array([[0,0,1]]),rot=SO3().R*0.1)
            
    def init_logger(self):
        # init response label
        self.logger_label = self.init_response_label(size=[600,300],fontsize=50)
        self.logger_thread = QThread()
        self.logger_worker = sts_logger(init_args = self.init_args["init_flags"],
                                        session_data  = self.session_data)
    def init_shortcuts(self):
        # to start / stop logging at button press
        if self.log_args["on"]:
            # to start logging
            start_log = QShortcut("S", self)
            start_log.activated.connect(self.logger_worker.start_worker)
            # to stop logging and close everything
            stop_log = QShortcut("enter", self)
            stop_log.activated.connect(self.logger_worker.reset_logger)
            # to redo previous take
            redo_log = QShortcut("R", self)
            redo_log.activated.connect(self.logger_worker.redo_take)
        close = QShortcut("Q", self)
        close.activated.connect(self.gui_timer.stop)
        close.activated.connect(self.close_workers)
    def init_IOs(self):
        """
        Init IO thread and worker
        """
        if self.corc_args["on"]:
            self.init_corc()
        if self.xsens_args["on"]:
            self.init_xsens()
        if self.log_args["on"]:
            self.init_logger()
            self.logger_response = {
            "print_text":"Idle\n",
            "logger_fps":0.0
        }
        
        """
        Move sensor worker to thread 
        Connect intersensor signals
        Connect signals to sensor GUI callbacks
        Connect thread start to start sensor worker, update number of open threads
        Connect worker stop to stop thread at closeup
        Connect thread finished to close app
        """
        if self.corc_args["on"]:
            # move worker to thread
            self.corc_worker.moveToThread(self.corc_thread)
            # connect to gui
            self.corc_worker.data_ready.connect(self.update_corc,type=Qt.ConnectionType.QueuedConnection)
            # connect thread start to start worker   
            self.corc_thread.started.connect(self.corc_worker.start_worker)
            self.corc_thread.started.connect(self.add_opened_threads)
            # connect worker stop to stop thread at closeup
            self.corc_worker.stopped.connect(self.corc_thread.exit)
            # connect thread finished to close app
            self.corc_thread.finished.connect(self.close_app)
        if self.xsens_args["on"]:
            # move worker to thread
            self.xsens_worker.moveToThread(self.xsens_thread)
            # connect to gui
            self.xsens_worker.data_ready.connect(self.update_xsens,type=Qt.ConnectionType.QueuedConnection)
            # connect thread start to start worker   
            self.xsens_thread.started.connect(self.xsens_worker.start_worker)
            self.xsens_thread.started.connect(self.add_opened_threads)
            # connect worker stop to stop thread at closeup
            self.xsens_worker.stopped.connect(self.xsens_thread.exit)
            # connect thread finished to close app
            self.xsens_thread.finished.connect(self.close_app)
        if self.log_args["on"]:
            # move worker to thread
            self.logger_worker.moveToThread(self.logger_thread)
            # connect corc and xsens to logger
            if self.corc_args["on"]:
                self.corc_worker.data_ready.connect(self.logger_worker.update_corc,type=Qt.ConnectionType.QueuedConnection)
            if self.xsens_args["on"]:
                self.xsens_worker.data_ready.connect(self.logger_worker.update_xsens,type=Qt.ConnectionType.QueuedConnection)
            # connect logger to sensor gui
            self.logger_worker.time_ready.connect(self.update_logger,type=Qt.ConnectionType.QueuedConnection)
            # connect thread start to add opened threads  
            self.logger_thread.started.connect(self.add_opened_threads)
            # connect worker stop to stop thread at closeup
            self.logger_worker.stopped.connect(self.logger_thread.exit)
            # connect worker collect finish to stop app
            self.logger_worker.collect_finish.connect(self.gui_timer.stop)
            self.logger_worker.collect_finish.connect(self.close_workers)
            # connect thread finished to close app
            self.logger_thread.finished.connect(self.close_app)

        """
        Start threads
        """
        if self.corc_args["on"]:
            self.corc_thread.start()
            self.corc_thread.setPriority(QThread.Priority.TimeCriticalPriority)
        if self.xsens_args["on"]:
            self.xsens_thread.start()
            self.xsens_thread.setPriority(QThread.Priority.TimeCriticalPriority)
        if self.log_args["on"]:
            self.logger_thread.start()
            self.logger_thread.setPriority(QThread.Priority.TimeCriticalPriority)
        
    """
    Update Sensor Variable Callbacks
    """
    @Slot(dict)
    def update_corc(self,corc_response):
        self.corc_response = corc_response
    def update_xsens(self,xsens_response):
        self.xsens_response = xsens_response
    @Slot(dict)
    def update_logger(self,logger_response):
        self.logger_response = logger_response

    """
    Update Main GUI Helper Functions and Callback
    """
    def update_gui(self):
        super().update_gui()
        if self.gui_args["on"]:
            # update live stream time buffer
            if self.gui_args["vel"]:
                self.update_live_stream_buffer(live_stream=self.sts_vel_live_stream)
        if hasattr(self, 'corc_response'):
            # update rft info
            corc_data = self.corc_response["raw_data"]
            fps = self.corc_response["corc_fps"]
            txt = ""
            self.update_response_label(self.corc_label,f"FPS:{fps}\nCORC Running Time:{corc_data[0]}s\n{txt}")

        if hasattr(self, 'xsens_response'):
            # update rft info
            timecode = self.xsens_response["timecode"]
            fps = self.xsens_response["xsens_fps"]
            right_list = self.xsens_response["right"]["list"]
            left_list = self.xsens_response["left"]["list"]             
            
            # update the gui
            CoM = self.xsens_response["CoM"]
            if self.gui_args["on"] and self.gui_args["vel"]:
                try:
                    self.update_live_stream_plot(self.sts_vel_live_stream,self.sts_vel_live_stream_plots[0],CoM[3:6],dim=3)
                except:
                    print(CoM)
            
            # update COM point and frame
            if self.gui_args["on"] and self.gui_args["3d"]:
                try:
                    self.update_frame(self.com_frame,np.array(CoM[0:3]),SO3().R*0.1)
                except:
                    print(CoM)
                
            txt = f"XSENS Timecode:{timecode}\n"

            for i,key in  enumerate(['trunk_ie','trunk_aa','trunk_fe',
                        'clav_dep_ev','clav_prot_ret',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps','wrist_fe','wrist_dev']):
                txt += f"{key:15}: {left_list[i]:8.4f} {right_list[i]:8.4f}\n"
            txt += f"CoM Position\t: {CoM[0:3]}\n"
            txt += f"CoM Velocity\t: {CoM[3:6]}\n"
            self.update_response_label(self.xsens_label,f"FPS:{fps}\n{txt}")

        if hasattr(self, 'logger_response'):
            print_text = self.logger_response["print_text"]
            fps = self.logger_response["logger_fps"]
            self.update_response_label(self.logger_label,f"{print_text}")
    
    """
    Cleanup
    """
    def close_worker_thread(self,worker):
        QMetaObject.invokeMethod(worker, "stop", Qt.ConnectionType.QueuedConnection)
    def close_workers(self):
        if hasattr(self, 'corc_response'):
            self.close_worker_thread(self.corc_worker)
        if hasattr(self, 'xsens_response'):
            self.close_worker_thread(self.xsens_worker)
        if hasattr(self, 'logger_response'):
            self.close_worker_thread(self.logger_worker)
    @Slot()
    def close_app(self):
        self.num_closed_threads += 1
        if self.num_closed_threads == self.num_opened_threads:
            print("Shutting down app...")
            for plt in self.plt_items:
                plt.clear()
                plt.deleteLater()
            self.close()
            self.deleteLater()

if __name__ == "__main__":
    try:
        argv = sys.argv[1]
    except:
        argv ={
               "init_flags":{"corc":{"on":False,
                                     "ip":"127.0.0.1",
                                     "port":2048},
                            "xsens":{"on":True,
                                     "ip":"0.0.0.0",
                                     "port":9764},
                             "gui":{"on":True,
                                    "freq":30,
                                    "3d":True,
                                    "vel":True},
                             "log":{"on":True}
                             },
                "session_data":{
                    "exp_id":"exp2",
                    "subject_id":"test2",
                    "take_num":0,
                }
               }
        
        argv = json.dumps(argv)

    init_args = json.loads(argv)
    app = QtWidgets.QApplication(sys.argv)
    w = sts_gui(init_args)
    
    title_str = f"sts-CORC-{init_args['session_data']['exp_id']}"+"-"
    title_str += init_args['session_data']['subject_id']+"-"
    w.setWindowTitle(title_str)
    w.show()

    app.exec()
    app.quit()
    import shiboken6
    shiboken6.delete(app)  # Immediate deletion
    sys.exit()


    # 169.254.45.31

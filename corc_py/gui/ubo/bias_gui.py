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
from bias_logger import bias_logger
print("Importing GUI")
from base.pycorc_gui import pycorc_gui
print("Importing IOs")
from pycorc_io.corc.corc_FLNL_client import corc_FLNL_client
from pycorc_io.xsens.xsens_server import xsens_server
from pycorc_io.xsens.ub_pckg.ub import ub
from pycorc_io.package_utils.unpack_json import get_subject_params
print("Importing PySide6")
from PySide6 import QtWidgets
from PySide6.QtGui import QShortcut
from PySide6.QtCore import QThread,Qt,QMetaObject,Slot

NUM_RFT = 3
rft_key = ["clav","ua","fa"]
class bias_gui(pycorc_gui):
    def __init__(self,init_args):
        self.init_args = init_args
        self.corc_args = self.init_args["init_flags"]["corc"]
        self.xsens_args = self.init_args["init_flags"]["xsens"]
        self.gui_args = self.init_args["init_flags"]["gui"]
        self.log_args = self.init_args["init_flags"]["log"]
        self.session_data = self.init_args["session_data"]
        
        body_path = os.path.join(os.path.dirname(__file__), '../../..',f'logs/pycorc_recordings/{self.session_data["exp_id"]}/subject_measurements/{self.session_data["subject_id"]}')
        self.body_params_rbt,self.ft_grav,self.removed_bias = get_subject_params(body_path)
        self.bias_response = np.array([0 for i in range(18)])
        
        for dict_type,param_dict in zip(["BODY PARAMS","FT_GRAV","REMOVED_BIAS"],[self.body_params_rbt,self.ft_grav,self.removed_bias]):
            print()
            print(dict_type)
            for (key,value) in param_dict.items():
                print(f"{key}:{value}")
        print()

        self.num_closed_threads = 0
        self.num_opened_threads = 0
        self.requested_threads = 3

        super().__init__(freq=self.gui_args["freq"],gui_3d=self.gui_args["3d"])

        if self.gui_args["force"]:
            self.ubo_wrenches_live_stream = self.init_live_stream(num_columns=2)
            
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
        # init corc live stream plot
        self.ubo_F_live_stream_plots = []
        self.ubo_M_live_stream_plots = []
        if self.gui_args["on"] and self.gui_args["force"]:
            for i in range(NUM_RFT):
                self.ubo_F_live_stream_plots.append(self.add_live_stream_plot(live_stream=self.ubo_wrenches_live_stream,sensor_name= f"UBO_{i} Force",unit="N",dim=3))
                self.ubo_M_live_stream_plots.append(self.add_live_stream_plot(live_stream=self.ubo_wrenches_live_stream,sensor_name= f"UBO_{i} Moment",unit="Nm",dim=3))
    def init_xsens(self):
        # init response label
        self.xsens_label = self.init_response_label(size=[300,200])
        # init corc thread and worker
        self.xsens_thread = QThread()
        self.xsens_worker = xsens_server(ip=self.xsens_args["ip"],port=self.xsens_args["port"])
    def init_skeleton(self):
        # init ub model
        self.skeleton = {
            "right":{},
            "left":{}
        }
        # init xsens skeleton for future use
        for side,color in zip(["right","left"],["purple", "orange"]):
            self.skeleton[side]["ub_xsens"] = ub(self.body_params_rbt,model="ubo",arm_side=side)
            robot_joints, robot_ee = self.skeleton[side]["ub_xsens"].ub_fkine([0]*12)
            
            if self.gui_args["on"] and self.gui_args["3d"]:
                if self.xsens_args["on"]:
                    body = self.init_line(points=robot_joints.t,color=color)
                    ees = [self.init_frame(pos=ee_pose.t,rot=ee_pose.R*0.1) for ee_pose in robot_ee]
                    self.skeleton[side]["body"] = body
                    self.skeleton[side]["ees"] = ees
    
                    if self.corc_args["on"]:
                        if side == "right":
                            forces = [0] + [self.init_line(points=np.vstack([ee_pose.t, ee_pose.t + np.matmul(ee_pose.R,np.array([0,0,0.2]))]),color="pink") for ee_pose in robot_ee[1:]]
                        else:
                            forces = [0 for ee_pose in robot_ee]
                    else:
                        forces = [0 for ee_pose in robot_ee]
                    self.skeleton[side]["forces"] = forces
    def init_logger(self):
        # init response label
        self.logger_label = self.init_response_label(size=[250,150])
        self.logger_thread = QThread()
        self.logger_worker = bias_logger(init_args = self.init_args["init_flags"],
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
        close = QShortcut("Q", self)
        close.activated.connect(self.gui_timer.stop)
        close.activated.connect(self.close_workers)
        
    def init_IOs(self):
        """
        Init IO thread and worker
        """
        self.init_skeleton()
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
            self.logger_worker.bias_ready.connect(self.update_bias,type=Qt.ConnectionType.QueuedConnection)
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
    def update_bias(self,bias_response):
        self.bias_response = bias_response["bias"]

    """
    Update Main GUI Helper Functions and Callback
    """
    def update_gui(self):
        super().update_gui()
        if self.gui_args["on"]:
            # update live stream time buffer
            if self.gui_args["force"]:
                self.update_live_stream_buffer(live_stream=self.ubo_wrenches_live_stream)
        if hasattr(self, 'corc_response'):
            # update rft info
            corc_data = self.corc_response["raw_data"]
            fps = self.corc_response["corc_fps"]
            txt = ""

            # get the orientation of the sensors if exist:
            if hasattr(self, 'xsens_response'):
                robot_joints, robot_ee = self.skeleton["right"]["ub_xsens"].ub_fkine(self.xsens_response["right"]["list"])
            
            for i in range(NUM_RFT):
                # add back the initial removed sensor base bias and remove the subsequent installation bias obtained from the logger
                cur_removed_bias = np.array(self.removed_bias[rft_key[i]])
                force_data = np.array(corc_data[1+i*6:1+i*6+6])+cur_removed_bias-self.bias_response[i*6:i*6+6]
                
                # weight compensate with if robot_ee exist
                if hasattr(self, 'xsens_response'):
                    pose = robot_ee[i + 1]
                    weight_comp = [x * self.ft_grav[rft_key[i]] for x in [0,0,-1]]
                    force_data = force_data - np.array(list(np.matmul(pose.R.T, np.array(weight_comp))) + [0,0,0])

                txt += f"UBO_{i} Force(N): {force_data[0]:8.4f} {force_data[1]:8.4f} {force_data[2]:8.4f} | Moment(Nm): {force_data[3]:8.4f} {force_data[4]:8.4f} {force_data[5]:8.4f}\n"
                txt += f"Force Mag{np.linalg.norm(force_data[:3]):8.4f} N | Moment Mag:{np.linalg.norm(force_data[3:]):8.4f} Nm\n"

                if self.gui_args["on"] and self.gui_args["force"]:
                    self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_F_live_stream_plots[i],force_data,dim=3)
                    self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_M_live_stream_plots[i],force_data[3:],dim=3)
            self.update_response_label(self.corc_label,f"FPS:{fps}\nCORC Running Time:{corc_data[0]}s\n{txt}")

        if hasattr(self, 'xsens_response'):
            # update rft info
            timecode = self.xsens_response["timecode"]
            fps = self.xsens_response["xsens_fps"]
            right_list = self.xsens_response["right"]["list"]
            left_list = self.xsens_response["left"]["list"]

            txt = f"XSENS Timecode:{timecode}\n"

            for i,key in  enumerate(['trunk_ie','trunk_aa','trunk_fe',
                        'clav_dep_ev','clav_prot_ret',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps','wrist_fe','wrist_dev']):
                txt += f"{key:15}: {left_list[i]:8.4f} {right_list[i]:8.4f}\n"
            self.update_response_label(self.xsens_label,f"FPS:{fps}\n{txt}")

            if self.gui_args["on"] and self.gui_args["3d"]:
                for side in ["right","left"]:
                    robot_joints, robot_ee = self.skeleton[side]["ub_xsens"].ub_fkine(self.xsens_response[side]["list"])                        
                    self.update_line(self.skeleton[side]["body"],points=robot_joints.t)
                    
                    for i,(frame,force,pose) in enumerate(zip(self.skeleton[side]["ees"],self.skeleton[side]["forces"],robot_ee)):
                        self.update_frame(frame,pos=pose.t,rot=pose.R*0.1)
                        if i != 0 and side == "right" and hasattr(self, 'corc_response'):
                            j = i - 1
                            # add back the initial removed sensor base bias and remove the subsequent installation bias obtained from the logger
                            net_bias = np.array(self.removed_bias[rft_key[j]])-self.bias_response[j*6:j*6+6]
                            force_data = np.array(corc_data[1+j*6:1+j*6+6])+np.array(net_bias)

                            weight_comp = [x * self.ft_grav[rft_key[j]] for x in [0,0,-1]]
                            force_data = force_data - np.array(list(np.matmul(pose.R.T, np.array(weight_comp))) + [0,0,0])
                            self.update_line(force,points=np.vstack([pose.t, pose.t + np.matmul(pose.R,force_data[:3])*0.02]))
        if hasattr(self, 'logger_response'):
            print_text = self.logger_response["print_text"]
            fps = self.logger_response["logger_fps"]
            self.update_response_label(self.logger_label,f"{print_text}FPS:{fps}")
    
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
        subject_id = input("subject id?: sub<>: ")
        argv ={
               "init_flags":{"corc":{"on":True,
                                     "ip":"127.0.0.1",
                                     "port":2048},
                            "xsens":{"on":True,
                                     "ip":"0.0.0.0",
                                     "port":9764},
                             "gui":{"on":True,
                                    "freq":30,
                                    "3d":True,
                                    "force":False},
                             "log":{"on":True}},
                "session_data":{
                    "exp_id":"exp1",
                    "subject_id":subject_id,
                }
               }
        
        argv = json.dumps(argv)

    init_args = json.loads(argv)
    app = QtWidgets.QApplication(sys.argv)
    w = bias_gui(init_args)
    
    title_str = f"UBO-CORC-{init_args['session_data']['exp_id']}"+"-"
    title_str += init_args['session_data']['subject_id']+"-"
    title_str += "BIAS"+" "
    w.setWindowTitle(title_str)
    w.show()

    app.exec()
    app.quit()
    import shiboken6
    shiboken6.delete(app)  # Immediate deletion
    sys.exit()


    # 169.254.45.31

print("Importing Paths")
import os, sys,json
for mod in [m for m in sys.modules if 'PyQt5' in m]:
    del sys.modules[mod]
    
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)
print("Importing Logger")
from ubo_logger import ubo_logger
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
class ubo_gui(pycorc_gui):
    def __init__(self,init_args):
        self.init_args = init_args
        self.corc_args = self.init_args["init_flags"]["corc"]
        self.xsens_args = self.init_args["init_flags"]["xsens"]
        self.gui_args = self.init_args["init_flags"]["gui"]
        self.log_args = self.init_args["init_flags"]["log"]
        self.session_data = self.init_args["session_data"]
        
        body_path = os.path.join(os.path.dirname(__file__), '../../..',f'logs/pycorc_recordings/{self.session_data["exp_id"]}/subject_measurements/{self.session_data["subject_id"]}')
        self.body_params_rbt,self.ft_grav,self.removed_bias = get_subject_params(body_path)
        self.init_bias = np.load(f"{body_path}/UBOAvgBias.npy")
        self.bias_response = np.array([0 for i in range(18)])
        for dict_type,param_dict in zip(["BODY PARAMS","FT_GRAV","REMOVED_BIAS"],[self.body_params_rbt,self.ft_grav,self.removed_bias]):
            print()
            print(dict_type)
            for (key,value) in param_dict.items():
                print(f"{key}:{value}")
        for i,key in enumerate(rft_key):
            print(f"Init Bias {key}:\t{self.init_bias[i*6:(i+1)*6]}")
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
        self.threads={}
        self.workers={}
        self.labels={}
        self.responses={}
        self.init_IOs()
        self.init_shortcuts()

    """
    Init Sensor Worker / Thread / GUI Helper Functions
    """
    def add_opened_threads(self):
        self.num_opened_threads += 1
        print(f"Threads Opened:{self.num_opened_threads}")
        
        if self.num_opened_threads == self.requested_threads:
            QMetaObject.invokeMethod(self.workers["logger"] , "sensors_ready", Qt.ConnectionType.QueuedConnection)
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
    def init_worker(self,topic):
        label = self.init_response_label(size=[525,200])
        self.labels[topic] = label
        thread = QThread()
        self.threads[topic] = thread
        if topic =="corc":
            worker = corc_FLNL_client(ip=self.corc_args["ip"],port=self.corc_args["port"])
            # init corc live stream plot
            self.ubo_F_live_stream_plots = []
            self.ubo_M_live_stream_plots = []
            if self.gui_args["on"] and self.gui_args["force"]:
                for i in range(NUM_RFT):
                    self.ubo_F_live_stream_plots.append(self.add_live_stream_plot(live_stream=self.ubo_wrenches_live_stream,sensor_name= f"UBO_{i} Force",unit="N",dim=3))
                    self.ubo_M_live_stream_plots.append(self.add_live_stream_plot(live_stream=self.ubo_wrenches_live_stream,sensor_name= f"UBO_{i} Moment",unit="Nm",dim=3))
        elif topic == "xsens":
            worker = xsens_server(ip=self.xsens_args["ip"],port=self.xsens_args["port"])
        elif topic == "logger":
            worker = ubo_logger(init_args=self.init_args["init_flags"],session_data=self.session_data)
        self.workers[topic] = worker
    
    def init_shortcuts(self):
        # to start / stop logging at button press
        if self.log_args["on"]:
            # to start logging
            start_log = QShortcut("S", self)
            start_log.activated.connect(self.workers["logger"].start_worker)
            # to stop logging and close everything
            next_log = QShortcut("enter", self)
            next_log.activated.connect(self.workers["logger"].reset_logger)
            # to redo previous take
            redo_log = QShortcut("R", self)
            redo_log.activated.connect(self.workers["logger"].redo_take)
        close = QShortcut("Q", self)
        close.activated.connect(self.gui_timer.stop)
        close.activated.connect(self.close_workers)
        
    def init_IOs(self):
        """
        Init IO thread and worker
        """
        self.init_skeleton()
        if self.corc_args["on"]:
            self.init_worker("corc")
        if self.xsens_args["on"]:
            self.init_worker("xsens")
        if self.log_args["on"]:
            self.init_worker("logger")
            self.responses["logger"] = {"print_text":"Idle\n",
                                        "fps":0.0}
        
        """
        Move sensor worker to thread 
        Connect intersensor signals
        Connect thread start to start sensor worker, update number of open threads
        Connect worker stop to stop thread at closeup
        Connect thread finished to close app
        Connect signals to sensor GUI callbacks
        """
        for topic in self.workers:
            # move worker to thread
            self.workers[topic].moveToThread(self.threads[topic])
            # connect thread start to start worker   
            self.threads[topic].started.connect(self.add_opened_threads)
            self.threads[topic].started.connect(self.workers[topic].start_worker)
            # connect worker stop to stop thread at closeup
            self.workers[topic].stopped.connect((self.threads[topic]).exit)
            # connect thread finished to close app
            self.threads[topic].finished.connect(self.close_app)
            # connect to gui
            self.workers[topic].data_ready.connect(self.update_response,type=Qt.ConnectionType.QueuedConnection)
            
            if self.log_args["on"] and topic == "logger":
                # connect corc and xsens to logger
                if self.corc_args["on"]:
                    self.workers["corc"].data_ready.connect(self.workers[topic].update_corc,type=Qt.ConnectionType.QueuedConnection)
                if self.xsens_args["on"]:
                    self.workers["xsens"].data_ready.connect(self.workers[topic].update_xsens,type=Qt.ConnectionType.QueuedConnection)
                # connect bias collection to sensor gui
                self.workers[topic].bias_ready.connect(self.update_bias,type=Qt.ConnectionType.QueuedConnection)
                # connect logger collect finish to stop app
                self.workers[topic].collect_finish.connect(self.gui_timer.stop)
                self.workers[topic].collect_finish.connect(self.close_workers)
        """
        Start threads
        """
        for topic in self.workers:
            self.threads[topic].start()
            self.threads[topic].setPriority(QThread.Priority.TimeCriticalPriority)
    """
    Update Sensor Variable Callbacks
    """
    @Slot(dict)
    def update_response(self,response):
        self.responses[response["topic"]] = response
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
        for topic in self.workers:
            if topic in self.responses: # check if key exists
                if topic == "corc":
                    # update rft info
                    corc_data = self.responses[topic]["raw_data"]
                    fps = self.responses[topic]["fps"]
                    txt = ""
        
                    # get the orientation of the sensors if exist:
                    if "xsens" in self.responses:
                        robot_joints, robot_ee = self.skeleton["right"]["ub_xsens"].ub_fkine(self.responses["xsens"]["right"]["list"])
                    
                    for i in range(NUM_RFT):
                        # add back the initial removed sensor base bias and remove the subsequent installation bias obtained from the logger
                        cur_removed_bias = np.array(self.removed_bias[rft_key[i]])
                        force_data = np.array(corc_data[1+i*6:1+i*6+6])+cur_removed_bias-self.init_bias[i*6:i*6+6]-self.bias_response[i*6:i*6+6]
                        
                        # weight compensate with if robot_ee exist
                        if "xsens" in self.responses:
                            pose = robot_ee[i + 1]
                            weight_comp = [x * self.ft_grav[rft_key[i]] for x in [0,0,-1]]
                            force_data = force_data - np.array(list(np.matmul(pose.R.T, np.array(weight_comp))) + [0,0,0])
        
                        txt += f"UBO_{i} Force(N): {force_data[0]:8.4f} {force_data[1]:8.4f} {force_data[2]:8.4f} | Moment(Nm): {force_data[3]:8.4f} {force_data[4]:8.4f} {force_data[5]:8.4f}\n"
                        txt += f"Force Mag{np.linalg.norm(force_data[:3]):8.4f} N | Moment Mag:{np.linalg.norm(force_data[3:]):8.4f} Nm\n"
        
                        if self.gui_args["on"] and self.gui_args["force"]:
                            self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_F_live_stream_plots[i],force_data,dim=3)
                            self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_M_live_stream_plots[i],force_data[3:],dim=3)
                    self.update_response_label(self.labels[topic],f"FPS:{fps}\nCORC Running Time:{corc_data[0]}s\n{txt}")
        
                elif topic == "xsens":
                    # update rft info
                    timecode = self.responses[topic]["timecode"]
                    fps = self.responses[topic]["fps"]
                    right_list = self.responses[topic]["right"]["list"]
                    left_list = self.responses[topic]["left"]["list"]
        
                    txt = f"XSENS Timecode:{timecode}\n"
        
                    for i,key in  enumerate(['trunk_ie','trunk_aa','trunk_fe',
                                'clav_dep_ev','clav_prot_ret',
                                'shoulder_fe','shoulder_aa','shoulder_ie',
                                'elbow_fe','elbow_ps','wrist_fe','wrist_dev']):
                        txt += f"{key:15}: {left_list[i]:8.4f} {right_list[i]:8.4f}\n"
                    self.update_response_label(self.labels[topic],f"FPS:{fps}\n{txt}")
        
                    if self.gui_args["on"] and self.gui_args["3d"]:
                        for side in ["right","left"]:
                            robot_joints, robot_ee = self.skeleton[side]["ub_xsens"].ub_fkine(self.responses[topic][side]["list"])                        
                            self.update_line(self.skeleton[side]["body"],points=robot_joints.t)
                            
                            for i,(frame,force,pose) in enumerate(zip(self.skeleton[side]["ees"],self.skeleton[side]["forces"],robot_ee)):
                                self.update_frame(frame,pos=pose.t,rot=pose.R*0.1)
                                if i != 0 and side == "right" and "corc" in self.responses:
                                    j = i - 1
                                    # add back the initial removed sensor base bias and remove the subsequent installation bias obtained from the logger
                                    net_bias = np.array(self.removed_bias[rft_key[j]])-self.bias_response[j*6:j*6+6]-self.init_bias[j*6:j*6+6]
                                    force_data = np.array(corc_data[1+j*6:1+j*6+6])+np.array(net_bias)
        
                                    weight_comp = [x * self.ft_grav[rft_key[j]] for x in [0,0,-1]]
                                    force_data = force_data - np.array(list(np.matmul(pose.R.T, np.array(weight_comp))) + [0,0,0])
                                    self.update_line(force,points=np.vstack([pose.t, pose.t + np.matmul(pose.R,force_data[:3])*0.02]))
                elif topic == "logger":
                    print_text = self.responses[topic]["print_text"]
                    fps = self.responses[topic]["fps"]
                    self.update_response_label(self.labels[topic],f"{print_text}\nFPS:{fps}")
    """
    Cleanup
    """
    def close_worker_thread(self,worker):
        QMetaObject.invokeMethod(worker, "stop", Qt.ConnectionType.QueuedConnection)
    def close_workers(self):
        for topic in self.workers:
            self.close_worker_thread(self.workers[topic])
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

                             "log":{"on":True}
                             },
                "session_data":{
                    "exp_id":"exp1",
                    "patient_id":"p1",
                    "subject_id":"test",
                    "var_id":"var_2",
                    "take_num":0,
                }
               }
        
        argv = json.dumps(argv)

    init_args = json.loads(argv)
    app = QtWidgets.QApplication(sys.argv)
    w = ubo_gui(init_args)
    
    title_str = f"UBO-CORC-{init_args['session_data']['exp_id']}"+"-"
    title_str += init_args['session_data']['patient_id']+"-"
    title_str += init_args['session_data']['subject_id']+"-"
    title_str += init_args['session_data']['var_id']+" "
    w.setWindowTitle(title_str)
    w.show()

    app.exec()
    app.quit()
    import shiboken6
    shiboken6.delete(app)  # Immediate deletion
    sys.exit()


    # 169.254.45.31

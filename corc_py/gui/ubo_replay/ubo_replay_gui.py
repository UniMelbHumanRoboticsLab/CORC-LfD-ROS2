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
print("Importing Replayer")
from ubo_replayer import ubo_replayer
print("Importing GUI")
from base.pycorc_gui import pycorc_gui
print("Importing IOs")
from pycorc_io.xsens.ub_pckg.ub import ub
from pycorc_io.package_utils.unpack_json import get_subject_params
print("Importing PySide6")
from PySide6 import QtWidgets
from PySide6.QtGui import QShortcut
from PySide6.QtCore import QThread,Qt,QMetaObject,Slot

NUM_RFT = 3
rft_key = ["clav","ua","fa"]
class ubo_replay_gui(pycorc_gui):
    def __init__(self,init_args):
        self.init_args = init_args
        self.corc_args = self.init_args["init_flags"]["corc"]
        self.xsens_args = self.init_args["init_flags"]["xsens"]
        self.gui_args = self.init_args["init_flags"]["gui"]
        self.replay_args = self.init_args["init_flags"]["replay"]
        self.session_data = self.init_args["session_data"]

        self.num_closed_threads = 0
        self.num_opened_threads = 0

        self.place_holder_angles ={
                            'trunk_ie':0,
                            'trunk_aa':0,
                            'trunk_fe':0,
                            'clav_dep_ev':0,
                            'clav_prot_ret':0,
                            'shoulder_fe':0,
                            'shoulder_aa':0,
                            'shoulder_ie':0,
                            'elbow_fe':0,
                            'elbow_ps':-170,
                            'wrist_fe':0,
                            'wrist_dev':0
                        }
        self.body_params_rbt,self.ft_grav,self.removed_bias = get_subject_params(os.path.join(os.path.dirname(__file__), '../../..',f'logs/pycorc_recordings/{self.session_data["exp_id"]}/subject_measurements/{self.session_data["subject_id"]}'))
        
        for dict_type,param_dict in zip(["BODY PARAMS","FT_GRAV","REMOVED_BIAS"],[self.body_params_rbt,self.ft_grav,self.removed_bias]):
            print()
            print(dict_type)
            for (key,value) in param_dict.items():
                print(f"{key}:{value}")
        print()
        
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
    def init_corc(self):
        # init response label
        self.corc_label = self.init_response_label(size=[525,100])
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
    def init_skeleton(self):
        self.skeleton = {
            "right":{},
            "left":{}
        }
        # init xsens skeleton for future use
        self.hand_traj = self.init_line(points=np.vstack(([0,0,0],[1,1,1])),color="white")
        for side,color in zip(["right","left"],["purple", "orange"]):
            self.skeleton[side]["ub_xsens"] = ub(self.body_params_rbt,model="ubo",arm_side=side)
            robot_joints, robot_ee = self.skeleton[side]["ub_xsens"].ub_fkine(list(self.place_holder_angles.values()))
            if self.gui_args["on"] and self.gui_args["3d"]:
                body = self.init_line(points=robot_joints.t,color=color)
                ees = [self.init_frame(pos=ee_pose.t,rot=ee_pose.R*0.03) for ee_pose in robot_ee]
                self.skeleton[side]["body"] = body
                self.skeleton[side]["ees"] = ees
                if self.corc_args["on"]:
                    if side == "right":
                        forces = [0] + [self.init_line(points=np.vstack([ee_pose.t, ee_pose.t + np.matmul(ee_pose.R,np.array([0,0,0.2]))]),color="white") for ee_pose in robot_ee[1:]]
                    else:
                        forces = [0 for ee_pose in robot_ee]
                else:
                    forces = [0 for ee_pose in robot_ee]
                self.skeleton[side]["forces"] = forces
    def init_replayer(self):
        # init response label
        self.replayer_label = self.init_response_label(size=[250,150])
        self.replayer_thread = QThread()
        self.replayer_worker = ubo_replayer(init_args = self.init_args["init_flags"],
                                                session_data  = self.session_data)
    def init_shortcuts(self):
        # to start / stop logging at button press
        if self.replay_args["on"]:
            # jump right
            jump_r = QShortcut("3", self)
            jump_r.activated.connect(self.replayer_worker.jump_r)
            # jump left
            jump_l = QShortcut("9", self)
            jump_l.activated.connect(self.replayer_worker.jump_l)
            
            # to start logging
            start_log = QShortcut("6", self)
            start_log.activated.connect(self.replayer_worker.start_stop_take)
            # to go to next take 
            next_log = QShortcut("2", self)
            next_log.activated.connect(self.replayer_worker.next_take)
            prev_log = QShortcut("8", self)
            prev_log.activated.connect(self.replayer_worker.prev_take)
            # to go to next var
            next_var = QShortcut("1", self)
            next_var.activated.connect(self.replayer_worker.next_var)
            prev_var = QShortcut("7", self)
            prev_var.activated.connect(self.replayer_worker.prev_var)
            # to go to next patient
            next_var = QShortcut("5", self)
            next_var.activated.connect(self.replayer_worker.next_role)
            prev_var = QShortcut("4", self)
            prev_var.activated.connect(self.replayer_worker.prev_role)
            
        close = QShortcut("0", self)
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
        if self.replay_args["on"]:
            self.init_replayer()
            self.replayer_response = {
            "print_text":"Idle\n",
            "replayer_fps":0.0
        }
        
        """
        Move sensor worker to thread 
        Connect intersensor signals
        Connect signals to sensor GUI callbacks
        Connect thread start to start sensor worker, update number of open threads
        Connect worker stop to stop thread at closeup
        Connect thread finished to close app
        """
        if self.replay_args["on"]:
            # move worker to thread
            self.replayer_worker.moveToThread(self.replayer_thread)
            # connect replayer to sensor gui
            self.replayer_worker.time_ready.connect(self.update_replayer,type=Qt.ConnectionType.QueuedConnection)
            # connect hand_traj to sensor gui
            self.replayer_worker.traj_ready.connect(self.update_hand_traj,type=Qt.ConnectionType.QueuedConnection)
            # connect thread start to add opened threads  
            self.replayer_thread.started.connect(self.add_opened_threads)
            # connect replayer finished saving to close workers
            self.replayer_worker.finish_save.connect(self.close_workers)
            # connect worker stop to stop thread at closeup
            self.replayer_worker.stopped.connect(self.replayer_thread.exit)
            # connect thread finished to close app
            self.replayer_thread.finished.connect(self.close_app)

        """
        Start threads
        """
        if self.replay_args["on"]:
            self.replayer_thread.start()
            self.replayer_thread.setPriority(QThread.Priority.TimeCriticalPriority)
        
    """
    Update Sensor Variable Callbacks
    """
    @Slot(dict)
    def update_replayer(self,replayer_response):
        self.replayer_response = replayer_response
        if self.replayer_response["corc"]:
            self.corc_response = self.replayer_response["corc"]
        if self.replayer_response["xsens"]:
            self.xsens_response = self.replayer_response["xsens"]
    @Slot(list)
    def update_hand_traj(self,hand_traj_response):
        self.update_line(self.hand_traj,self.skeleton["right"]["ub_xsens"].ub_model[0].fkine(np.deg2rad(hand_traj_response)).t)
        p=0
            
        
    """
    Update Main GUI Helper Functions and Callback
    """
    def update_gui(self):
        super().update_gui()
        # update live stream time buffer
        if self.gui_args["force"]:
            self.update_live_stream_buffer(live_stream=self.ubo_wrenches_live_stream)
        if hasattr(self, 'corc_response'):
            # update rft info
            corc_data = self.corc_response["raw_data"]
            txt = ""
            # get the orientation of the sensors if exist:
            if hasattr(self, 'xsens_response'):
                robot_joints, robot_ee = self.skeleton["right"]["ub_xsens"].ub_fkine(self.xsens_response["right"])
            else:
                robot_joints, robot_ee = self.skeleton["right"]["ub_xsens"].ub_fkine(list(self.place_holder_angles.values()))

            for i in range(NUM_RFT):
                force_data = np.array(corc_data[1+i*6:1+i*6+6])
                txt += f"UBO_{i} Force(N): {force_data[0]:8.4f} {force_data[1]:8.4f} {force_data[2]:8.4f} | Moment(Nm): {force_data[3]:8.4f} {force_data[4]:8.4f} {force_data[5]:8.4f}\n"
                txt += f"Force Mag{np.linalg.norm(force_data[:3]):8.4f} N | Moment Mag:{np.linalg.norm(force_data[3:]):8.4f} Nm\n"

                if self.gui_args["on"] and self.gui_args["force"]:
                    self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_F_live_stream_plots[i],force_data,dim=3)
                    self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_M_live_stream_plots[i],force_data[3:],dim=3)
            self.update_response_label(self.corc_label,f"CORC Running Time:{corc_data[0]}s\n{txt}")
        if hasattr(self, 'xsens_response'):
            if hasattr(self, 'xsens_response'):
                # update rft info
                right_list = self.xsens_response["right"]
                left_list = self.xsens_response["left"]

                f = ""
                p = "left"
                g = "right"
                txt = f"{self.xsens_response['timecode']}\n{f:15}: {p:8} {g:8}\n"

                for i,key in  enumerate(['trunk_ie','trunk_aa','trunk_fe',
                            'clav_dep_ev','clav_prot_ret',
                            'shoulder_fe','shoulder_aa','shoulder_ie',
                            'elbow_fe','elbow_ps','wrist_fe','wrist_dev']):
                    txt += f"{key:15}: {left_list[i]:8.4f} {right_list[i]:8.4f}\n"
                self.update_response_label(self.xsens_label,f"{txt}")

            if self.gui_args["on"] and self.gui_args["3d"]:
                for side in ["right","left"]:
                    if hasattr(self, 'xsens_response'):
                        robot_joints, robot_ee = self.skeleton[side]["ub_xsens"].ub_fkine(self.xsens_response[side])
                    else:
                        robot_joints, robot_ee = self.skeleton[side]["ub_xsens"].ub_fkine(list(self.place_holder_angles.values()))
                        
                    self.update_line(self.skeleton[side]["body"],points=robot_joints.t)
                    for i,(frame,force,pose) in enumerate(zip(self.skeleton[side]["ees"],self.skeleton[side]["forces"],robot_ee)):
                        self.update_frame(frame,pos=pose.t,rot=pose.R*0.1)
                        if i != 0 and side == "right" and hasattr(self, 'corc_response'):
                            j = i - 1
                            force_data = np.array(corc_data[1+j*6:1+j*6+6])

                            self.update_line(force,points=np.vstack([pose.t, pose.t + np.matmul(pose.R,force_data[:3])*0.02]))
        if hasattr(self, 'replayer_response'):
            print_text = self.replayer_response["print_text"]
            fps = self.replayer_response["replayer_fps"]
            self.update_response_label(self.replayer_label,f"{print_text}FPS:{fps}")
    
    """
    Cleanup
    """
    def close_worker_thread(self,worker):
        QMetaObject.invokeMethod(worker, "stop", Qt.ConnectionType.QueuedConnection)
    def close_workers(self):
        if hasattr(self, 'replayer_response'):
            self.close_worker_thread(self.replayer_worker)
    @Slot()
    def close_app(self):
        self.num_closed_threads += 1
        if self.num_closed_threads == self.num_opened_threads:
            print("Shutting down app...")
            # clear all plots
            for plt in self.plt_items:
                plt.clear()
                plt.deleteLater()
            self.close()
            self.deleteLater()

if __name__ == "__main__":
    try:
        argv = sys.argv[1]
    except:
        var = 6
        argv ={
               "init_flags":{"corc" :{"on":True},
                            "xsens" :{"on":True},
                            "gui"   :{
                                    "on":True,
                                    "freq":60,
                                    "3d":True,
                                    "force":False},
                            "replay":{"on":True}
                             },
                "session_data":{
                    "exp_id":"exp1",
                    "patient_id":"p1",
                    "subject_id":"sub3",
                    "var_id":"var_1",
                    "take_num":3,
                }
               }
        
        argv = json.dumps(argv)

    init_args = json.loads(argv)
    app = QtWidgets.QApplication(sys.argv)
    w = ubo_replay_gui(init_args)
    title_str = f"UBO-CORC-{init_args['session_data']['exp_id']}"+"-"
    title_str += init_args['session_data']['patient_id']+"-"
    title_str += init_args['session_data']['subject_id']
    w.setWindowTitle(title_str)
    w.show()

    app.exec()
    app.quit()
    import shiboken6
    shiboken6.delete(app)  # Immediate deletion
    sys.exit()

    # 169.254.45.31
"""PySide6 application for CORC ROS2"""
import os, sys  
import yaml
import signal

# import global local ROS2 packages
from corc_interfaces.msg import SyncData

# import custom libraries within local ROS2 package
print("Importing Listeners")
sys.path.append(os.path.join(os.path.dirname(__file__)))
# import extra custom libraries within this ROS2 package but in other folders (need to include __init__.py for each new folder)
from ros_io.ros_listeners import ros_listener

# import extra custom libraries outside local ROS2 package (don't need to include __init__.py for each new folder)
print("Importing GUI")
from ament_index_python.packages import get_package_share_directory
package_share = get_package_share_directory('sm_py_nodes')
# get workspace root to grab the custom libraries
if "CORC-LfD-ROS2" in package_share:
    workspace_root = package_share[:package_share.index("CORC-LfD-ROS2") + len("CORC-LfD-ROS2")]
    sys.path.append(workspace_root)
from corc_py.pycorc_io.package_utils.unpack_json import get_subject_params
from corc_py.pycorc_io.xsens.ub_pckg.ub import ub
from corc_py.pycorc_io.corc.emu import emu_robot
from corc_py.gui.base.pycorc_gui import pycorc_gui

# import virtual environment libraries
print("Importing PySide6")
from PySide6 import QtWidgets
from PySide6.QtGui import QShortcut
from PySide6.QtCore import QThread,Qt,QMetaObject,Slot
import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

class corc_viz(pycorc_gui):
    """
    Initilization
    """
    def __init__(self,freq=30,open_gui=True,conf="lfd_corc_node"):
        super().__init__(freq=freq,gui_3d=open_gui)
        self.setWindowTitle("Live Sensor Plot")

        self.num_closed_threads = 0
        self.num_opened_threads = 0
        
        self.ros_threads={}
        self.ros_workers={}
        self.ros_labels={}
        self.ros_responses={}
        
        # Load YAML config
        self.config = self.load_yaml(conf)
        self.session_param = {
            "exp_id":self.config["exp_id"],
            "patient_id":self.config["patient_id"],
            "sub_id":self.config["sub_id"],
            }
        self.setWindowTitle(f"{self.session_param['exp_id']}-{self.session_param['patient_id']}-{self.session_param['sub_id']}")
        self.session_param["directory"] = f'{workspace_root}/logs/ros2_recordings/{self.session_param['exp_id']}/{self.session_param['patient_id']}/{self.session_param['sub_id']}/'
        self.body_params,self.ft_grav,self.removed_bias = get_subject_params(f'{workspace_root}/logs/ros2_recordings/{self.session_param['exp_id']}/subject_measurements/{self.session_param['sub_id']}/',"emu")
        
        self.init_IOs()
        self.init_shortcuts()
        signal.signal(signal.SIGINT, self.signal_handler)
    """
    Init Sensor Worker / Thread / GUI Helper Functions
    """
    def add_opened_threads(self):
        self.num_opened_threads += 1
        print(f"Threads Opened:{self.num_opened_threads}")
    def init_shortcuts(self):
        close = QShortcut("Q", self)
        close.activated.connect(self.gui_timer.stop)
        close.activated.connect(self.close_workers)
    def init_ros_listener(self,topic,msg_type):
        # init response label
        self.ros_labels[topic] = self.init_response_label(size=[525,200])
        # init corc thread and worker
        self.ros_threads[topic] = QThread()
        self.ros_workers[topic] = ros_listener(topic,msg_type)
    def init_twins(self):
        # init emu model
        self.emu = {
            }
        # init ub model
        self.skeleton = {
            "right":{},
            "left":{}
        }
        # init xsens skeleton
        for side,color in zip(["right","left"],["purple", "orange"]):
            self.skeleton[side]["ub"] = ub(self.body_params,model="emu",arm_side=side)
            ub_segments, ub_ees = self.skeleton[side]["ub"].ub_fkine([0]*12)
            # grab the ground for the upper body and plot it once
            self.init_frame(pos=ub_segments[0].t,rot=ub_segments[0].R*0.1)
            
            body = self.init_line(points=ub_segments.t,color=color)
            ees = [self.init_frame(pos=ee_pose.t,rot=ee_pose.R*0.1) for ee_pose in ub_ees]
            if side == "right":
                forces = [0] + [self.init_line(points=np.vstack([ee_pose.t, ee_pose.t + np.matmul(ee_pose.R,np.array([0,0,0.1]))]),color="yellow") for ee_pose in ub_ees[1:]]
            else:
                forces = [0 for ee_pose in ub_ees]
            self.skeleton[side]["body"] = body
            self.skeleton[side]["ees"] = ees
            self.skeleton[side]["forces"] = forces
        # init EMU robot
        self.emu["emu"] = emu_robot()
        emu_segments, emu_ees = self.emu["emu"].emu_fkine([0]*4)
        self.init_frame(pos=emu_segments[0].t,rot=emu_segments[0].R*0.1)
        
        body = self.init_line(points=emu_segments.t,color=color)
        ees = [self.init_frame(pos=ee_pose.t,rot=ee_pose.R*0.1) for ee_pose in emu_ees]
        forces = [self.init_line(points=np.vstack([ee_pose.t, ee_pose.t + np.matmul(ee_pose.R,np.array([0,0,0.1]))]),color="yellow") for ee_pose in emu_ees]
        self.emu["body"] = body
        self.emu["ees"] = ees
        self.emu["forces"] = forces
    def init_IOs(self):
        self.init_twins()
        """
        Init IO thread and worker
        """
        self.init_ros_listener("sync_data",SyncData)
        """
        Move sensor worker to thread 
        Connect thread start to start sensor worker
        Update number of open threads
        Connect worker stop to stop thread at closeup
        Connect thread finished to close app
        Connect signals to sensor GUI callbacks
        """
        for topic in self.ros_workers:
            # move worker to thread
            self.ros_workers[topic].moveToThread(self.ros_threads[topic])
            # connect thread start to start worker   
            self.ros_threads[topic].started.connect(self.add_opened_threads)
            self.ros_threads[topic].started.connect(self.ros_workers[topic].start_worker)
            # connect worker stop to stop thread at closeup
            self.ros_workers[topic].stopped.connect((self.ros_threads[topic]).exit)
            # connect thread finished to close app
            self.ros_threads[topic].finished.connect(self.close_app)
            # connect to gui
            self.ros_workers[topic].data_ready.connect(self.update_response,type=Qt.ConnectionType.QueuedConnection)
        """
        Start threads
        """
        for topic in self.ros_workers:
            self.ros_threads[topic].start()
            self.ros_threads[topic].setPriority(QThread.Priority.TimeCriticalPriority)

    """
    Update ROS Listener Callbacks
    """
    @Slot(dict)
    def update_response(self,ros_response):
        self.ros_responses[ros_response["topic"]] = ros_response
    def update_gui(self):
        super().update_gui()
        for topic in self.ros_workers:
            if topic in self.ros_responses: # check if key exists
                # update ros_worker labels
                msg = self.ros_responses[topic]["msg"]
                fps = self.ros_responses[topic]["listener_fps"]
                txt = f"{topic} FPS:{fps}\n"
                
                if isinstance(msg,SyncData):
                    txt += f"{msg.sync_id}\n"
                    txt += f"Var {msg.var_num} Rep {msg.rep_num} Sbmvmt {msg.sbmvmt_num} \nElapsed: {msg.timestamp}"
                    for side in ["right","left"]:
                        if side == "right":
                            ub_segments, ub_ees = self.skeleton[side]["ub"].ub_fkine(np.array(msg.right.position))
                        else:
                            ub_segments, ub_ees = self.skeleton[side]["ub"].ub_fkine(np.array(msg.left.position))
                        self.update_line(self.skeleton[side]["body"],points=ub_segments.t)
                        
                        for i,(frame,force,pose) in enumerate(zip(self.skeleton[side]["ees"],self.skeleton[side]["forces"],ub_ees)):
                            self.update_frame(frame,pos=pose.t,rot=pose.R*0.1)
                            if i != 0 and side == "right":
                                # # add back the initial removed sensor base bias and remove the subsequent gravity
                                # net_bias = np.array(self.removed_bias["fa"])
                                # force_data = np.array(msg.wrench)+np.array(net_bias)
    
                                # weight_comp = [x * self.ft_grav["fa"] for x in [0,0,-1]]
                                # force_data = force_data - np.array(list(np.matmul(pose.R.T, np.array(weight_comp))) + [0,0,0])
                                # self.update_line(force,points=np.vstack([pose.t, pose.t + np.matmul(pose.R,force_data[:3])*0.02]))
                                self.update_line(force,points=np.vstack([pose.t, pose.t + np.array(msg.wrench)*0.02]))

                        emu_segments, emu_ees = self.emu["emu"].emu_fkine(np.array(msg.joint_position))
                        self.update_line(self.emu["body"],points=emu_segments.t)
                        
                        for i,(frame,force,pose) in enumerate(zip(self.emu["ees"],self.emu["forces"],emu_ees)):
                            self.update_frame(frame,pos=pose.t,rot=pose.R*0.1)
                            self.update_line(force,points=np.vstack([pose.t, pose.t + np.array(msg.wrench)*0.02]))
                                
                self.update_response_label(self.ros_labels[topic],txt)
    """
    Cleanup
    """
    def close_worker_thread(self,worker):
        QMetaObject.invokeMethod(worker, "stop", Qt.ConnectionType.QueuedConnection)
    def close_workers(self):
        for topic in self.ros_workers:
            self.close_worker_thread(self.ros_workers[topic])
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
    """
    Helper Functions
    """
    def load_yaml(self,conf):
        config_file = os.path.join(
            workspace_root, 'config', f'{conf}.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            print(f"{conf} Loaded")
            return config
    def signal_handler(self,sig,frame):# Handle CTRL+C gracefully
        self.close_workers()
        
def main():
    """Main entry point."""
    try:
        # Create Qt application
        app = QtWidgets.QApplication(sys.argv)
        # Create and show window
        window = corc_viz()
        window.show()

        app.exec()
        app.quit()
        import shiboken6
        shiboken6.delete(app)  # Graceful deletion
        sys.exit()
    except Exception as e:
        print(f'Fatal error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()

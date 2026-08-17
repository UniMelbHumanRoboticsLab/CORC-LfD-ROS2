import os,sys
import numpy as np
import pandas as pd

from PySide6.QtCore import QObject, Signal,QTimer,Slot,QElapsedTimer,Qt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pycorc_io.package_utils.unpack_json import get_subject_params
from pycorc_io.xsens.ub_pckg.ub import ub

NUM_RFT = 3
rft_key = ["clav","ua","fa"]
replay = False
class ubo_replayer(QObject):
    time_ready = Signal(dict)
    traj_ready = Signal(list)
    finish_save = Signal()
    stopped = Signal()
    def __init__(self,init_args,session_data):
        super().__init__()
        self.init_args = init_args
        self.exp_id = session_data["exp_id"]
        self.patient_id = session_data["patient_id"]
        self.patient_num = int(self.patient_id[-1])
        self.subject_id = session_data["subject_id"]
        self.var_id = session_data["var_id"]
        self.var_num = int(self.var_id[-1])
        self.take_num = session_data["take_num"]
        self.take_text = "Bias"
        self.subject_path = f"logs/pycorc_recordings/{self.exp_id}/{self.patient_id}/{self.subject_id}"
        
        body_path = os.path.join(os.path.dirname(__file__), '../../..',f'logs/pycorc_recordings/{self.exp_id}/subject_measurements/{self.subject_id}')
        self.body_params_rbt,self.ft_grav,self.removed_bias = get_subject_params(body_path)
        self.init_bias = np.load(f"{body_path}/UBOAvgBias.npy")
        for i,key in enumerate(rft_key):
            print(f"Init Bias {key}:\t{self.init_bias[i*6:(i+1)*6]}")
        print()
        
        
        self.skeleton = ub(self.body_params_rbt,model="ubo",arm_side="right")
        
        # FPS Calculator
        self.replayer_frame_count = 0
        self.replayer_fps = 0
        self.replayer_timer = QElapsedTimer() # use the system clock
        self.replayer_timer.start()
        self.replayer_cur_time = self.replayer_timer.elapsed()
        self.replayer_last_time = self.replayer_timer.elapsed()
        self.elapsed_time = 0
        self.total_frames = 1000

        print("UBO Replayer Started")
        self.read_logged_data()
    def read_logged_data(self):
        self.save_path = os.path.join(os.path.dirname(__file__), '../../..',f"{self.subject_path}/{self.var_id}/raw")
        self.data = pd.read_csv(f"{self.save_path}/UBORecord{self.take_num}Log.csv")
        self.var_bias = np.load(f"{self.save_path}/UBOAvgBias.npy")
        ph_path = os.path.join(os.path.dirname(__file__), '../../..',f"logs/pycorc_recordings/{self.exp_id}/p1/{self.subject_id}/var_1/raw")
        trunk_bias_cur_var = np.array([np.mean(pd.read_csv(f"{self.save_path}/UBORecordBiasLog.csv")[["trunk_ie_right"]].values),
                               np.mean(pd.read_csv(f"{self.save_path}/UBORecordBiasLog.csv")[["trunk_aa_right"]].values),
                               np.mean(pd.read_csv(f"{self.save_path}/UBORecordBiasLog.csv")[["trunk_fe_right"]].values),
                               0,0,0,0,0,0,0,0,0])
        trunk_bias_init = np.array([np.mean(pd.read_csv(f"{ph_path}/UBORecordBiasLog.csv")[["trunk_ie_right"]].values),
                               np.mean(pd.read_csv(f"{ph_path}/UBORecordBiasLog.csv")[["trunk_aa_right"]].values),
                               np.mean(pd.read_csv(f"{ph_path}/UBORecordBiasLog.csv")[["trunk_fe_right"]].values),
                               0,0,0,0,0,0,0,0,0])
        trunk_bias = trunk_bias_cur_var-trunk_bias_init
        trunk_bias = np.array([self.data["trunk_ie_right"].values[0],
                               self.data["trunk_aa_right"].values[0],
                               self.data["trunk_fe_right"].values[0],
                               0,0,0,0,0,0,0,0,0])
        # trunk_bias = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        
        
        # read logged data        
        self.total_frames = len(self.data) if self.init_args["corc"]["on"]==1 else 1000
        self.frame_id = 0
        if self.init_args["corc"]["on"]:
            self.corc_data = self.data[["elapsed_time","F1x", "F1y", "F1z", "T1x", "T1y", "T1z",
                    "F2x", "F2y", "F2z", "T2x", "T2y", "T2z",
                    "F3x", "F3y", "F3z", "T3x", "T3y", "T3z"]].values
            
        if self.init_args["xsens"]["on"]:
            joints = ['trunk_ie','trunk_aa','trunk_fe',
                        'clav_dep_ev','clav_prot_ret',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps',
                        'wrist_fe','wrist_dev']
            self.right_xsens_data = self.data[[f"{joint}_right" for joint in joints]].values-trunk_bias  # last 7 columns are CORC data
            self.left_xsens_data = self.data[[f"{joint}_left" for joint in joints]].values-trunk_bias  # last 7 columns are CORC data
            self.timecode = self.data[["timecode"]].values
        
        self.traj_ready.emit(self.right_xsens_data)
        print(f"\n{self.exp_id}/{self.patient_id}/{self.subject_id}/{self.var_id}/{self.take_num}\nLogged Time: {self.corc_data[-1,0]}")
        print(trunk_bias)
        for i,key in enumerate(rft_key):
            print(f"Variation Bias {key}:\t{self.var_bias[i*6:(i+1)*6]}")
    """
    Replayer Callback
    """
    def return_data(self,print_text):
        """
        process corc data to remove gravity and bias
        """
        corc_data = self.corc_data[self.frame_id,:]
        right = self.right_xsens_data[self.frame_id,:]
        left = self.left_xsens_data[self.frame_id,:]
        robot_ee = self.skeleton.fkine(right)

        processed = [0]
        print_text += f"{self.exp_id}/{self.patient_id}/{self.subject_id}/{self.var_id}\nLogged Time: {self.corc_data[-1,0]}\n"
        print_text += f"Replaying Take {self.take_num}\n"
        print_text += f"Logger Time Elapsed:{corc_data[0]}\n"
        for i in range(NUM_RFT):
            # get initial_bias
            removed_bias = np.array(self.removed_bias[rft_key[i]])
            force_data = np.array(corc_data[1+i*6:1+i*6+6])+np.array(removed_bias)-self.var_bias[i*6:i*6+6]-self.init_bias[i*6:i*6+6]
            
            # weight compensate
            pose = robot_ee[i + 1]
            weight_comp = [x * self.ft_grav[rft_key[i]] for x in [0,0,-1]]
            force_data = force_data - np.array(list(np.matmul(pose.R.T, np.array(weight_comp))) + [0,0,0])
            processed += force_data.tolist()
        processed = np.array(processed)
        
        data = {
            "print_text":print_text,
            "replayer_fps":0,
            "frame_id": self.frame_id,
            "xsens": {
                "timecode":self.timecode[self.frame_id,:],
                "right": right,
                "left": left,
            } if self.init_args["xsens"]["on"] else None ,
            "corc": {"raw_data":processed} if self.init_args["corc"]["on"] else None,
        }
        return data
    def replay_current_data(self):
        self.frame_id += 1
        if self.frame_id >= self.total_frames:
            self.frame_id = 0
            self.replayer_start_time = self.replayer_timer.elapsed()
            if not replay:
                if self.take_num == 4 and self.var_num == 6:
                    self.next_var()
                else:
                    self.next_take()

        # update FPS
        print_text = ""
        self.replayer_frame_count += 1
        self.replayer_cur_time = self.replayer_timer.elapsed()
        self.elapsed_time = (self.replayer_cur_time-self.replayer_start_time)/1000
        

        if self.replayer_cur_time-self.replayer_last_time >= 1000:
            self.replayer_fps = self.replayer_frame_count * 1000 / (self.replayer_cur_time-self.replayer_last_time)
            self.replayer_last_time = self.replayer_cur_time
            self.replayer_frame_count = 0

        # emit the signal
        data = self.return_data(print_text)
        self.time_ready.emit(data)
    

    """
    External Signals and Initialisation Callbacks
    """
    @Slot()
    def start_stop_take(self):
        if hasattr(self, 'poll_timer') and self.poll_timer.isActive():
            # emit the signal
            data = self.return_data(f"Stop Take {self.take_num}\n")
            self.time_ready.emit(data)
            self.poll_timer.stop()
            # print(self.poll_timer.isActive())
        else:
            self.poll_timer = QTimer()
            self.poll_timer.setTimerType(Qt.PreciseTimer)
            self.poll_timer.timeout.connect(self.replay_current_data)
            self.poll_timer.start(int(1/100*1000))
            self.replayer_start_time = self.replayer_timer.elapsed()
            
            self.traj_ready.emit(self.right_xsens_data)
    @Slot()
    def stop(self):
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        self.stopped.emit()
    @Slot()
    def jump_l(self):
        if self.frame_id > 0.15*self.total_frames:
            self.frame_id -= int(0.1*self.total_frames)
        else:
            self.frame_id = int(0.8*self.total_frames)
    @Slot()
    def jump_r(self):
        if self.frame_id < 0.85*self.total_frames:
            self.frame_id += int(0.1*self.total_frames)
        else:
            self.frame_id = 0

    
    def send_text(self):
        data = self.return_data("")
        self.time_ready.emit(data)
    @Slot()
    def next_take(self):
        if hasattr(self, 'poll_timer'):
            if self.take_num <4:
                self.take_num += 1
            else:
                print("============================================")
                self.take_num = 1
            self.read_logged_data()
            self.replayer_start_time = self.replayer_timer.elapsed()
        self.send_text()
    @Slot()
    def prev_take(self):
        if hasattr(self, 'poll_timer'):
            if self.take_num >1:
                self.take_num -= 1
            else:
                print("============================================")
                self.take_num = 4
            self.read_logged_data()
            self.replayer_start_time = self.replayer_timer.elapsed()
        self.send_text()
    @Slot()
    def next_var(self):
        if hasattr(self, 'poll_timer'):
            if self.var_num < 6:
                self.var_num += 1
                self.take_num = 1
                self.var_id = f"var_{self.var_num}"
                self.read_logged_data()
            else:
                self.var_num = 1
                self.take_num = 1
                self.var_id = f"var_{self.var_num}"
                self.read_logged_data()
            self.replayer_start_time = self.replayer_timer.elapsed()
        self.send_text()
    @Slot()
    def prev_var(self):
        if hasattr(self, 'poll_timer'):
            if self.var_num > 1:
                self.var_num -= 1
                self.take_num = 1
                self.var_id = f"var_{self.var_num}"
                self.read_logged_data()
            else:
                self.var_num = 6
                self.take_num = 1
                self.var_id = f"var_{self.var_num}"
                self.read_logged_data()
            self.replayer_start_time = self.replayer_timer.elapsed()
        self.send_text()
    @Slot()
    def next_role(self):
        if hasattr(self, 'poll_timer'):
            if self.patient_num < 3:
                self.patient_num += 1
                self.var_num = 1
                self.take_num = 1
                self.var_id = f"var_{self.var_num}"
                self.patient_id = f"p{self.patient_num}"
                self.subject_path = f"logs/pycorc_recordings/{self.exp_id}/{self.patient_id}/{self.subject_id}"
                self.read_logged_data()
            else:
                self.patient_num = 1
                self.var_num = 1
                self.take_num = 1
                self.var_id = f"var_{self.var_num}"
                self.patient_id = f"p{self.patient_num}"
                self.subject_path = f"logs/pycorc_recordings/{self.exp_id}/{self.patient_id}/{self.subject_id}"
                self.read_logged_data()
            self.replayer_start_time = self.replayer_timer.elapsed()
        self.send_text()
    @Slot()
    def prev_role(self):
        if hasattr(self, 'poll_timer'):
            if self.patient_num > 1:
                self.patient_num -= 1
                self.var_num = 1
                self.take_num = 1
                self.var_id = f"var_{self.var_num}"
                self.patient_id = f"p{self.patient_num}"
                self.subject_path = f"logs/pycorc_recordings/{self.exp_id}/{self.patient_id}/{self.subject_id}"
                self.read_logged_data()
            else:
                self.patient_num = 3
                self.var_num = 1
                self.take_num = 1
                self.var_id = f"var_{self.var_num}"
                self.patient_id = f"p{self.patient_num}"
                self.subject_path = f"logs/pycorc_recordings/{self.exp_id}/{self.patient_id}/{self.subject_id}"
                self.read_logged_data()
            self.replayer_start_time = self.replayer_timer.elapsed()
        self.send_text()



        
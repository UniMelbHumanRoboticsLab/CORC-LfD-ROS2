import os
import numpy as np
import pandas as pd

from pycorc_io.package_utils.unpack_json import get_subject_params
from pycorc_io.xsens.ub_pckg.ub import ub
from PySide6.QtCore import QObject, Signal,QTimer,Slot,QElapsedTimer,Qt
NUM_RFT = 3
rft_key = ["clav","ua","fa"]
class ubo_logger(QObject):
    time_ready = Signal(dict)
    bias_ready = Signal(dict)
    collect_finish = Signal()
    stopped = Signal()
    def __init__(self,init_args,session_data):
        super().__init__()
        self.init_args = init_args
        self.exp_id = session_data["exp_id"]
        self.patient_id = session_data["patient_id"]
        self.subject_id = session_data["subject_id"]
        self.var_id = session_data["var_id"]
        self.take_num = session_data["take_num"]
        self.take_text = "Bias"
        self.subject_path = f"logs/pycorc_recordings/{self.exp_id}/{self.patient_id}/{self.subject_id}"
        
        self.save_path = os.path.join(os.path.dirname(__file__), '../../..',f"{self.subject_path}/{self.var_id}/raw")
        body_path = os.path.join(os.path.dirname(__file__), '../../..',f'logs/pycorc_recordings/{self.exp_id}/subject_measurements/{self.subject_id}')
        self.body_params_rbt,self.ft_grav,self.removed_bias = get_subject_params(body_path)
        self.init_bias = np.load(f"{body_path}/UBOAvgBias.npy")
        self.skeleton = ub(self.body_params_rbt,model="ubo",arm_side="right")
        
        self.sensors_ready_flag = False
        
        # FPS Calculator
        self.logger_frame_count = 0
        self.logger_fps = 0
        self.logger_timer = QElapsedTimer() # use the system clock
        self.logger_timer.start()
        self.logger_cur_time = self.logger_timer.elapsed()
        self.logger_last_time = self.logger_timer.elapsed()
        self.elapsed_time = 0
        
        print("UBO Logger Started")
    
    """
    Logging Callback
    """
    def log_current_data(self):
        # update FPS
        print_text = f"{self.subject_id}-{self.patient_id}-{self.var_id}\nTake {self.take_text}\n"
        self.logger_frame_count += 1
        self.logger_cur_time = self.logger_timer.elapsed()
        self.elapsed_time = (self.logger_cur_time-self.logger_start_time)/1000
        print_text += f"Time Elapsed:{self.elapsed_time}\n"
        if self.logger_cur_time-self.logger_last_time >= 1000:
            self.logger_fps = self.logger_frame_count * 1000 / (self.logger_cur_time-self.logger_last_time)
            self.logger_last_time = self.logger_cur_time
            self.logger_frame_count = 0

        if self.init_args["corc"]["on"] and hasattr(self, 'corc_response'):
            # print_text += f'CORC:{self.corc_response["corc_fps"]}\n'
            self.corc_arr.append(self.corc_response["raw_data"]+[self.elapsed_time])

        if self.init_args["xsens"]["on"] and hasattr(self, 'xsens_response'):
            # print_text += f'XSENS:{self.xsens_response["xsens_fps"]}\n'

            timecode = self.xsens_response["timecode"]
            right = self.xsens_response["right"]["list"]
            left = self.xsens_response["left"]["list"]
            self.xsens_arr.append([timecode]+right+left)
        
        """
        TODO: 
            if take num is 0, 
            use the xsens readings to calculate gravity bias
            remove it from the corc reading to get rft bias at that sample point
            append the rft bias arrar without any forces acting on it (including gravity)
        """
        if hasattr(self, 'corc_response') and hasattr(self, 'xsens_response') and self.take_num == 0:
            robot_joints, robot_ee = self.skeleton.ub_fkine(right)
            corc_data =  self.corc_response["raw_data"]
            bias_data = []
            for i in range(NUM_RFT):
                # add back the removed gravity bias
                cur_removed_bias = np.array(self.removed_bias[rft_key[i]])
                force_data = np.array(corc_data[1+i*6:1+i*6+6])+cur_removed_bias-self.init_bias[i*6:i*6+6]
                
                # weight compensate with if robot_ee exist to get the actual installation bias
                pose = robot_ee[i + 1]
                weight_comp = [x * self.ft_grav[rft_key[i]] for x in [0,0,-1]]
                force_data = force_data - np.array(list(np.matmul(pose.R.T, np.array(weight_comp))) + [0,0,0])
                bias_data += force_data.tolist()
            self.bias_arr.append(bias_data)
                
        # emit the signal
        data = {
            "print_text":print_text,
            "logger_fps":self.logger_fps
        }
        self.time_ready.emit(data)
    
    """
    Initialization Callback
    """
    @Slot()
    def start_worker(self):
        if self.sensors_ready_flag:
            self.poll_timer = QTimer()
            self.poll_timer.setTimerType(Qt.PreciseTimer)
            self.poll_timer.timeout.connect(self.log_current_data)
            self.poll_timer.start(int(1/100*1000))
            self.logger_start_time = self.logger_timer.elapsed()
    
            # corc
            if self.init_args["corc"]["on"]:
                self.corc_arr = []
            # xsens
            if self.init_args["xsens"]["on"]:
                self.xsens_arr = []
            # bias
            if self.init_args["corc"]["on"] and self.init_args["xsens"]["on"]:
                self.bias_arr = []
        else:
            data = {
                "print_text":"Bro wait\n",
                "logger_fps":0
            }
            self.time_ready.emit(data)
        
    """
    External Signals Callbacks
    """
    @Slot()
    def sensors_ready(self): # [self.rft_data_arr,self.rft_pose,self.rft_fps]
        self.sensors_ready_flag = True
        data = {
            "print_text":"Sensors Ready\n",
            "logger_fps":0
        }
        self.time_ready.emit(data)
    def update_corc(self,corc_response): # [self.rft_data_arr,self.rft_pose,self.rft_fps]
        self.corc_response = corc_response
    @Slot(dict)
    def update_xsens(self,xsens_response): # [self.rft_data_arr,self.rft_pose,self.rft_fps]
        self.xsens_response = xsens_response
    @Slot()
    def stop(self):
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        self.stopped.emit()
    @Slot()
    def redo_take(self):
        # reset everything
        if self.init_args["corc"]["on"]:
            self.corc_arr = []
        if self.init_args["xsens"]["on"]:
            self.xsens_arr = []
        self.logger_start_time = self.logger_timer.elapsed()
        
        self.take_num -= 1
        self.take_text = f"{self.take_num}"
        print(f"Redo {self.subject_id}-{self.patient_id}-{self.var_id} Take {self.take_text}")
    @Slot()
    def reset_logger(self):
        print(f"\nTake {self.take_text} Elapsed Time: {self.elapsed_time}")

        if hasattr(self, 'corc_response') and hasattr(self, 'xsens_response'):
            joints = ['trunk_ie','trunk_aa','trunk_fe',
                      'clav_dep_ev','clav_prot_ret',
                      'shoulder_fe','shoulder_aa','shoulder_ie',
                      'elbow_fe','elbow_ps',
                      'wrist_fe','wrist_dev']
            xsens_column = ["timecode"]+[f"{joint}_{side}" for side in ["right","left"] for joint in joints]
            corc_column = [
                        "corc time",
                        "F1x","F1y","F1z","T1x","T1y","T1z",
                        "F2x","F2y","F2z","T2x","T2y","T2z",
                        "F3x","F3y","F3z","T3x","T3y","T3z",
                        "elapsed_time"
                       ]
            total_arr = np.hstack((np.array(self.xsens_arr),np.array(self.corc_arr)))
            df = pd.DataFrame(total_arr,columns=xsens_column+corc_column)
            self.save_file(path=f"{self.save_path}/",df=df,item=f"UBORecord{self.take_text}Log")
            
            """
            emit the average installation bias for this variation to the GUI avd save everything as well
            """
            if self.take_num == 0:
                mean_bias = np.mean(np.array(self.bias_arr),axis=0)
                for i in range(3):
                    bias = mean_bias[i*6:i*6+6]
                    
                    txt = ""
                    for w in bias:
                        txt+=f"{w:.4f},"
                    print(f"Installation Bias{i}:",txt)
                    
                self.bias_ready.emit({
                    "bias":mean_bias.tolist()
                        })
                
                bias_column = [
                            "F1x","F1y","F1z","T1x","T1y","T1z",
                            "F2x","F2y","F2z","T2x","T2y","T2z",
                            "F3x","F3y","F3z","T3x","T3y","T3z",
                           ]
                bias_arr = np.array(self.bias_arr)
                df = pd.DataFrame(bias_arr,columns=bias_column)
                self.save_file(path=f"{self.save_path}/",df=df,item="UBOConstBiasLog")
                np.save(f"{self.save_path}/UBOAvgBias", mean_bias)

        # reset everything
        if self.init_args["corc"]["on"]:
            self.corc_arr = []
        if self.init_args["xsens"]["on"]:
            self.xsens_arr = []
        self.logger_start_time = self.logger_timer.elapsed()
            
        self.take_num += 1
        self.take_text = f"{self.take_num}"
        
        if self.take_num > 4:
            print(f"Finished collecting {self.subject_path}/{self.var_id}")
            self.collect_finish.emit()
        
    """
    Helper Function
    """    
    def save_file(self,path,df:pd.DataFrame,item):
        if not os.path.isdir(path):
            os.makedirs(path)
        df.to_csv(f"{path}/{item}.csv",index=False)    

    
import os
import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)
import pandas as pd

from PySide6.QtCore import QObject, Signal,QTimer,Slot,QElapsedTimer,Qt
import debugpy

class FMCLoggerBase(QObject):
    time_ready = Signal(dict)
    finish_save = Signal()
    stopped = Signal()
    def __init__(self,sensor_flags,exp_id,subject_name,take_num):
        super().__init__()
        self.sensor_flags = sensor_flags
        self.take_num = take_num
        self.dyad_path = f"./experiments/{exp_id}/{subject_name}/"

        # FPS Calculator
        self.logger_frame_count = 0
        self.logger_fps = 0
        self.logger_timer = QElapsedTimer() # use the system clock
        self.logger_timer.start()
        self.logger_cur_time = self.logger_timer.elapsed()
        self.logger_last_time = self.logger_timer.elapsed()
        self.elapsed_time = 0

        self.finger_arr_index = {key: idx for idx, key in enumerate(["name","parent","local_T","global_t","global_R"])}
        self.pos_hold_l = np.empty((9,3), dtype='float64')
        self.quat_hold_l = np.empty((9,4), dtype='float64')
        self.pos_hold_r = np.empty((9,3), dtype='float64')
        self.quat_hold_r = np.empty((9,4), dtype='float64')
        

        print("FMC Logger Started")
    
    """
    Logging Callback
    """
    def compress_info(self,pos,quat,force):
        compressed_info = np.empty(1+ 9*3 + 9*4 + 9*3, dtype='float64')

        pos_new = pos.reshape(-1)
        quat_new = quat.reshape(-1)
        force_new = force.reshape(-1)

        compressed_info[0] = self.elapsed_time
        compressed_info[1:9*3+1] = pos_new
        compressed_info[1+9*3:9*7+1] = quat_new
        compressed_info[1+9*7:] = force_new

        return compressed_info
    def log_current_data(self):
        
        # update FPS
        print_text = f"Logging\n"
        self.logger_frame_count += 1
        self.logger_cur_time = self.logger_timer.elapsed()
        self.elapsed_time = (self.logger_cur_time-self.logger_start_time)/1000
        print_text += f"Time Elapsed:{self.elapsed_time}\n"
        if self.logger_cur_time-self.logger_last_time >= 1000:
            self.logger_fps = self.logger_frame_count * 1000 / (self.logger_cur_time-self.logger_last_time)
            self.logger_last_time = self.logger_cur_time
            self.logger_frame_count = 0

        # log sensor
        if self.sensor_flags["vive"] == 1 and hasattr(self, 'vive_response'):
            print_text += f'VIVE:{self.vive_response["vive_fps"]}\n'
        if self.sensor_flags["ss"] == 1:
            if hasattr(self, 'left_hand_response'):
                print_text += f'LH:{self.left_hand_response["hand_fps"]}\n'
                np.take(self.left_hand_response["fingers_dict"]["global_t_vecs"], [20,21,22,23,24,25,26,27,28],axis=0,out=self.pos_hold_l)
                np.take(self.left_hand_response["fingers_dict"]["global_quat_vecs"], [20,21,22,23,24,25,26,27,28],axis=0,out=self.quat_hold_l)
                compressed_info = self.compress_info(self.pos_hold_l,self.quat_hold_l,self.left_hand_response["fingers_dict"]["force_vecs"])
                self.left_hand_arr.append(compressed_info)
            if hasattr(self, 'right_hand_response'):
                print_text += f'RH:{self.right_hand_response["hand_fps"]}\n'
                np.take(self.right_hand_response["fingers_dict"]["global_t_vecs"], [20,21,22,23,24,25,26,27,28],axis=0,out=self.pos_hold_r)
                np.take(self.right_hand_response["fingers_dict"]["global_quat_vecs"], [20,21,22,23,24,25,26,27,28],axis=0,out=self.quat_hold_r)
                compressed_info = self.compress_info(self.pos_hold_r,self.quat_hold_r,self.right_hand_response["fingers_dict"]["force_vecs"])
                self.right_hand_arr.append(compressed_info)
        if self.sensor_flags["rft"] == 1 and hasattr(self, 'rft_response'):
            print_text += f'RFT:{self.rft_response["rft_fps"]}\n'
            self.rft_arr.append(np.concatenate((np.array([self.elapsed_time]),self.rft_response["rft_data_arr"])))

        # emit the signal
        data = {
            "print_text":print_text,
            "logger_fps":self.logger_fps,
        }
        self.time_ready.emit(data)
    
    """
    Initialization Callback
    """
    @Slot()
    def start_worker(self):
        self.poll_timer = QTimer()
        self.poll_timer.setTimerType(Qt.PreciseTimer)
        self.poll_timer.timeout.connect(self.log_current_data)
        self.poll_timer.start(int(1/150*1000))
        self.logger_start_time = self.logger_timer.elapsed()

        # vive
        if self.sensor_flags["vive"] == 1:
            self.vive_arr = []
        # hand
        if self.sensor_flags["ss"] == 1:
            self.left_hand_arr = []
            self.right_hand_arr = []
        # rft
        if self.sensor_flags["rft"] == 1:
            self.rft_arr = []
        
    """
    External Signals Callbacks
    """
    @Slot(dict)
    def update_vive(self,vive_response): # [trackers_pos,trackers_frame,trackers_name,wrists_pos,wrists_frame,self.vive_fps]
        self.vive_response = vive_response
    @Slot(dict)
    def update_left_hand(self,left_hand_response): # [self.finger_arr,self.force_vecs,self.print_text,self.hand_fps]
        self.left_hand_response = left_hand_response
    @Slot(dict)
    def update_right_hand(self,right_hand_response): # [self.finger_arr,self.force_vecs,self.print_text,self.hand_fps]
        self.right_hand_response = right_hand_response
    @Slot(dict)
    def update_rft(self,rft_response): # [self.rft_data_arr,self.rft_pose,self.rft_fps]
        self.rft_response = rft_response
    @Slot()
    def stop(self):
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        self.stopped.emit()
    @Slot()
    def reset_logger(self):
        # debugpy.debug_this_thread()
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        
        print(f"Elapsed Time: {self.elapsed_time}")
        # emit the signal
        self.time_ready.emit({
            "print_text":"Logger Saving \n",
            "logger_fps": 0.0
        })

        if self.sensor_flags["ss"] == 1:
            if hasattr(self, 'left_hand_response'):
                bone_names = []
                for i in [20,21,22,23,24,25,26,27,28]:
                    bone_names.append(self.left_hand_response["fingers_dict"]["names"][i])
                columns = ["time"]+[f"{bone}_pos_{dim}"for bone in bone_names for dim in ["x","y","z"]]+[f"{bone}_quat_{dim}"for bone in bone_names for dim in ["x","y","z","w"]]+[f"{bone}_f_{dim}"for bone in bone_names for dim in ["x","y","z"]]
                df = pd.DataFrame(self.left_hand_arr,columns=columns)
                self.save_file(path=f"{self.dyad_path}/{self.take_num}/",df=df,item="left_hand")
            if hasattr(self, 'right_hand_response'):
                bone_names = []
                for i in [20,21,22,23,24,25,26,27,28]:
                    bone_names.append(self.right_hand_response["fingers_dict"]["names"][i])
                columns = ["time"]+[f"{bone}_pos_{dim}"for bone in bone_names for dim in ["x","y","z"]]+[f"{bone}_quat_{dim}"for bone in bone_names for dim in ["x","y","z","w"]]+[f"{bone}_f_{dim}"for bone in bone_names for dim in ["x","y","z"]]
                df = pd.DataFrame(self.right_hand_arr,columns=columns)
                self.save_file(path=f"{self.dyad_path}/{self.take_num}/",df=df,item="right_hand")
        if self.sensor_flags["rft"] == 1 and hasattr(self, 'rft_response'):
            columns = ["time","Fx","Fy","Fz","Tx","Ty","Tz"]
            df = pd.DataFrame(self.rft_arr,columns=columns)
            self.save_file(path=f"{self.dyad_path}/{self.take_num}/",df=df,item="rft_wrenches")

        # emit the signal
        self.time_ready.emit({
            "print_text":"Logger Stopped \n",
            "logger_fps": 0.0
        })
        self.finish_save.emit()
        
    """
    Helper Function
    """    
    def save_file(self,path,df:pd.DataFrame,item):
        if not os.path.isdir(path):
            os.makedirs(path)
        df.to_csv(f"{path}/{item}.csv")
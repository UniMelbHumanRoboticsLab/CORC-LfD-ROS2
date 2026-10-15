import os, sys 

# import global local ROS2 packages
from message_filters import Subscriber, ApproximateTimeSynchronizer
import rclpy
from sensor_msgs.msg import JointState
from corc_interfaces.msg import TaskDemo,SyncData,SessionParam,DesiredWrench

# import custom libraries within local ROS2 package  
sys.path.append(os.path.join(os.path.dirname(__file__)))
from base_node import base_node,workspace_root

# # import extra custom libraries outside local ROS2 package (don't need to include __init__.py for each new folder)
sys.path.append(workspace_root)
from corc_py.pycorc_io.package_utils.unpack_json import get_subject_params
from corc_py.pycorc_io.xsens.ub_pckg.ub import ub
from corc_py.pycorc_io.corc.emu import emu_robot
from data_process.post_process_pkg import JointVelEstimatorND,align_dtw,warp_target_to_ref
from data_process.file_util_pkg import create_dir
from data_visual.plot_pkg import split_plot_all
from lfd.tpgmm_pkg.tpgmm_util import arrange_data,get_optim_nbGauss,unpack_tp
from lfd.tpgmm_pkg.TPGMM import TPGMM


# import extra custom libraries within this ROS2 package but in other folders (need to include __init__.py for each new folder)
from sm_py_common.common import right_joints,left_joints,header,q_index,qdot_index,emu_kin_index,emu_transf_force_index,sbmvmt_index,var_train_combis,convert_to_tps

# import virtual environment libraries
import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)
import matplotlib.pyplot as plt
import pandas as pd
from transitions import State
import gc
import random
import pickle
import re
import warnings
warnings.filterwarnings(action="ignore",category=UserWarning)

xsens_placeholder = JointState()
xsens_placeholder.name = right_joints+left_joints
test = np.zeros(len(right_joints+left_joints))
test[8] = 90
xsens_placeholder.position = test

class lfd_node(base_node):
    """
    demo node derived from base node
    """
    def __init__(self,node_name="lfd"):
        super().__init__(node_name)
        
        # add new states and transitions
        self.new_states = [
            State(name="demo",on_enter=self.in_demo,on_exit=self.after_demo),
            State(name="learn",on_enter=self.in_learn,on_exit=self.after_learn),
            State(name="new_tp",on_enter=self.in_newtp,on_exit=self.after_newtp),
            State(name="repro_lfd",on_enter=self.in_repro_lfd,on_exit=self.after_repro_lfd),
            ]
        self.machine.add_states(self.new_states)
        
        self.machine.add_transition("to_init", "*", "init") # redefine "all states - init" when we add new states
        self.machine.add_transition("to_quit", "*", "quit") # redefine "all states - quit" when we add new states
        
        self.machine.add_transition("to_demo", "standby", "demo")
        self.machine.add_transition("to_standby", "demo", "standby")
        
        self.machine.add_transition("to_learn", "standby", "learn")
        self.machine.add_transition("to_standby", "learn", "standby")
        
        self.machine.add_transition("to_newtp", "standby", "new_tp")
        self.machine.add_transition("to_repro_lfd", "new_tp", "repro_lfd")
        self.machine.add_transition("to_standby", "repro_lfd", "standby")
        
        # Init body model
        body_params,ft_grav,removed_bias = get_subject_params(f'{workspace_root}/logs/ros2_recordings/{self.session_param['exp_id']}/subject_measurements/{self.session_param['sub_id']}/',"emu")
        self.skeleton = ub(body_params,model="emu",arm_side="right",ft_grav=ft_grav,removed_bias=removed_bias)
        # Init emu model
        self.emu = emu_robot()
        
        # Init session parameters and publish
        self.session_param["max_sbmvmt"] = self.config["max_sbmvmt"]
        self.session_param["num_rep"] = self.config["num_rep"]
        self.session_param["learn_input"] = self.config["learn_input"]
        self.session_param["learn_output"] = self.config["learn_output"]
        self.session_param["corc_on"] = self.config["corc"]
        self.session_param["xsens_on"] = self.config["xsens"]
        self.session_param["num_of_dims"] = 0
        self.session_param_pub = self.create_publisher(SessionParam, 'session_param', self.qos)
        if self.session_param["learn_input"] == "task":
            self.session_param["num_of_dims"] += len(emu_kin_index)
            self.session_param["input_index"] = emu_kin_index
        elif self.session_param["learn_input"] == "joint":
            self.session_param["num_of_dims"] += len(q_index)+len(qdot_index)
            self.session_param["input_index"] = q_index+qdot_index
            
        if self.session_param["learn_output"] == "task":
            self.session_param["num_of_dims"] += len(emu_transf_force_index)
            self.session_param["output_index"] = emu_transf_force_index
        elif self.session_param["learn_output"] == "joint":
            self.session_param["num_of_dims"] += len(q_index)
            self.session_param["output_index"] = q_index
        self.publish_session_param([0 for i in range(7)])
               
        # Init session variables
        self.var_num = 0
        self.learn_status = [0 for i in range(7)] # first one is if learning is finished, rest is if each variation is learning succeed?
        self.sbmvmt_num = 0
        self.rep_num = 0
        self.data_dict = {
            "demo":{},
            "old_tp":{},
            "new_tp":{},
            "tpgmms":{},
            }
        self.data = []
        self.expected_data = []
        self.start_time = 0
        self.shutting_down = False # to prevent race conditions when deleting self.skeleton reference
        
    """
    State Transition Callback 
    """
    def corc_state_transition(self, msg): 
        super().corc_state_transition(msg)
        self.var_num = msg.var_num
        if msg.state_id == "demo":
            self.state_transition_log("demo")
            self.to_demo()
        elif msg.state_id == "learn":
            self.state_transition_log("learn")
            self.to_learn()
        elif msg.state_id == "new_tp":
            self.state_transition_log("new_tp")
            self.to_newtp()
        elif msg.state_id == "repro_lfd":
            self.state_transition_log("repro_lfd")
            self.to_repro_lfd()
        
    """
    State Entry, During and Exit 
    """
    def in_quit(self):
        # additional cleanup code
        self.shutting_down=True
        self.get_logger().info("Deleting extra objects")
        del self.skeleton # cleanup the spatial geometry references to prevent nanobind leakage
        del self.emu # cleanup the spatial geometry references to prevent nanobind leakage
        
        gc.collect()
        super().in_quit()
        
    """ 2. learn state
    entry:
        DTW based on task or joint kinematics and override the one in memory - set in demo_corc_node.yaml
        convert submovement segments to task parameters
        visual check demonstrations
        
    during:
        - no during as you want this block to finish running before state transition
    exit:
        - reset the learning finished status in CORC
    """
    def in_learn(self):
        self.get_logger().info("In learn")
        
        self.get_logger().info("Pre-processing\n")
        all_data = []
        time_list,q_traj_list,qdot_traj_list,wrench_list,sbmvmt_list,rep_label_list = [],[],[],[],[],[]
        for var_key, reps_dict in self.data_dict["demo"].items():
            for i,(rep_key,rep_data) in enumerate(reps_dict.items()):
                if i == 0:
                    ref = rep_data.copy()
                    target_interaction = rep_data.copy()
                else:
                    target_interaction = rep_data.copy()
                all_data.append(target_interaction[:,self.session_param["input_index"]+self.session_param["output_index"] ].copy())
                # for plotting
                time_list.append(ref[:,0].copy())
                q_traj_list.append(target_interaction[:,q_index].copy())
                qdot_traj_list.append(target_interaction[:,qdot_index].copy())
                wrench_list.append(target_interaction[:,emu_transf_force_index].copy())
                sbmvmt_list.append(np.expand_dims(target_interaction[:,sbmvmt_index].copy(),axis=1))
                rep_label_list.append(f'{var_key}.{rep_key}')
                
        if len(all_data) <=0:
            self.get_logger().warn("No demos")
            pass
        else:
            split_plot_all(list(self.data_dict["demo"]),time_list,q_traj_list,rep_label_list,
                           rep_split=self.session_param["num_rep"],data_type="q_deg",
                           fig_label=f"q_rad data {self.session_param['sub_id']}_{self.session_param['patient_id']}",plot=True)
            split_plot_all(list(self.data_dict["demo"]),time_list,sbmvmt_list,rep_label_list,
                           rep_split=self.session_param["num_rep"],data_type="index",
                           fig_label=f"index data {self.session_param['sub_id']}_{self.session_param['patient_id']}",plot=True)
            split_plot_all(list(self.data_dict["demo"]),time_list,wrench_list,rep_label_list,
                           rep_split=self.session_param["num_rep"],data_type="F",
                           fig_label=f"Force data {self.session_param['sub_id']}_{self.session_param['patient_id']}",plot=True)
            split_plot_all(list(self.data_dict["demo"]),time_list,qdot_traj_list,rep_label_list,
                           rep_split=self.session_param["num_rep"],data_type="qdot_deg",
                           fig_label=f"qdot_rad data {self.session_param['sub_id']}_{self.session_param['patient_id']}",plot=True)
            plt.show()
            
            self.get_logger().info("\033[94mBIC Stage \033[0m")
            # get the optim number of gauss once
            all_data = np.vstack(all_data)
            demo_wrench = all_data[:,len(self.session_param["input_index"]):]
            self.max_mag = np.max(demo_wrench,axis=0)
            self.session_param["num_of_frames"] = self.session_param["max_sbmvmt"]+1
            num_of_gauss,bic_df = get_optim_nbGauss(all_data)
            self.get_logger().info("\033[94mBIC Complete\033[0m\n")
            
            self.get_logger().info("\033[94mEM Stage")
            # start learning for each variation
            for var_key, reps_dict in self.data_dict["demo"].items():
                self.get_logger().info(f"=============={var_key} Training")
                # select a valid training combi for this variation
                valid_train = var_train_combis[var_key]
                selected_combi = [f"var_{i}" for i in random.choice(valid_train)]
                self.get_logger().info(f"Valid train combination for {var_key}: {selected_combi}")
                
                # compile the train list
                train_list = []
                for train_var in selected_combi:
                    if train_var in self.data_dict["demo"]:
                        for rep_key,rep_data in self.data_dict["demo"][train_var].items():
                            rep_tps = self.data_dict["old_tp"][train_var][rep_key]
                            rep_dict = {"time":rep_data[:,0],
                            			"data":rep_data[:,self.session_param["input_index"]+self.session_param["output_index"]],
                            			"tp":rep_tps,
                                        "id":f"{train_var}_{rep_key}" }
                            train_list.append(rep_dict) 
                    else:
                        self.get_logger().warn(f"{train_var} does not exist")
                if len(train_list) < 1: 
                    fail = True
                    self.get_logger().warn("Not enough training data. Skipping training")
                else:
                    train_tp_data,train_sample_list = arrange_data(train_list,self.session_param["num_of_frames"],self.session_param["num_of_dims"])
                    
                    fail = True
                    retry = 0
                    while fail and retry < 5:
                        self.get_logger().info(f"Training Attempt {retry}")
                        # start = times.perf_counter()
                        tpgmm = TPGMM(num_of_gauss,self.session_param["num_of_frames"],self.session_param["num_of_dims"],
                                      priors=np.ones((num_of_gauss,)),kP=0,kV=0,diagRegFact=1e-8,version="fast")
                        tpgmm.init_gmm(train_tp_data)
                        LL,fail = tpgmm.fit_em(nbMinSteps=5,nbMaxSteps=100,maxDiffLL=1e-8,updateComp=np.ones((3,1)),tp_data=train_tp_data)
                        self.get_logger().info(f"Training Success?: {not fail}")
                        retry += 1
                    if fail:
                        tpgmm.training_status = "Fail"
                        self.get_logger().warn(f"{self.session_param['directory'] } failed to converge| Pending post check")
                    else:
                        tpgmm.training_status = "Success"
                        
                        create_dir(f"{self.session_param['directory']}/tpgmm")
                        tpgmm_file_path = f"{self.session_param['directory']}/tpgmm/tpgmm_{var_key}.pkl"
                        with open(tpgmm_file_path, 'wb') as file:
                            # Dump data with highest protocol for best performance
                            pickle.dump(tpgmm,file, protocol=pickle.HIGHEST_PROTOCOL)
                        self.data_dict["tpgmms"][var_key] = tpgmm
                        
                    cur_var = int(re.search(r'\d+', var_key).group())
                    self.learn_status[cur_var] = int(not fail)
            self.get_logger().info("\033[94mEM Complete\033[0m\n")
        
        self.learn_status[0] = 1
        self.publish_session_param(self.learn_status)
    def after_learn(self):
        # reset learning status in CORC node
        self.learn_status[0] = 0
        self.publish_session_param(self.learn_status)
        
    """ 1. demo state
    entry:
        - subscribes to XSENS and CORC data topic
    during:
        - sync callback XSENS and CORC data to same time frame
        - stores demo per variation per rep in memory
        - online joint velocity and LPF
        - publishes to sync_data topic for corc_viz
    exit:
        - convert joint kinematics to JA
        - DTW and process task parameters
        - saves raw demo per variation per rep to local storage
    """
    def sync_demo(self, *args):
        emu,xsens = self.unpack_args(args)
        if emu.sbmvmt_num == 0 and self.sbmvmt_num!=0:
            self.save_rep()
        else:
            # update identifier and process data
            self.sbmvmt_num = emu.sbmvmt_num
            self.rep_num = emu.rep_num
            sample_dict = self.unpack_process_sync_data(emu,xsens)
            self.visual_sync_data(sample_dict)
            # publish to plotjuggler
            j_msg = JointState()
            j_msg.name = right_joints
            j_msg.position = sample_dict["right_joint_pos"]
            j_msg.velocity = sample_dict["right_joint_vel"]
            j_msg.header.stamp.sec = self.sbmvmt_num
            self.j_kine.publish(j_msg)
            
            j_msg2 = JointState()
            j_msg2.name = ["emu_x","emu_y","emu_z","lfd_x","lfd_y","lfd_z"]
            j_msg2.effort = sample_dict["force_transform"].tolist()+[0,0,0]
            j_msg2.header.stamp.sec = self.sbmvmt_num
            self.w_dyna.publish(j_msg2)
            if self.sbmvmt_num > 0:
                # when user start the demo
                self.compile_sync_data(sample_dict)
                

    def in_demo(self):
        self.get_logger().info("In demo")
        self.sync_id = "demo"
        self.init_sync(1/500,self.sync_demo)
    def after_demo(self):
        # DTW Process
        rep_keys_shuffled = list(self.reps_dict.keys())
        random.shuffle(rep_keys_shuffled) # shuffle the reps for each variation
        # save the shuffling order for later sanity checks
        shuffle = pd.DataFrame(rep_keys_shuffled)
        shuffle.to_csv(f'{self.session_param['directory']}/var_{self.var_num}/{self.sync_id}/shuffle.csv', index=False, header=False)
        var_key = f"var_{self.var_num}"
        self.data_dict["old_tp"][var_key] = {}
        for i,rep_key in enumerate(rep_keys_shuffled):
            if i == 0:
                ref = self.reps_dict[rep_key].copy()
                target_interaction = self.reps_dict[rep_key].copy()
            else:
                target_interaction = self.reps_dict[rep_key].copy()
            
            # DTW alignment
            paths = align_dtw(ref[:,q_index],target_interaction[:,q_index])
            warped_interaction = warp_target_to_ref(ref,target_interaction,paths)
            warped_interaction[:,0] = ref[:,0]
            
            self.reps_dict[rep_key] = warped_interaction.copy()
            
            # get submovement indices
            submovement_indices = warped_interaction[:,sbmvmt_index]
            submovement_indices[-1] += 1
            segments = np.r_[0, np.flatnonzero(np.diff(submovement_indices) != 0) + 1]
            
            # extract and convert the segmented movement into meaningful TPs
            submovements = warped_interaction[segments,:]
            self.data_dict["old_tp"][var_key][f"{rep_key}"] = convert_to_tps(submovements,self.skeleton,self.session_param["num_of_dims"],
                                                                             self.session_param["learn_input"],self.session_param["learn_output"],
                                                                             self.session_param["input_index"],self.session_param["output_index"])
            
        self.data_dict[self.sync_id][f"var_{self.var_num}"] = self.reps_dict
        # check the number of demos we have
        self.check_rep()
        # cleanup subscribers
        self.cleanup_sync()
        
    """ 3. new tp state
    entry:
        - subscribes to XSENS and CORC data topic
    during:
        - sync callback XSENS and CORC data to same time frame
        - stores newTp per variation per rep in memory
        - publishes to sync_data topic for corc_viz
    exit:
        - check new TP
    """
    def sync_newtp(self, *args):
        emu,xsens = self.unpack_args(args)
        if emu.sbmvmt_num == 0:
            self.reset_rep()
        else:
            # update identifier and process data
            self.sbmvmt_num = emu.sbmvmt_num
            self.rep_num = emu.rep_num
            sample_dict = self.unpack_process_sync_data(emu,xsens)
            
            if self.sbmvmt_num > 0:
                # when user gives the TPs
                self.compile_sync_data(sample_dict)
                self.visual_sync_data(sample_dict)
    def in_newtp(self):
        self.get_logger().info("In new_tp")
        self.sync_id = "new_tp"
        self.init_sync(1/62.5,self.sync_newtp)
    def after_newtp(self):
        # check the tps we have
        self.save_rep()
        # cleanup subscribers
        self.cleanup_sync()
        
    """ 4. repro_lfd state
    entry:
        - subscribes to XSENS and CORC data topic
    during:
        - sync callback XSENS and CORC data to same time frame, publishes to sync_data topic for corc_viz
        - stores interaction during repro_lfd per variation per rep in memory
        - nline joint velocity and LPF, LfD deploy
        - publish to CORC as desired wrench
        - 
    exit:
        - saves interaction during repro_lfd per variation per rep to local storage
    """
    def sync_repro_lfd(self, *args):
        emu,xsens = self.unpack_args(args)
        if emu.sbmvmt_num == 0 and self.sbmvmt_num!=0:
            self.save_rep()
            # reset the desired wrench in CORC
            msg = DesiredWrench()
            msg.name = ["hi"]
            msg.desired_wrench = [0,0,0]
            self.desired_wrench_pub.publish(msg)
        else:
            # update identifier and process data
            self.sbmvmt_num = emu.sbmvmt_num
            self.rep_num = emu.rep_num
            sample_dict = self.unpack_process_sync_data(emu,xsens)
            self.visual_sync_data(sample_dict)
            if self.sbmvmt_num > 0:
                # start saving/publishing when user asks for interaction
                synced_data,_ = self.compile_sync_data(sample_dict)
                
                # Deploy LFD here
                if self.session_param["learn_input"] == "joint":
                    data_in = np.deg2rad(synced_data[self.session_param["input_index"]])
                else:
                    data_in = synced_data[self.session_param["input_index"]]
                expected_data_sample,_,_ = self.data_dict["tpgmms"][f"var_{self.var_num}"].GMR_sample(data_in)
                self.expected_data.append(expected_data_sample)
                
                # publish LFD desired wrench
                msg = DesiredWrench()
                msg.name = ["hi"]
                msg.desired_wrench = expected_data_sample
                self.desired_wrench_pub.publish(msg)
                
                # publish to plotjuggler
                j_msg = JointState()
                j_msg.name = right_joints
                j_msg.position = sample_dict["right_joint_pos"]
                j_msg.velocity = sample_dict["right_joint_vel"]
                j_msg.header.stamp.sec = self.sbmvmt_num
                self.j_kine.publish(j_msg)
                
                j_msg2 = JointState()
                j_msg2.name = ["emu_x","emu_y","emu_z","lfd_x","lfd_y","lfd_z"]
                j_msg2.effort = sample_dict["force_transform"].tolist()+expected_data_sample.tolist()
                j_msg2.header.stamp.sec = self.sbmvmt_num
                self.w_dyna.publish(j_msg2)
    def in_repro_lfd(self):
        self.get_logger().info("In repro_lfd")
        print(f'conditioned status: {self.data_dict["tpgmms"][f"var_{self.var_num}"].conditioned}')
        self.sync_id = "repro_lfd"
        self.init_sync(1/500,self.sync_repro_lfd)
        self.desired_wrench_pub = self.create_publisher(DesiredWrench, 'desired_wrench', 10)
    def after_repro_lfd(self):
        # cleanup subscribers
        self.cleanup_sync()

    """"
    Helper Functions
    """
    def publish_session_param(self,learn_status):
        msg = SessionParam()
        msg.exp_id = self.session_param["exp_id"]
        msg.patient_id = self.session_param["patient_id"]
        msg.sub_id = self.session_param["sub_id"]
        msg.max_sbmvmt = self.session_param["max_sbmvmt"]
        msg.num_rep = self.session_param["num_rep"]
        msg.learn_status = learn_status
        msg.learn_output = self.session_param["learn_output"]
        self.session_param_pub.publish(msg)

    """
    Sync Helper Function
    """
    # init sync and velocity estimator
    def init_sync(self,delay,sync_callback):
        self.reps_dict = {}
        self.subs = []
        if self.session_param["corc_on"]:
            emu_task_sub = Subscriber(self, TaskDemo, "emu_task_dynamics",self.qos)
            self.subs.append(emu_task_sub)
        if self.session_param["xsens_on"]:
            xsens_sub = Subscriber(self, JointState, "joint_states",self.qos)
            self.subs.append(xsens_sub)
        
        self.data_sync = ApproximateTimeSynchronizer(self.subs, 10,delay)
        self.data_sync.registerCallback(sync_callback)
        
        self.sync_pub = self.create_publisher(SyncData, 'sync_data', 10)
        self.j_kine = self.create_publisher(JointState, 'j_proc', 10)
        self.w_dyna = self.create_publisher(JointState, 'w_proc', 10)
        
        self.joint_vel_est = JointVelEstimatorND(len(right_joints))
    # during sync
    def unpack_args(self,args):
        # Unpack based on number of arguments
        if len(args) == 1:
            if self.session_param["corc_on"]:
                emu = args[0]
                xsens = xsens_placeholder
        elif len(args) == 2:
            emu, xsens = args
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(args)}")
        return emu,xsens
    def unpack_emu(self,emu):
        timestamp = emu.header.stamp.sec+emu.header.stamp.nanosec/1e9
        emu_pos = np.array(emu.position, dtype=np.float64)
        emu_vel = np.array(emu.velocity, dtype=np.float64)
        emu_pos_j = np.array(emu.joint_position, dtype=np.float64)* 180 / np.pi
        # emu_pos_j = np.array([0,30,45])
        corrected_emu = np.array([emu_pos_j[0],emu_pos_j[1],emu_pos_j[2]-emu_pos_j[1],0])
        # print(corrected_emu)
        emu_vel_j = np.array(emu.joint_velocity, dtype=np.float64)
        emu_wrench = np.array(emu.wrench_m, dtype=np.float64)
        emu_int = np.array(emu.wrench_int, dtype=np.float64)
        emu_comm = np.array(emu.wrench_com, dtype=np.float64)
        return timestamp,emu_pos,emu_vel,corrected_emu,emu_vel_j,emu_wrench,emu_int,emu_comm
    def unpack_xsens(self,xsens):
        joint_map = {name: i for i, name in enumerate(xsens.name)}
        # Extract right arm joint positions
        right_joint_pos = np.array([
            xsens.position[joint_map[joint]] for joint in right_joints
        ])
        left_joint_pos = np.array([
            xsens.position[joint_map[joint]] for joint in left_joints
        ])
        return right_joint_pos,left_joint_pos
    def unpack_process_sync_data(self,emu,xsens):
        if not self.shutting_down: # prevent race condition when deleting self.skeleton
            # parse synced emu and xsens
            timestamp,emu_pos,emu_vel,emu_pos_j,emu_vel_j,emu_wrench,emu_int,emu_comm = self.unpack_emu(emu)
            right_joint_pos,left_joint_pos = self.unpack_xsens(xsens)
            # online joint vel and LPF
            right_joint_vel = self.joint_vel_est.update(timestamp,right_joint_pos)
            
            # add back the initial removed sensor base bias
            net_bias = np.array(self.skeleton.removed_bias["fa"])
            force_data = emu_wrench+np.array(net_bias)
            
            #  remove the subsequent gravity
            # _, ub_ees = self.skeleton.ub_fkine(right_joint_pos)
            # pose = ub_ees[-1]
            _, emu_ee = self.emu.emu_fkine(emu_pos_j)
            pose = emu_ee[-1]
            weight_comp = [x * self.skeleton.ft_grav["fa"] for x in [0,0,-1]]
            force_data = force_data - np.array(list(np.matmul(pose.R.T, np.array(weight_comp))) + [0,0,0])
            
            # get force in EMU frame, disregard moment, negate to be force on human
            force_transform = -np.matmul(pose.R,force_data[:3])
            # print(f"Position:{pose.t} \t IntForce:{force_transform}\t IntForceMag:{np.linalg.norm(force_transform)}")
            
            sample_dict = {
                "timestamp":timestamp,
                "rep_num":emu.rep_num,
                "sbmvmt_num":emu.sbmvmt_num,
                "emu_pos":emu_pos,
                "emu_vel":emu_vel,
                "emu_pos_j":emu_pos_j,
                "emu_vel_j":emu_vel_j,
                "emu_wrench":emu_wrench,
                "force_transform":force_transform,
                "emu_int":emu_int,
                "emu_comm":emu_comm,
                "right_joint_pos":right_joint_pos,
                "right_joint_vel":right_joint_vel,
                "left_joint_pos":left_joint_pos,
                }
            return sample_dict
        else:
            sample_dict = {
                "timestamp":0,
                "rep_num":0,
                "sbmvmt_num":0,
                "emu_pos":np.zeros(3),
                "emu_vel":np.zeros(3),
                "emu_pos_j":np.zeros(3),
                "emu_vel_j":np.zeros(3),
                "emu_wrench":np.zeros(6),
                "force_transform":np.zeros(3),
                "emu_int":np.zeros(3),
                "emu_comm":np.zeros(3),
                "right_joint_pos":np.zeros(12),
                "right_joint_vel":np.zeros(12),
                "left_joint_pos":np.zeros(12),
                }
            return sample_dict
    def compile_sync_data(self,sample_dict):
        if not self.shutting_down: # prevent race condition when deleting self.skeleton
            corrected_time = sample_dict["timestamp"]-self.start_time
            synced_data = np.concatenate((np.array([corrected_time]),np.array([sample_dict["rep_num"]]),np.array([sample_dict["sbmvmt_num"]]),
                                          sample_dict["emu_pos"].copy(),sample_dict["emu_vel"].copy(),
                                          sample_dict["emu_wrench"].copy(),sample_dict["force_transform"].copy(),sample_dict["emu_int"].copy(),sample_dict["emu_comm"].copy(),
                                          sample_dict["right_joint_pos"].copy(),sample_dict["right_joint_vel"].copy()))
            self.data.append(synced_data)
            return synced_data,corrected_time
        else:
            return np.zeros(48),0.0
    def visual_sync_data(self,sample_dict):
        if not self.shutting_down: # prevent race condition when deleting self.skeleton
            if len(self.data)>0:
                corrected_time = sample_dict["timestamp"]-self.start_time
            else:
                self.start_time = sample_dict["timestamp"]
                corrected_time = sample_dict["timestamp"]-self.start_time
            
            msg = SyncData()
            msg.sync_id = self.sync_id
            msg.timestamp = corrected_time
            msg.var_num = self.var_num
            msg.rep_num = self.rep_num
            msg.sbmvmt_num = self.sbmvmt_num
            
            msg.position = sample_dict["emu_pos"].copy()
            msg.joint_position = sample_dict["emu_pos_j"].copy()
            msg.velocity = sample_dict["emu_vel"].copy()
            msg.wrench = sample_dict["force_transform"].copy()
            
            msg.right.position = sample_dict["right_joint_pos"].copy()
            msg.right.velocity = sample_dict["right_joint_vel"].copy()
            msg.left.position = sample_dict["left_joint_pos"].copy()
            self.sync_pub.publish(msg)
    # after sync
    def cleanup_sync(self):
        # reset everything for the next rep
        self.reset_rep()
        self.reps_dict = {}
        self.sync_id = "unknown"
        # delete every subscriber, publisher, estimator for the next demo
        for cur_sub in self.subs:
            cur_sub.sub.destroy()
        del self.data_sync
        self.sync_pub.destroy()
        self.j_kine.destroy()
        self.w_dyna.destroy()
        del self.joint_vel_est
    def reset_rep(self):
        self.rep_num = 0
        self.sbmvmt_num = 0
        self.data = []
        self.expected_data = []
        self.start_time = 0
    def save_file(self,path,df:pd.DataFrame,item):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        df.to_csv(f"{path}/{item}.csv",index=False)  
    def save_rep(self):
        # to save the synch data per var per rep cleanly
        save_directory = f"{self.session_param['directory']}/var_{self.var_num}/{self.sync_id}"
        self.data = np.array(self.data)
        # convert joint kinematics to radians
        self.data[:,q_index+qdot_index] = np.deg2rad(self.data[:,q_index+qdot_index])
        
        # only save the predefined number of reps
        if self.rep_num <= self.session_param["num_rep"]:
            if "new_tp" not in self.sync_id:
                df = pd.DataFrame(self.data,columns=header)
                self.save_file(path=save_directory,df=df,item=f"IntRecord{self.rep_num}Log")
                self.get_logger().info(f"Logged {self.sync_id} var_{self.var_num} rep_{self.rep_num}. Elapsed (s): {self.data[-1][0]}")
                
                # save demos into memory (if we redo a variation, it will wipe out all repetitions of that variations)
                # or save the wrench commands from the LfD
                if self.sync_id == "demo":
                    self.reps_dict[f"rep_{self.rep_num}"] = self.data
                    self.data_dict[self.sync_id][f"var_{self.var_num}"] = self.reps_dict
                else:
                    # save the expected wrenches from the LfD
                    save_wrenches = np.array(self.expected_data)
                    df = pd.DataFrame(save_wrenches,columns=["Fx","Fy","Fz"])
                    self.save_file(path=save_directory,df=df,item=f"LfDIntRecord{self.rep_num}Log")
            else:
                task_params = convert_to_tps(self.data,self.skeleton,self.session_param["num_of_dims"],
                                            self.session_param["learn_input"],self.session_param["learn_output"],
                                            self.session_param["input_index"],self.session_param["output_index"])
                self.data_dict[self.sync_id][f"var_{self.var_num}"] = task_params
                new_tp = unpack_tp(task_params,self.session_param["num_of_frames"],self.session_param["num_of_dims"])
                df = pd.DataFrame(task_params,
                                  columns=[f"A{j}" for j in range(1,self.session_param["num_of_dims"]*self.session_param["num_of_dims"]+1)] + [f"b{j}" for j in range(1,self.session_param["num_of_dims"]+1)])
                self.save_file(path=save_directory,df=df,item=f"IntTPs{self.rep_num}Log")
                # recondition to new TP
                self.data_dict["tpgmms"][f"var_{self.var_num}"].conditionTPGMM(new_tp,len(self.session_param["input_index"])-1,self.session_param["num_of_dims"]-1)
                print(f'conditioned status: {self.data_dict["tpgmms"][f"var_{self.var_num}"].conditioned}')
        # reset everything for the next rep
        self.reset_rep()
    def check_rep(self):
        # check the number of repro interaction we have
        var_key = f"var_{self.var_num}"
        if var_key in self.data_dict[self.sync_id].keys():
            for rep_key in self.data_dict[self.sync_id][var_key]:
                print(f"{self.sync_id} {var_key} {rep_key} {(self.data_dict[self.sync_id][var_key][rep_key].shape[0])/62.5}")
        else:
            print(f"{self.sync_id} {var_key} not yet demonstrated")

def main(args=None):
    lfd = None
    try:
        rclpy.init(args=args)
        lfd = lfd_node()
        rclpy.spin(lfd)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # exit node safely
        if lfd is not None:
            lfd.destroy_node()
        try:
            rclpy.shutdown()   # unconditional + guarded
        except Exception:
            pass
if __name__ == '__main__':
    main()
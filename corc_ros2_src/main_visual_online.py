import os, sys
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(
    precision=10,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)
import random
import time as times
import pickle
import re
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from data_process.post_process_pkg import lpf,calc_fixed_diff,align_dtw,warp_target_to_ref
from data_process.file_util_pkg import get_raw_data_with_vel,create_dir

from corc_py.pycorc_io.package_utils.unpack_json import get_subject_params
from corc_py.pycorc_io.xsens.ub_pckg.ub import ub
from data_visual.plot_pkg import plot_multi_dim,interactive_plot,create_custom_3d_fig,split_plot_all,plot_stats
from lfd.tpgmm_pkg.tpgmm_util import unpack_tp

from corc_ros2_src.sm_py_nodes.sm_py_common.common import q_index,qdot_index,emu_kin_index,emu_transf_force_index,sbmvmt_index,emu_interaction_index,emu_command_index

learn_input = "joint"
learn_output = "task"
num_of_dims = 0
max_submvmt = 2

if learn_input == "task":
    num_of_dims += len(emu_kin_index)
    input_index = emu_kin_index
elif learn_input == "joint":
    num_of_dims += len(q_index)+len(qdot_index)
    input_index = q_index+qdot_index
    
if learn_output == "task":
    num_of_dims += len(emu_transf_force_index)
    output_index = emu_transf_force_index
elif learn_output == "joint":
    num_of_dims += len(q_index)
    output_index = q_index
    
p_start = 2
sub_start  = 1
for p in range(p_start,p_start+1):
    if p == 2:
        sm_num = 2
    else:
        sm_num=4
        
    for sub in range(sub_start,sub_start+1):
        """ init session parameters and perform train / test / validation spliting"""
        session_param = {
            "exp_id":"exp2",
            "patient_id":f"p{p}",
            "subject_id":"vincent",
            "sbmvmt_num":sm_num,
            "num_rep":3,
            "variants":["var_1","var_2"]#,"var_3"] #
        }
        subject_path = os.path.join(os.path.dirname(__file__),"..",f'logs/ros2_recordings/{session_param["exp_id"]}/{session_param["patient_id"]}/{session_param["subject_id"]}')
        """ init xsens skeleton model """
        body_path = os.path.join(os.path.dirname(__file__),"..", f'logs/ros2_recordings/{session_param["exp_id"]}/subject_measurements/{session_param["subject_id"]}')
        body_params_rbt,ft_grav,removed_bias = get_subject_params(body_path,"emu")
        skeleton = ub(body_params_rbt,model="emu",arm_side="right",ft_grav=ft_grav,removed_bias=removed_bias)
        
        data_dict = {
            "demo":{},
            "old_tp":{},
            "tpgmms":{},
            "new_tp":{},
            "repro_lfd":{},
            "lfd_commanded_wrench":{},
            "dtw_order":{}
            }
        
        # compile all demos, interaction during lfd, and lfd commanded
        for var in session_param["variants"]:
            data_dict["demo"][var] = {}
            data_dict["repro_lfd"][var] = {}
            data_dict["lfd_commanded_wrench"][var] = {}
            for n,rep in enumerate([1,2,3]):
                data_path = f'{subject_path}/{var}/demo/IntRecord{rep}Log.csv'
                full_data,time_data_unscaled,q_traj,qdot_traj = get_raw_data_with_vel(data_path)
                data_dict["demo"][var][f"rep_{rep}"] = full_data.values
                
                # data_path = f'{subject_path}/{var}/repro_lfd/IntRecord{rep}Log.csv'
                # full_data,time_data_unscaled,q_traj,qdot_traj = get_raw_data_with_vel(data_path)
                # data_dict["repro_lfd"][var][f"rep_{rep}"] = full_data.values
                
                # data_path = f'{subject_path}/{var}/repro_lfd/LfDIntRecord{rep}Log.csv'
                # data_dict["lfd_commanded_wrench"][var][f"rep_{rep}"] = pd.read_csv(data_path).values
        
        # preprocessing and DTW
        all_data = []
        var_id_list = []
        demo_time_list,q_traj_list,qdot_traj_list,demo_phri_wrench_list,demo_phri_wrench_robot_list,sbmvmt_list,demo_rep_label_list = [],[],[],[],[],[],[]
        for var_key, reps_dict in data_dict["demo"].items():
            # shuffle the reps for each variation
            data_path = f'{subject_path}/{var}/demo/shuffle.csv'
            rep_keys_shuffled = pd.read_csv(data_path,header=None).values[:,0]
            for i,rep_key in enumerate(rep_keys_shuffled):
                if i == 0:
                    ref = reps_dict[rep_key].copy()
                    data_dict["dtw_order"][var_key] = rep_keys_shuffled
                    target_interaction = reps_dict[rep_key].copy()
                else:
                    target_interaction = reps_dict[rep_key].copy()
                
                # DTW alignment
                paths = align_dtw(ref[:,q_index],target_interaction[:,q_index])
                warped_interaction = warp_target_to_ref(ref,target_interaction,paths)
                reps_dict[rep_key] = warped_interaction.copy()
                all_data.append(warped_interaction[:,input_index+output_index].copy())
                    
                # for plotting
                demo_time_list.append(ref[:,0].copy()/ref[-1,0])
                q_traj_list.append(warped_interaction[:,q_index].copy())
                qdot_traj_list.append(warped_interaction[:,qdot_index].copy())
                
                wrench = warped_interaction[:,emu_transf_force_index].copy()
                mag = np.expand_dims(np.linalg.norm(wrench,axis=1),axis=1)
                wrench_with_mag = np.hstack((wrench,mag))
                demo_phri_wrench_list.append(wrench_with_mag)
                
                wrench = warped_interaction[:,emu_interaction_index].copy()
                mag = np.expand_dims(np.linalg.norm(wrench,axis=1),axis=1)
                wrench_with_mag = np.hstack((wrench,mag))
                demo_phri_wrench_robot_list.append(wrench_with_mag)
                demo_rep_label_list.append(f'{var_key}.{rep_key}')
                var_id_list.append(var_key)
        split_plot_all(session_param["variants"],demo_time_list,q_traj_list,demo_rep_label_list,rep_split=session_param["num_rep"],data_type="q_rad",fig_label=f"q_rad demo {session_param['subject_id']}_{session_param['patient_id']}",plot=True)
        split_plot_all(session_param["variants"],demo_time_list,qdot_traj_list,demo_rep_label_list,rep_split=session_param["num_rep"],data_type="qdot_rad",fig_label=f"qdot_rad demo {session_param['subject_id']}_{session_param['patient_id']}",plot=True)
        
        # # step 2: grab the learned TPGMM and new task parameters
        # all_data = np.vstack(all_data)
        # demo_wrench = all_data[:,len(input_index):]
        # mag = np.linalg.norm(demo_wrench,axis=1)
        # max_mag = np.max(demo_wrench,axis=0)
        # num_of_frames = max_submvmt+1
        # for var_key, reps_dict in data_dict["demo"].items():
        #     tpgmm_file_path = f'{subject_path}/tpgmm/tpgmm_{var_key}.pkl'
        #     with open(tpgmm_file_path, 'rb') as outp:                        
        #         tpgmm = pickle.load(outp)
        #     data_dict["tpgmms"][var_key] = tpgmm
        #     data_dict["new_tp"][var_key] = pd.read_csv(f'{subject_path}/{var_key}/new_tp/IntTPs1Log.csv').values
        #     new_tp = unpack_tp(data_dict["new_tp"][var_key],num_of_frames,num_of_dims)
            
        #     # from spatialmath import SO3, SE3
        #     # A = new_tp.A
        #     # b = new_tp.b
        #     # fig,ax = create_custom_3d_fig()
        #     # for Ai,bi in zip(A,b):
        #     #     q = np.array(bi[:10].tolist()+[[0]]+[[0]])[:,0]
        #     #     joints_pose,ees_pose=skeleton.ub_fkine(np.rad2deg(q)) # this has all the frames of the robot joints
        #     #     tp_trans_mat = ees_pose[-1]
        #     #     # ees_pose[0].plot(frame="0", length=0.05, ax=ax,color='k',flo=(0.01,0.01,0.01))
        #     #     ees_pose[-1].plot(frame="task", length=0.05, ax=ax,color='k',flo=(0.01,0.01,0.01))
        #     #     sanity = SE3.Rt(Ai[-3:,-3:], tp_trans_mat.t)
        #     #     sanity.plot(frame="sanity", length=0.05, ax=ax,color="r",flo=(0.005,0.005,0.005))
        #     #     joints_pose[0].plot(frame="human", length=0.05, ax=ax,color='k',flo=(0.01,0.01,0.01))
        #     #     skeleton.plot_joints_ees_frame(joints_pose,ees_pose,show_ee_frames=False,show_joint_frames=False,ax=ax)
        #     # plt.show()
            
        #     # recondition to new TP
        #     tpgmm.conditionTPGMM(new_tp,len(input_index)-1,num_of_dims-1)
            
        # # sanity check the demonstrations, and reproduction wrenches
        # time_list,q_traj_list,qdot_traj_list,phri_wrench_list,phri_wrench_robot_list,command_wrench_list,rep_label_list = [],[],[],[],[],[],[]
        
        # for var_key, reps_dict in data_dict["repro_lfd"].items():
        #     rep_keys_shuffled = data_dict["dtw_order"][var_key]
        #     for i,rep_key in enumerate(rep_keys_shuffled):
        #         if i == 0:
        #             ref = reps_dict[rep_key].copy()
        #             target_interaction = reps_dict[rep_key].copy()
        #         else:
        #             target_interaction = reps_dict[rep_key].copy()
                    
        #         # DTW alignment
        #         paths = align_dtw(ref[:,q_index+qdot_index],target_interaction[:,q_index+qdot_index])
        #         warped_interaction = warp_target_to_ref(ref,target_interaction,paths)
        #         warped_command = warp_target_to_ref(ref,data_dict["lfd_commanded_wrench"][var_key][rep_key],paths)
        #         reps_dict[rep_key] = warped_interaction.copy()
                    
        #         # for plotting
        #         time_list.append(ref[:,0].copy()/ref[-1,0])
        #         q_traj_list.append(warped_interaction[:,q_index].copy())
        #         qdot_traj_list.append(warped_interaction[:,qdot_index].copy())
                               
        #         # phri wrench robot
        #         wrench = warped_interaction[:,emu_interaction_index].copy()
        #         mag = np.expand_dims(np.linalg.norm(wrench,axis=1),axis=1)
        #         wrench_with_mag = np.hstack((wrench,mag))
        #         phri_wrench_robot_list.append(wrench_with_mag)
                
        #         # phri wrench ft sensor
        #         wrench = warped_interaction[:,emu_transf_force_index].copy()
        #         mag = np.expand_dims(np.linalg.norm(wrench,axis=1),axis=1)
        #         wrench_with_mag = np.hstack((wrench,mag))
        #         phri_wrench_list.append(wrench_with_mag)
                
        #         # lfd wrench robot
        #         wrench = warped_interaction[:,emu_command_index].copy()
        #         mag = np.expand_dims(np.linalg.norm(wrench,axis=1),axis=1)
        #         wrench_with_mag = np.hstack((wrench,mag))
        #         command_wrench_list.append(wrench_with_mag)
                
        #         rep_label_list.append(f'{var_key}.{rep_key}')
                
        # split_plot_all(session_param["variants"],time_list,q_traj_list,rep_label_list,rep_split=session_param["num_rep"],data_type="q_rad",fig_label=f"q_rad repro {session_param['subject_id']}_{session_param['patient_id']}",plot=True)
        # split_plot_all(session_param["variants"],time_list,qdot_traj_list,rep_label_list,rep_split=session_param["num_rep"],data_type="qdot_rad",fig_label=f"qdot_rad repro {session_param['subject_id']}_{session_param['patient_id']}",plot=True)
        split = session_param["num_rep"]*len(session_param["variants"])
        
        all_wrenches = [demo_phri_wrench_list,demo_phri_wrench_robot_list]#,phri_wrench_list,phri_wrench_robot_list,command_wrench_list]
        all_wrench_tpes = ["demo_hhi","demo_hhi_robot"]#,"repro_hhi","repro_hhi_robot","repro_hhi_command"]
        for j,(wrench_list,wrench_type) in enumerate(zip(all_wrenches,all_wrench_tpes)):
            if j == 0 or j==1:
                x_list = demo_time_list
                label_list = demo_rep_label_list
                if j == 0:
                    stats_fig,stats_ax = None,None
            # else:
            #     x_list = time_list
            #     label_list = rep_label_list
                
            stats_fig,stats_ax = plot_multi_dim(
                x_list=x_list,
                data_list=wrench_list,
                dim=4,
                labels=[f"{i}.force {wrench_type}" for i in label_list],
                xtype="time",
                datatype="F",
                fig_label=f"force demo-phri-command compare {session_param['subject_id']}_{session_param['patient_id']}",
                split=split,
                legend=False,
                figsize=(16,10),
                color_set="bright",
                prev_fig=stats_fig,prev_ax=stats_ax,
                shuffle=j
            )
            
        for j,(wrench_list,wrench_type) in enumerate(zip(all_wrenches,all_wrench_tpes)):
            if j == 0 or j==1:
                x_list = demo_time_list
                label_list = demo_rep_label_list
            # else:
            #     x_list = time_list
            #     label_list = rep_label_list
            if j == len(all_wrenches)-1:
                legend=True
            else:
                legend=False
            
            time_list_per_var,gt_list_per_var,unique_var_id_list = split_plot_all(var_id_list,x_list,wrench_list,label_list,rep_split=session_param["num_rep"])
            plot_stats(
                time_list_per_var,
                gt_list_per_var,
                fig=stats_fig,
                axs=stats_ax,
                labels=[f"{x}.force {wrench_type}" for x in unique_var_id_list],
                relimit=True,
                split=split,
                color_set="bright",
                shuffle=j,
                legend=False
            )
        interactive_plot(stats_fig,stats_ax)
        
        plt.show()
            
        
        
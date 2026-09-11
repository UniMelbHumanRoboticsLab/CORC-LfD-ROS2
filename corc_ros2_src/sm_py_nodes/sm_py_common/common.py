import sys,os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from data_process.file_util_pkg import valid_test_combis
from data_visual.plot_pkg import plot_multi_dim,interactive_plot,create_custom_3d_fig,split_plot_all
import matplotlib.pyplot as plt

right_joints = ["skeleton_pelvis_t8_NA_y", "skeleton_pelvis_t8_NA_x", "skeleton_pelvis_t8_NA_z",
                "skeleton_right_c7_shoulder_x", "skeleton_right_c7_shoulder_y",
                "skeleton_right_shoulder_z", "skeleton_right_shoulder_x", "skeleton_right_shoulder_y",
                "skeleton_right_elbow_z", "skeleton_right_elbow_y",
                "skeleton_right_wrist_z", "skeleton_right_wrist_x"]
left_joints = ["skeleton_pelvis_t8_NA_y", "skeleton_pelvis_t8_NA_x", "skeleton_pelvis_t8_NA_z",
                "skeleton_left_c7_shoulder_x", "skeleton_left_c7_shoulder_y",
                "skeleton_left_shoulder_z", "skeleton_left_shoulder_x", "skeleton_left_shoulder_y",
                "skeleton_left_elbow_z", "skeleton_left_elbow_y",
                "skeleton_left_wrist_z", "skeleton_left_wrist_x"]

emu_header = ["x", "y", "z","xd", "yd", "zd","Fx", "Fy", "Fz","Mx", "My", "Mz","Fx_emu","Fy_emu","Fz_emu"]

q_usb = ['trunk_ie','trunk_aa','trunk_fe',
'clav_dep_ev','clav_prot_ret',
'shoulder_fe','shoulder_aa','shoulder_ie',
'elbow_fe','elbow_ps',
'wrist_fe','wrist_dev']

q_right = [f"q_{q}" for q in q_usb]
qdot_right = [f"qdot_{q}" for q in q_usb]

header = ["time","rep_num","sbmvmt"]+emu_header+q_right+qdot_right

time_index = 0
rep_index = 1
sbmvmt_index = 2
emu_kin_index = list(range(3, 9))
emu_wrench_index = list(range(9, 15))
emu_transf_force_index = list(range(15, 18))
q_index = list(range(18, 30-2))
qdot_index = list(range(30, len(header)-2))

_,_,var_train_combis = valid_test_combis(6,4)

def convert_to_tps(submovements,skeleton,num_of_dims,learn_input,learn_output,input_index,output_index):
    tps = []
    # fig,ax = create_custom_3d_fig()
    
    for sbmvmt in submovements:
        joints_pose,ees_pose=skeleton.ub_fkine(np.rad2deg(sbmvmt[q_index[0]:q_index[0]+12])) # this has all the frames of the robot joints
        tp_trans_mat = ees_pose[-1]
        
        A = np.eye(num_of_dims)
        if learn_input == "task":
            mid_input = int(len(input_index)/2)
            A[0:mid_input,0:mid_input] = tp_trans_mat.R
            A[mid_input:len(input_index),mid_input:len(input_index)] = tp_trans_mat.R
            
        if learn_output == "task":
            A[len(input_index):,len(input_index):] = tp_trans_mat.R
    
        B = np.hstack([sbmvmt[input_index],np.zeros(len(output_index))])
        tps.append(np.hstack([A.flatten(),B]))
        
    #     ees_pose[0].plot(frame="0", length=0.05, ax=ax,color='k',flo=(0.01,0.01,0.01))
    #     ees_pose[-1].plot(frame="1", length=0.05, ax=ax,color='k',flo=(0.01,0.01,0.01))
    #     skeleton.plot_joints_ees_frame(joints_pose,ees_pose,show_ee_frames=False,show_joint_frames=False,ax=ax)
    # plt.show(block=True)
    task_params = np.vstack([tps])
    return task_params
        



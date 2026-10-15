# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_process.file_util_pkg import get_raw_data

p_start = 1
sub_start  = 11
all_avg= []
all_avg_corc = []
for p in range(p_start,4):
    if p == 1:
        sm_num = 2
    else:
        sm_num=4
        
    for sub in range(sub_start,25):
        """ init session parameters and perform train / test / validation spliting"""
        session_data = {
            "exp_id":"exp1",
            "patient_id":f"p{p}",
            "subject_id":f"sub{sub}",
            "sbmvmt_num":sm_num,
            "num_rep":4,
            "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
        }
        subject_path = os.path.join(os.path.dirname(__file__), '../..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
        
        for var in session_data["variants"]:
            bias_path = f'{subject_path}/{var}/raw'
            avg_var_bias = np.load(f"{bias_path}/UBOAvgBias.npy")
            # print(f"\n{var}_install_Bias:",avg_var_bias,"\n")
            
            if os.path.isfile(f"{bias_path}/dtw_order.csv"):
                dtw_order = np.loadtxt(f"{bias_path}/dtw_order.csv", delimiter=',', dtype=int)
            else:
                dtw_order = np.random.permutation([1, 2, 3, 4])
                # print(f"DTW Order: {dtw_order}")
                np.savetxt(f"{bias_path}/dtw_order.csv", dtw_order, delimiter=',', fmt='%d')
        
            for n,rep in enumerate(dtw_order):
                # exist = os.path.exists(f'{subject_path}/{var}/processed/index/UBOIndex{rep}.txt') and os.path.exists(f'{subject_path}/{var}/processed/index/UBOStartEnd{rep}.txt') 
                exist = os.path.exists(f'{subject_path}/{var}/processed/UBORecord{rep}Log.csv')
                data_path = f'{subject_path}/{var}/raw/UBORecord{rep}Log.csv'
                # print("processing ",data_path)
                full_data,time_data_unscaled,corc_data_unscaled,q_traj_unscaled = get_raw_data(data_path)
                
                hi = full_data["timecode"].values.to_numpy()
                hi2 = full_data["corc time"].values
                from datetime import datetime
                def timecode_to_seconds(timecode):
                    """Convert HH:MM:SS.mmm to total seconds"""
                    corr_timecode = timecode[:-1]
                    time_obj = datetime.strptime(corr_timecode, '%H:%M:%S.%f')
                    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1_000_000
                    return total_seconds
                
                # Vectorize the function
                vectorized_convert = np.vectorize(timecode_to_seconds)
                result = vectorized_convert(hi)     
                unique_values = np.unique(result)
                differences = np.diff(unique_values)
                avg = np.mean(differences)
                if avg > 0.5:
                    # print(p,sub,var,rep)
                    # print(avg)
                    pass
                else:
                    all_avg.append(avg)
                    
                unique_values = np.unique(hi2)
                differences = np.diff(unique_values)
                avg = np.mean(differences)
                if avg > 0.5:
                    # print(p,sub,var,rep)
                    # print(avg)
                    pass
                else:
                    all_avg_corc.append(avg)
                    
                    
all_avg = np.array(all_avg)
all_avg_corc = np.array(all_avg_corc)
print("XSENS Sampling:",1/(np.mean(all_avg)))
print("CORC Sampling:",1/(np.mean(all_avg_corc)))

# plot_multi_dim(
#         time_list,
#         qdot_traj_list,
#         10,
#         rep_label_list,
#         'Time(s)',
#         "qdot_rad",
#         sharex=True,
#         semilogx=False,
#         fig_label=f"joint angle velocity - rad/s",
#         show_stats=True)

# plot_multi_dim(
#         time_list,
#         sbmvmt_list,
#         1,
#         rep_label_list,
#         'Time(s)',
#         "sbmvmt",
#         sharex=True,
#         semilogx=False,
#         fig_label="submovement")

# sanity check velocity (zero crossing should match peak positions)
# for i,(qtraj,qdot_traj) in enumerate(zip(q_traj_list,qdot_traj_list)):
#     plot_multi_dim(
#             [time_data_norm,time_data_norm],
#             [qtraj,qdot_traj],
#             10,
#             ["traj","vel"],
#             'Time(s)',
#             "qqdot",
#             sharex=True,
#             semilogx=False,
#             fig_label=f"qqdot_{i}",
#             show_zero_cross=True)



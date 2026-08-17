import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

sys.path.append(os.path.join(os.path.dirname(__file__)))
from post_process_pkg import lpf,calc_fixed_diff,calc_mag,segment_sbmvmts,rescale,align_dtw,warp_target_to_ref
from file_util_pkg import get_raw_data,separate_train_test_val,create_dir

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_visual.plot_pkg import plot_multi_dim,plot_3d_trajectory,split_plot_all,interactive_plot
from corc_py.pycorc_io.xsens.ub_pckg.ub import ub
from corc_py.pycorc_io.package_utils.unpack_json import get_subject_params
import FreeSimpleGUI as sg
#importing Pathlib
from pathlib import Path

p_start = 1
sub_start  = 11
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
        subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
        

        # separate variants for train test and save splits to csv
        # delete existing splits 
        [file.unlink() for file in Path(f'{subject_path}/splits').iterdir() if file.is_file()] 
        create_dir(f'{subject_path}/splits')
        separate_train_test_val(session_data["variants"],f'{subject_path}',num_train=5)
        # assert 0
        """ init xsens skeleton model """
        body_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/subject_measurements/{session_data["subject_id"]}')
        body_params_rbt,ft_grav,removed_bias = get_subject_params(body_path)
        init_bias = np.load(f"{body_path}/UBOAvgBias.npy")
        print("Init_install_Bias:",init_bias)
        skeleton = ub(body_params_rbt,model="ubo",arm_side="right")
        NUM_RFT = 3
        rft_keys = ["clav","ua","fa"]
        q = ['trunk_ie','trunk_aa','trunk_fe',
                'clav_dep_ev','clav_prot_ret',
                'shoulder_fe','shoulder_aa','shoulder_ie',
                'elbow_fe','elbow_ps']
        qdot = [f"{joint}_dot" for joint in q]
        tau = [f"tau_{joint}" for joint in q]
        
        """ compile all data across all variations-takes in current p1 actor - task """
        """ remove gravity effects from corc data """
        """ filter everything with low pass filter """
        """ get the external torque generated in that submovement """
        """ segment the submovements for each repetition """
        """ extract task parameters from each submovement """
        full_data_dict = {}
        time_list,q_traj_list,qdot_traj_list,tau_traj_list,hand_3d_traj_list,sbmvmt_list,rep_label_list = [],[],[],[],[],[],[]
        fc = 10  # cut-off frequency
        fs = 100 # resample frequency
        dt = 1/fs
        reprocess = False
        plot_sub = False
        for var in session_data["variants"]:
            full_data_dict[var] = {}
            bias_path = f'{subject_path}/{var}/raw'
            avg_var_bias = np.load(f"{bias_path}/UBOAvgBias.npy")
            print(f"\n{var}_install_Bias:",avg_var_bias,"\n")
            
            if os.path.isfile(f"{bias_path}/dtw_order.csv"):
                dtw_order = np.loadtxt(f"{bias_path}/dtw_order.csv", delimiter=',', dtype=int)
                print(f"Found DTW Order: {dtw_order}")
            else:
                dtw_order = np.random.permutation([1, 2, 3, 4])
                print(f"DTW Order: {dtw_order}")
                np.savetxt(f"{bias_path}/dtw_order.csv", dtw_order, delimiter=',', fmt='%d')
        
            # # uncomment to remove all processed folders to redo the processing
            # exist = os.path.exists(f'{subject_path}/{var}/processed/')
            # import shutil
            # if exist:
            #     shutil.rmtree(f'{subject_path}/{var}/processed/', ignore_errors=False)
            # assert 0
            for n,rep in enumerate(dtw_order):
                # exist = os.path.exists(f'{subject_path}/{var}/processed/index/UBOIndex{rep}.txt') and os.path.exists(f'{subject_path}/{var}/processed/index/UBOStartEnd{rep}.txt') 
                exist = os.path.exists(f'{subject_path}/{var}/processed/UBORecord{rep}Log.csv')

                # check if data processed?
                if exist and not reprocess:
                    # # uncomment for index removal shortcut
                    # try:
                    #     os.remove(f'{subject_path}/{var}/processed/index/UBOIndex{rep}.txt')
                    #     os.remove(f'{subject_path}/{var}/processed/index/UBOStartEnd{rep}.txt')
                    # except:
                    #     print(f"Already removed {subject_path}/{var}/processed/index")    
                    print(f'{subject_path}/{var}/processed/UBORecord{rep}Log.csv Processed')
                    full_df = pd.read_csv(f'{subject_path}/{var}/processed/UBORecord{rep}Log.csv')
                    print(f"Num of samples: {full_df.values.shape[0]}")
                    time_data_norm = full_df["norm_time"].values
                    q_traj_rad = full_df[q].values
                    qdot_traj_rad = full_df[qdot].values
                    hand_pos_traj = full_df[["x","y","z"]].values
                    tau_total = full_df[tau].values
                    indices_traj = full_df['index'].values[:,np.newaxis]
                    if n == 0:
                        reference_interaction = np.hstack((q_traj_rad,qdot_traj_rad,tau_total))
                        reference_index = indices_traj
                        ref_time_data_norm = time_data_norm
                else:
                    data_path = f'{subject_path}/{var}/raw/UBORecord{rep}Log.csv'
                    print("processing ",data_path)
                    full_data,time_data_unscaled,corc_data_unscaled,q_traj_unscaled = get_raw_data(data_path)
                    time_data_unscaled = time_data_unscaled - time_data_unscaled[0]
                    time_diff = np.diff(time_data_unscaled)
                    dt_unscaled = np.mean(time_diff)
                    if np.sum(time_diff<=0) > 0:
                        time_data_unscaled = np.linspace(time_data_unscaled[0],time_data_unscaled[-1],num=time_data_unscaled.shape[0],dtype=float)
                        
                    """ Deweight + Debias """
                    # add the initial removed bias, remove the initial installation bias collected
                    # remove actual installation bias collected during Take_bias
                    # remove gravity bias using xsens data FROM corc_data
                    robot_ees = skeleton.fkine(q_traj_unscaled.tolist())
                    removed_bias_all = []
                    total_weight_traj = []
                    for i in range(NUM_RFT):
                        # get initial removed_bias
                        removed_bias_all += removed_bias[rft_keys[i]]
                        
                        # calculate traj of current sensor weight in sensor frame
                        pose_traj = robot_ees[i + 1]
                        weight_comp = np.array([x * ft_grav[rft_keys[i]] for x in [0,0,-1]])
                        weight_traj = np.einsum('nji,j->ni', pose_traj.R, weight_comp) # transpose + matmul
                        weight_traj = np.hstack([weight_traj, np.zeros((weight_traj.shape[0], 3))])  # append nx3 zeros
                        total_weight_traj.append(weight_traj)
                    
                    total_weight_traj = np.hstack(total_weight_traj)
                    removed_bias_all = np.array(removed_bias_all)
                    corc_data_unscaled = corc_data_unscaled + removed_bias_all - avg_var_bias - total_weight_traj - init_bias
                    
                    """ Filter """
                    # filter the original data
                    filter = "bp"
                    plot_results = True # check if filter fucks up the data
                    corc_data   = lpf(time_data_unscaled,corc_data_unscaled,ts=dt_unscaled,fc=fc,filter_type="low",datatype=f"wrenches_{var}-Rep{rep}",   plot_results=plot_results)
                    q_traj      = lpf(time_data_unscaled,q_traj_unscaled,   ts=dt_unscaled,fc=fc,datatype=f"q_{var}-Rep{rep}",          plot_results=plot_results)
                    # calculate joint space kinematic and filter
                    qdot_traj   = calc_fixed_diff(q_traj, dt=dt_unscaled)
                    qdot_traj   = lpf(time_data_unscaled, qdot_traj, ts=dt_unscaled, fc=1.5, datatype=f"qdot_{var}-Rep{rep}", plot_results=plot_results)
                    # calculate and filter task space kinematic data for active movement segmentation
                    hand_pos_traj   = skeleton.fkine(q_traj.tolist())[0].t
                    hand_vel        = calc_fixed_diff(hand_pos_traj, dt=dt_unscaled)
                    hand_vel        = lpf(time_data_unscaled,hand_vel,ts=dt_unscaled,fc=1.5,datatype=f"hand_vel_{var}-Rep{rep}",plot_results=plot_results)
                    hand_speed      = calc_mag(hand_vel)
                    # calculate and filter joint space torques generated by each RFT and total
                    taus_traj       = skeleton.get_joints_torques_traj(q_traj, corc_data)
                    for rft_key in rft_keys:
                        taus_traj[rft_key]["filtered"] = lpf(time_data_unscaled,taus_traj[rft_key]["raw"],ts=dt_unscaled,fc=fc,datatype=f"tau_{rft_key}_{var}-Rep{rep}",plot_results=False)
                    taus_traj["total"]["filtered"] = lpf(time_data_unscaled,taus_traj["total"]["raw"],ts=dt_unscaled,fc=fc,datatype=f"tau_total_{var}-Rep{rep}",plot_results=plot_results)
                    
                    
                    """ Uniform Resampling """
                    # downsample for easier model learning
                    plot_results = False
                    time_data       = np.arange(time_data_unscaled[0],time_data_unscaled[-1]+dt_unscaled,1/fs,dtype=float)
                    corc_data       = rescale(time_data_unscaled, corc_data,        time_data,  datatype=f"wrenches_{var}-Rep{rep}",      plot_results=plot_results)
                    q_traj          = rescale(time_data_unscaled, q_traj,           time_data,  datatype=f"q_{var}-Rep{rep}",             plot_results=plot_results)
                    qdot_traj       = rescale(time_data_unscaled, qdot_traj,        time_data,  datatype=f"qdot_{var}-Rep{rep}",          plot_results=plot_results)
                    hand_pos_traj   = rescale(time_data_unscaled, hand_pos_traj,    time_data,  datatype=f"hand_pos_{var}-Rep{rep}",      plot_results=plot_results)
                    hand_speed      = rescale(time_data_unscaled, hand_speed,       time_data,  datatype=f"hand_speed_{var}-Rep{rep}",    plot_results=plot_results)
                    for rft_key in rft_keys+["total"]:
                        taus_traj[rft_key]["filtered-rescaled"] = rescale(time_data_unscaled, taus_traj[rft_key]["filtered"], time_data,  datatype=f"tau_{rft_key}_{var}-Rep{rep}",      plot_results=plot_results)
            
                    """ Active Movement Segment """
                    # create the following subdirectories if needed
                    create_dir(f'{subject_path}/{var}/processed/index')
                    # extract active movement
                    _,start_end_indices = segment_sbmvmts(time_data,hand_pos_traj,hand_speed,1,data_path=f'{subject_path}/{var}/processed/index/UBOStartEnd{rep}.txt',redo=False,skeleton={"ub":skeleton,"q":q_traj})
                    actual_time_data = time_data[start_end_indices[0]:start_end_indices[-1]]
                    time_data       = time_data[start_end_indices[0]:start_end_indices[-1]]-time_data[start_end_indices[0]]
                    corc_data       = corc_data[start_end_indices[0]:start_end_indices[-1],:]
                    q_traj          = q_traj[start_end_indices[0]:start_end_indices[-1],:10] # from here just collect the first 10 joints
                    
                    qdot_traj       = qdot_traj[start_end_indices[0]:start_end_indices[-1],:10]
                    hand_pos_traj   = hand_pos_traj[start_end_indices[0]:start_end_indices[-1],:]
                    hand_speed      = hand_speed[start_end_indices[0]:start_end_indices[-1],:]
                    for rft_key in rft_keys+["total"]:
                        taus_traj[rft_key]["filtered-rescaled"] = taus_traj[rft_key]["filtered-rescaled"][start_end_indices[0]:start_end_indices[-1],:]
                    
                    """ Time Normalization """
                    # use first demo as the reference for overall time alignment
                    # if var == session_data["variants"][0] and n == 0:
                    if n == 0:
                        ref_time_data_norm  = time_data / time_data[-1]
                        ref_time_data = time_data
                        time_data_norm       = ref_time_data_norm
                    else:
                        time_data_norm       = time_data / time_data[-1]
            
                    """conver to radians """
                    q_traj_rad      = np.deg2rad(q_traj)
                    qdot_traj_rad   = np.deg2rad(qdot_traj)
                    
                    """ DTW Realignment """
                    # use first demo as the reference for the DTW alignment
                    if n == 0:
                        reference_interaction = np.hstack((q_traj_rad,qdot_traj_rad,taus_traj["total"]["filtered-rescaled"]))
                        target_interaction = np.hstack((q_traj_rad,qdot_traj_rad,taus_traj["total"]["filtered-rescaled"]))
                    else:
                        target_interaction = np.hstack((q_traj_rad,qdot_traj_rad,taus_traj["total"]["filtered-rescaled"]))
                    
                    align_dim = 20
                    paths = align_dtw(reference_interaction[:,0:align_dim],target_interaction[:,0:align_dim])
                    placehold = sg.popup('',location=(1550,300),non_blocking=True)  
                    f1,a1 = plot_multi_dim(
                            [time_data,ref_time_data,ref_time_data],
                            [q_traj_rad,warp_target_to_ref(reference_interaction,q_traj_rad,paths),reference_interaction[:,0:10]],
                            10,
                            ["ori","dtw","ref"],
                            'Time(s)',
                            "q_rad",
                            sharex=True,
                            semilogx=False,
                            fig_label=f"joint angle rad {align_dim}",
                            show_stats=False,loc="+2000+800")
                    f2,a2 = plot_multi_dim(
                            [time_data,ref_time_data,ref_time_data],
                            [qdot_traj_rad,warp_target_to_ref(reference_interaction,qdot_traj_rad,paths),reference_interaction[:,10:20]],
                            10,
                            ["ori","dtw","ref"],
                            'Time(s)',
                            "qdot_rad",
                            sharex=True,
                            semilogx=False,
                            fig_label=f"joint angle vel rad/s {align_dim}",
                            show_stats=False,loc="+2900+800",
                            show_zero_cross=True)
                    f3,a3 = plot_multi_dim(
                            [time_data,ref_time_data,ref_time_data],
                            [taus_traj["total"]["filtered-rescaled"],warp_target_to_ref(reference_interaction,taus_traj["total"]["filtered-rescaled"],paths),reference_interaction[:,20:]],
                            10,
                            ["ori","dtw","ref"],
                            'Time(s)',
                            "tau",
                            sharex=True,
                            semilogx=False,
                            fig_label=f"joint torque {align_dim}",
                            show_stats=False,loc="+3800+800")
                    
                    interactive_plot(f1,a1)
                    interactive_plot(f2,a2)
                    interactive_plot(f3,a3)
                    
                    """ Submovement Segment using unaligned movement"""
                    # segment rescaled but not aligned submovements
                    haih = actual_time_data[:np.newaxis]
                    indices_traj,sbmvmt_indices = segment_sbmvmts(actual_time_data[:np.newaxis],hand_pos_traj,hand_speed,session_data["sbmvmt_num"],data_path=f'{subject_path}/{var}/processed/index/UBOIndex{rep}.txt',redo=False,skeleton={"ub":skeleton,"q":q_traj})
                    # f3,a3 = plot_multi_dim(
                    #         [actual_time_data,actual_time_data,actual_time_data],
                    #         [indices_traj,warp_target_to_ref(reference_interaction,indices_traj,paths),reference_index],
                    #         1,
                    #         ["ori","dtw","ref"],
                    #         'Time(s)',
                    #         "tau",
                    #         sharex=True,
                    #         semilogx=False,
                    #         fig_label=f"joint torque {align_dim}",
                    #         show_stats=False,loc="+3800+800")
                    # interactive_plot(f3,a3)
                    # plt.show()
                    
                    """ Extract TP from the unaligned movement"""
                    create_dir(f'{subject_path}/{var}/processed/tp')
                    # extract task parameters from each submovements (Input: joint kinematics, output: joint torques)
                    num_dim = len(q)+len(qdot)+len(tau)
                    task_params_A   = np.vstack([np.eye(num_dim).flatten() for _ in range(sbmvmt_indices.shape[0])])
                    task_params_b   = np.hstack([q_traj_rad[sbmvmt_indices,:], qdot_traj_rad[sbmvmt_indices,:],np.zeros((sbmvmt_indices.shape[0],len(tau)))])  # tau should have zero offset
                    task_params     = np.hstack([task_params_A, task_params_b])
                    tp_df = pd.DataFrame(task_params, columns=[f"A{j}" for j in range(1,num_dim*num_dim+1)] + [f"b{j}" for j in range(1,num_dim+1)])
                    tp_df.to_csv(f'{subject_path}/{var}/processed/tp/UBOTP{rep}Log.csv',index=False)
                    
                    """ Align interaction using DTW path """
                    corc_data       = warp_target_to_ref(reference_interaction,corc_data,paths)
                    q_traj_rad      = warp_target_to_ref(reference_interaction,q_traj_rad,paths)
                    qdot_traj_rad   = warp_target_to_ref(reference_interaction,qdot_traj_rad,paths)
                    hand_pos_traj   = warp_target_to_ref(reference_interaction,hand_pos_traj,paths)

                    for rft_key in rft_keys+["total"]:
                        taus_traj[rft_key]["filtered-rescaled"] = warp_target_to_ref(reference_interaction,taus_traj[rft_key]["filtered-rescaled"],paths)
            
                    """ Save Post Processed Data """
                    # compile all data
                    full_data_dict[var][rep] = {
                        "time": time_data_norm,
                        "corc": corc_data,
                        "q": q_traj,
                        "qdot": qdot_traj,
                        "taus": taus_traj,
                        "sbmvmt": sbmvmt_indices,
                        "task_params": task_params_b
                    }
                    tau_total = taus_traj["total"]["filtered-rescaled"]
                    t_warped =  warp_target_to_ref(reference_interaction,actual_time_data[:,np.newaxis],paths)
                    indices_traj = warp_target_to_ref(reference_interaction,indices_traj,paths)
                    full_processed_data = np.hstack([t_warped,ref_time_data_norm[:,np.newaxis],indices_traj,q_traj_rad,qdot_traj_rad,tau_total,hand_pos_traj])
                    
                    df_column = ["actual_time","norm_time","index"]+q+qdot+tau+["x","y","z"]
                    full_df = pd.DataFrame(full_processed_data,columns=df_column)
                    full_df.to_csv(f'{subject_path}/{var}/processed/UBORecord{rep}Log.csv',index=False)
        
                # for plotting
                time_list.append(ref_time_data_norm)
                q_traj_list.append(q_traj_rad)
                qdot_traj_list.append(qdot_traj_rad)
                tau_traj_list.append(tau_total)
                hand_3d_traj_list.append(hand_pos_traj)
                sbmvmt_list.append(indices_traj)  
                rep_label_list.append(f'{var}.Rep{rep}')
        if plot_sub:
            fig,ax = plot_3d_trajectory(traj_list=hand_3d_traj_list,label_list=rep_label_list,label=f"{session_data['subject_id']}_{session_data['patient_id']}")
            split_plot_all(session_data["variants"],time_list,q_traj_list,rep_label_list,rep_split=session_data["num_rep"],data_type="q_rad",fig_label=f"q_rad data {session_data['subject_id']}_{session_data['patient_id']}",plot=True)
            split_plot_all(session_data["variants"],time_list,sbmvmt_list,rep_label_list,rep_split=session_data["num_rep"],data_type="index",fig_label=f"index data {session_data['subject_id']}_{session_data['patient_id']}",plot=True)
            split_plot_all(session_data["variants"],time_list,tau_traj_list,rep_label_list,rep_split=session_data["num_rep"],data_type="tau",fig_label=f"tau data {session_data['subject_id']}_{session_data['patient_id']}",plot=True)
            split_plot_all(session_data["variants"],time_list,qdot_traj_list,rep_label_list,rep_split=session_data["num_rep"],data_type="qdot_rad",fig_label=f"qdot_rad data {session_data['subject_id']}_{session_data['patient_id']}",plot=True)
            plt.show()
            
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



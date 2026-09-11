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

sys.path.append(os.path.join(os.path.dirname(__file__)))
from post_process_pkg import lpf,calc_fixed_diff,align_dtw,warp_target_to_ref
from file_util_pkg import get_raw_data_with_vel,create_dir

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corc_py.pycorc_io.package_utils.unpack_json import get_subject_params
from corc_py.pycorc_io.xsens.ub_pckg.ub import ub
from data_visual.plot_pkg import plot_multi_dim,interactive_plot,create_custom_3d_fig,split_plot_all
from data_analyse.metrics_pkg import compute_cross_correlate
from lfd.tpgmm_pkg.tpgmm_util import arrange_data,get_optim_nbGauss,unpack_tp
from lfd.tpgmm_pkg.TPGMM import TPGMM

from corc_ros2_src.sm_py_nodes.sm_py_common.common import q_index,qdot_index,emu_kin_index,emu_transf_force_index,sbmvmt_index,var_train_combis,convert_to_tps

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
            "subject_id":f"sub{sub}",
            "sbmvmt_num":sm_num,
            "num_rep":3,
            "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
        }
        subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/ros2_recordings/{session_param["exp_id"]}/{session_param["patient_id"]}/{session_param["subject_id"]}')
        """ init xsens skeleton model """
        body_path = os.path.join(os.path.dirname(__file__), '..',f'logs/ros2_recordings/{session_param["exp_id"]}/subject_measurements/{session_param["subject_id"]}')
        body_params_rbt,ft_grav,removed_bias = get_subject_params(body_path,"emu")
        skeleton = ub(body_params_rbt,model="emu",arm_side="right",ft_grav=ft_grav,removed_bias=removed_bias)
        
        data_dict = {
            "demo":{},
            "old_tp":{},
            "tpgmms":{},
            "new_tp":{},
            }
        
        # B4 LfD state: to simulate compiling demo state
        for var in session_param["variants"]:
            data_dict["demo"][var] = {}
            for n,rep in enumerate([1,2,3]):
                data_path = f'{subject_path}/{var}/demo/IntRecord{rep}Log.csv'
                # print("processing ",data_path)
                full_data,time_data_unscaled,q_traj,qdot_traj = get_raw_data_with_vel(data_path)
                time_data_unscaled = time_data_unscaled - time_data_unscaled[0]
                time_diff = np.diff(time_data_unscaled)
                dt_unscaled = np.mean(time_diff)
                fs_unscaled = 1/dt_unscaled
                if np.sum(time_diff<=0) > 0:
                    time_data_unscaled = np.linspace(time_data_unscaled[0],time_data_unscaled[-1],num=time_data_unscaled.shape[0],dtype=float)
                
                """
                sanity check the online joint kinematics and filtering
                """
                # filter the original data
                # calculate joint space kinematic and filter
                qdot_traj_filt_filt   = calc_fixed_diff(q_traj, dt=dt_unscaled)
                qdot_traj_filt_filt   = lpf(time_data_unscaled, qdot_traj_filt_filt, ts=dt_unscaled, fc=10, datatype=f"qdot_{var}-Rep{rep}", plot_results=False)
                lags,corr,max_lag = compute_cross_correlate(qdot_traj[:,:10],qdot_traj_filt_filt[:,:10])
                time_delay = max_lag / fs_unscaled * 1000

                # f3,a3 = plot_multi_dim(
                #         [time_data_unscaled,time_data_unscaled,time_data_unscaled],
                #         [q_traj[:,:10],qdot_traj[:,:10],qdot_traj_filt_filt[:,:10]],
                #         10,
                #         ["q_deg","qdot_deg","qdot_deg_ff"],
                #         'Time(s)',
                #         "qkine_deg",
                #         sharex=True,
                #         semilogx=False,
                #         fig_label=f"joint kinematic sanity check {var} rep{rep}. {time_delay} ms",
                #         show_stats=False,loc="+3800+800")
                # interactive_plot(f3,a3)
                
                data_dict["demo"][var][f"rep_{rep}"] = full_data.values
        
        # during LFD state
        # step 1: preprocessing 
        all_data = []
        time_list,q_traj_list,qdot_traj_list,wrench_list,sbmvmt_list,rep_label_list = [],[],[],[],[],[]
        for var_key, reps_dict in data_dict["demo"].items():
            # shuffle the reps for each variation
            rep_keys_shuffled = list(reps_dict.keys())
            
            random.shuffle(rep_keys_shuffled)
            
            data_dict["old_tp"][var_key] = {}
            data_dict["new_tp"][var_key] = {}
            for i,rep_key in enumerate(rep_keys_shuffled):
                if i == 0:
                    ref = reps_dict[rep_key].copy()
                    target_interaction = reps_dict[rep_key].copy()
                else:
                    target_interaction = reps_dict[rep_key].copy()
                
                
                # DTW alignment
                paths = align_dtw(ref[:,q_index+qdot_index],target_interaction[:,q_index+qdot_index])
                warped_interaction = warp_target_to_ref(ref,target_interaction,paths)
                reps_dict[rep_key] = warped_interaction.copy()
                all_data.append(warped_interaction[:,input_index+output_index].copy())
                # f1,a1 = plot_multi_dim(
                #         [target_interaction[:,0],ref[:,0],ref[:,0]],
                #         [target_interaction[:,q_index],warped_interaction[:,q_index],ref[:,q_index]],
                #         10,
                #         ["ori","dtw","ref"],
                #         'Time(s)',
                #         "q_deg",
                #         sharex=True,
                #         semilogx=False,
                #         fig_label=f"joint angle {var_key} {rep_key}",
                #         show_stats=False,loc="+2000+800")
                # f2,a2 = plot_multi_dim(
                #         [target_interaction[:,0],ref[:,0],ref[:,0]],
                #         [target_interaction[:,qdot_index],warped_interaction[:,qdot_index],ref[:,qdot_index]],
                #         10,
                #         ["ori","dtw","ref"],
                #         'Time(s)',
                #         "qdot_deg",
                #         sharex=True,
                #         semilogx=False,
                #         fig_label=f"joint angle vel {var_key} {rep_key}",
                #         show_stats=False,loc="+2900+800",)
                # interactive_plot(f1,a1)
                # interactive_plot(f2,a2)
        
                # get submovement indices
                submovement_indices = warped_interaction[:,sbmvmt_index]
                submovement_indices[-1] += 1
                segments = np.r_[0, np.flatnonzero(np.diff(submovement_indices) != 0) + 1]
                
                # extract and convert the segmented movement into meaningful TPs
                submovements = warped_interaction[segments,:]
                data_dict["old_tp"][var_key][f"{rep_key}"] = convert_to_tps(submovements,skeleton,num_of_dims,learn_input,learn_output,input_index,output_index)
                if i == 0:
                    data_dict["new_tp"][var_key] = convert_to_tps(submovements,skeleton,num_of_dims,learn_input,learn_output,input_index,output_index)
                    
                # for plotting
                time_list.append(ref[:,0].copy())
                q_traj_list.append(warped_interaction[:,q_index].copy())
                qdot_traj_list.append(warped_interaction[:,qdot_index].copy())
                wrench_list.append(warped_interaction[:,emu_transf_force_index].copy())
                sbmvmt_list.append(np.expand_dims(warped_interaction[:,sbmvmt_index].copy(),axis=1))
                rep_label_list.append(f'{var_key}.{rep_key}')
        # split_plot_all(session_param["variants"],time_list,q_traj_list,rep_label_list,rep_split=session_param["num_rep"],data_type="q_deg",fig_label=f"q_rad data {session_param['subject_id']}_{session_param['patient_id']}",plot=True)
        # split_plot_all(session_param["variants"],time_list,sbmvmt_list,rep_label_list,rep_split=session_param["num_rep"],data_type="index",fig_label=f"index data {session_param['subject_id']}_{session_param['patient_id']}",plot=True)
        # split_plot_all(session_param["variants"],time_list,wrench_list,rep_label_list,rep_split=session_param["num_rep"],data_type="F",fig_label=f"tau data {session_param['subject_id']}_{session_param['patient_id']}",plot=True)
        # split_plot_all(session_param["variants"],time_list,qdot_traj_list,rep_label_list,rep_split=session_param["num_rep"],data_type="qdot_deg",fig_label=f"qdot_rad data {session_param['subject_id']}_{session_param['patient_id']}",plot=True)
        # plt.show()
        
        # step 2: LFD code
        # get the optim number of gauss once
        all_data = np.vstack(all_data)
        total_time = 0
        num_of_frames = max_submvmt+1
        start = times.perf_counter()
        num_of_gauss,bic_df = get_optim_nbGauss(all_data)
        # num_of_gauss = 1
        end = times.perf_counter()
        bic_time = (end - start)
        total_time += (end - start)
        print(f"BIC Search:Elapsed \t= {(bic_time):.4f}s\n")
                
        # start learning for each variation
        for var_key, reps_dict in data_dict["demo"].items():
            cur_var = int(re.search(r'\d+', var_key).group())
            print(cur_var)
            print(f"\n==================================\n{var_key} Training")
            # select a valid training combi for this variation
            valid_train = var_train_combis[var_key]
            selected_combi = [f"var_{i}" for i in random.choice(valid_train)]
            print(f"Valid train combination for {var_key}: {selected_combi}")
            
            # compile the train list
            train_list = []
            for train_var in selected_combi:
                if train_var in data_dict["demo"]:
                    for rep_key,rep_data in data_dict["demo"][train_var].items():
                        rep_tps = data_dict["old_tp"][train_var][rep_key]
                        rep_dict = {
                			"time":rep_data[:,0],
                			"data":rep_data[:,input_index+output_index],
                			"tp":rep_tps,
                            "id":f"{train_var}_{rep_key}"
                            }
                        
                        train_list.append(rep_dict) 
                else:
                    print(f"{train_var} does not exist")
            
            if len(train_list) < 1:
                fail = True
                print("Not enough training data. Skipping training")
            else:
                # compile the test list
                test_list = []
                for rep_key,rep_data in data_dict["demo"][var_key].items():
                    rep_tps = data_dict["old_tp"][var_key][rep_key]
                    rep_dict = {
            			"time":rep_data[:,0],
            			"data":rep_data[:,input_index+output_index],
            			"tp":rep_tps,
                        "id":f"{var_key}_{rep_key}"
                        }
                    test_list.append(rep_dict) 
    
                train_tp_data,train_sample_list = arrange_data(train_list,num_of_frames,num_of_dims)
                _,test_sample_list = arrange_data(test_list,num_of_frames,num_of_dims)
                
                fail = True
                retry = 0
                while fail and retry < 5:
                    print(f"\n====================\nTraining Attempt {retry}")
    
                    start = times.perf_counter()
                    tpgmm = TPGMM(num_of_gauss,num_of_frames,num_of_dims,priors=np.ones((num_of_gauss,)),kP=0,kV=0,diagRegFact=1e-8,version="fast")
                    # tpgmm.model_id = f'{session_param["exp_id"]}-{session_param["patient_id"]}-{session_param["subject_id"]}-tpgmm_{combi_num}_{sample_num}'
                    # tpgmm.bic = bic_df
                    tpgmm.init_gmm(train_tp_data)
                    LL,fail = tpgmm.fit_em(nbMinSteps=5,nbMaxSteps=100,maxDiffLL=1e-11,updateComp=np.ones((3,1)),tp_data=train_tp_data)
                    end = times.perf_counter()
                    total_time += (end - start)
                    print(f"Training Success?: {not fail}")
                    
                    print(f"Fast Training:Elapsed \t= {(end - start):.4f}s")
                    tpgmm.training_time = (end - start)
                    tpgmm.bic_time = bic_time
                    retry += 1
                if fail:
                    tpgmm.training_status = "Fail"
                    print(f"{subject_path} failed to converge| Pending post check")
                else:
                    tpgmm.training_status = "Success"
                    
                
                create_dir(f"{subject_path}/tpgmm")
                tpgmm_file_path = f'{subject_path}/tpgmm/tpgmm_{var_key}.pkl'
                with open(tpgmm_file_path, 'wb') as file:
                    # Dump data with highest protocol for best performance
                    pickle.dump(tpgmm,file, protocol=pickle.HIGHEST_PROTOCOL)
                # data_dict["tpgmms"][var_key] = tpgmm
            
                # after LFD state
                # step 3: deploy LfD
                new_tp = unpack_tp(data_dict["new_tp"][var_key],num_of_frames,num_of_dims)
                # recondition to new TP
                tpgmm.conditionTPGMM(new_tp,len(input_index)-1,num_of_dims-1)
                print(f"{tpgmm.norm}")
                
                recon_list = []
                recon_lut_list = []
                gt_list = []
                id_list = []
                time_list = []
                for val_dict,sample in zip(test_list,test_sample_list): 
                    DataIn  = sample.Data[:,0:len(input_index)] # REMINDER: convert the degs to rad in actual deployment as xsens gives degrees
                    DataOut = sample.Data[:,len(input_index):]
                    
                    id_list.append(val_dict["id"])
                    gt_list.append(DataOut)
                    time_list.append(val_dict["time"])
                    expected_data = np.zeros((len(DataIn),tpgmm.nbOut))
                    
                    for t,data in enumerate(DataIn):
                        expected_data_sample,priorsOutput_sample,expSigma_sample = tpgmm.GMR_sample(data)
                        expected_data[t,:] = expected_data_sample
                    recon_list.append(expected_data)
                    
                stats_fig,stats_ax = plot_multi_dim(
                    x_list=time_list,
                    data_list=gt_list,
                    dim=session_param["num_rep"],
                    labels=[f"{i}.GT" for i in id_list],
                    xtype="time",
                    datatype="F",
                    fig_label=f"{session_param['exp_id']} Test Compare",
                    split=session_param["num_rep"],
                    legend=False,
                    figsize=(18.33333*0.393701,8.5*0.393701),
                    color_set="bright",
                )
                stats_fig,stats_ax = plot_multi_dim(
                    x_list=time_list,
                    data_list=recon_list,
                    dim=3,
                    labels=[f"{i}.TPGMM" for i in id_list],
                    xtype="time",
                    datatype="F",
                    fig_label="Commanded Force",
                    split=session_param["num_rep"],
                    legend=False,
                    prev_fig=stats_fig,prev_ax=stats_ax,
                    color_set="bright",
                    shuffle=5
                )
    
                interactive_plot(stats_fig,stats_ax)
                plt.show(block=True)
                pass
        
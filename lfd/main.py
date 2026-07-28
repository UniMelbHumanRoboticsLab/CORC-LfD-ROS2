import sys,os
import numpy as np
np.set_printoptions(suppress=True,precision=4) # suppress scientific notation
import time as times
from datetime import datetime

from lut_pkg.LookUpTable import LookUpTable
from tpgmm_pkg.TPGMM import TPGMM
from tpgmm_pkg.tpgmm_util import arrange_data,get_optim_nbGauss,save_results
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_visual.plot_pkg import plot_multi_dim,plot_stats,interactive_plot,split_plot_all
from data_process.file_util_pkg import create_dir,compile_train_val_test_data
from data_analyse.stats_pkg import compute_central_tendency
import matplotlib.pyplot as plt

plt_results = False
retrain = False
# retrain = True
deploy = True

GREEN = '\033[92m'

RED = '\033[91m'
RESET = '\033[0m'

p_start = 1
sub_start  = 11

total_monte = 4

# 2 train var - 2 combinations
# 3 train var - 10 combinations
# 4 train var - 6 combinations (prev), 7 (current)
# 5 train var - 6 combinations

for train_var_num,total_combi in zip([4,5],[7,6]):
    for p in range(p_start,4):
        total_time = 0
        if p == 1:
            sm_num = 2
        else:
            sm_num=4
            
        for sub in range(sub_start,25):
            session_data = {
                "exp_id":f"exp1_trained_{train_var_num}",
                "patient_id":f"p{p}",
                "subject_id":f"sub{sub}",
                "sbmvmt_num":sm_num,
                "num_rep":4,
                "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
            }
            subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
            
            for combi_num in range(0,total_combi):
                for sample_num in range(0,total_monte):
                    print(f"\n====={session_data['exp_id']}-{session_data['patient_id']}-{session_data['subject_id']}-{combi_num}-{sample_num} =================")
                    train_list,val_list,test_list = compile_train_val_test_data(session_data,subject_path,combi_num,sample_num,False)
                    all_data = []
                    for train in train_list:
                        all_data.append(train["data"])
                    all_data = np.vstack(all_data)
                    print(f"Number of samples:{all_data.shape[0]}")
            
                    """
                    Transform training and test data
                    """
                    num_of_frames = session_data["sbmvmt_num"]+1
                    num_of_dims = 30
                    lastInputIndex = 19  # always arrange data such that [input indices, output indices]
                    lastOutputIndex = 29
                    train_tp_data,train_sample_list = arrange_data(train_list,num_of_frames,num_of_dims)
                    _,val_sample_list = arrange_data(val_list,num_of_frames,num_of_dims)
                    _,test_sample_list = arrange_data(test_list,num_of_frames,num_of_dims)
                    
            
                    """
                    Initialize and fit the GMM 
                    """  
                    tpgmm_file_path = f'{subject_path}/repro/tpgmm_{combi_num}_{sample_num}.pkl'
                    exist = os.path.exists(tpgmm_file_path)

                    if exist:
                        with open(tpgmm_file_path, 'rb') as outp:                        
                            tpgmm = pickle.load(outp)
                        print(type(tpgmm))
                            assert isinstance(tpgmm, TPGMM)
                            num_of_gauss = tpgmm.num_of_gauss
                            LL = tpgmm.converged_LL
                            
                            if len(LL) > 1:
                                print(f"{tpgmm.model_id} found")
                                stat_info = os.stat(tpgmm_file_path)
                                creation_time = stat_info.st_mtime
                                readable_time = datetime.fromtimestamp(creation_time)
                                print(f"File created: {readable_time}")
                                color = GREEN if tpgmm.training_status == "Success" else RED
                                print(f"{color}Training Status: {tpgmm.training_status}{RESET}")
                                print(f"Number of Gaussians: {num_of_gauss}")
                                print(f"Converged Likelihood: {LL}")
                                print(f"BIC Elapsed \t\t= {tpgmm.bic_time:.4f}s")
                                print(f"Training: Elapsed \t= {tpgmm.training_time:.4f}s")
                                total_time += tpgmm.training_time+tpgmm.bic_time
                    else:
                        fail = True
                        retry = 0
                        while fail and retry < 5:
                            print(f"\n====================\nTraining Attempt {retry}")
                            start = times.perf_counter()
                            hi = train_tp_data.transpose(0, 2, 1).reshape(-1, train_tp_data.shape[1])
                            num_of_gauss,bic_df = get_optim_nbGauss(all_data)
                            end = times.perf_counter()
                            bic_time = (end - start)
                            total_time += (end - start)
                            
                            start = times.perf_counter()
                            tpgmm = TPGMM(num_of_gauss,num_of_frames,num_of_dims,priors=np.ones((num_of_gauss,)),kP=0,kV=0,diagRegFact=1e-8,version="fast")
                            tpgmm.model_id = f'{session_data["exp_id"]}-{session_data["patient_id"]}-{session_data["subject_id"]}-tpgmm_{combi_num}_{sample_num}'
                            tpgmm.bic = bic_df
                            tpgmm.init_gmm(train_tp_data)
                            LL,fail = tpgmm.fit_em(nbMinSteps=2,nbMaxSteps=100,maxDiffLL=1e-5,updateComp=np.ones((3,1)),tp_data=train_tp_data)
                            end = times.perf_counter()
                            total_time += (end - start)
                            print(f"Training Success?: {not fail}")
                            print(f"BIC Search:Elapsed \t= {(bic_time):.4f}s")
                            print(f"Fast Training:Elapsed \t= {(end - start):.4f}s")
                            tpgmm.training_time = (end - start)
                            tpgmm.bic_time = bic_time
                            retry += 1
                        if fail:
                            tpgmm.training_status = "Fail"
                            print(f"{subject_path} failed to converge| Pending post check")
                        else:
                            tpgmm.training_status = "Success"
                        create_dir(f"{subject_path}/repro")
                        with open(tpgmm_file_path, 'wb') as file:
                            # Dump data with highest protocol for best performance
                            pickle.dump(tpgmm,file, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    """
                    Initialize the LookUp Table
                    """  
                    lut = LookUpTable(all_data, lastInputIndex, lastOutputIndex)
                    
                    """
                    Deployment pipeline
                    """
                    if deploy and not exist:
                        """
                        compile training data
                        """
                        gt_list = []
                        id_list = []
                        time_list = []
                        var_id_list = []
                        for train_dict,sample in zip(train_list,train_sample_list): 
                            DataIn  = sample.Data[:,0:lastInputIndex+1]
                            DataOut = sample.Data[:,lastInputIndex+1:]
                            sample_id = train_dict["id"].split(".")
                            var_id_list.append(sample_id[0])
                        
                            time_list.append(train_dict["time"])
                            gt_list.append(DataOut)
                            id_list.append(train_dict["id"])
                        # extract the validation ground truths for analysis
                        for i,(val_dict,sample) in enumerate(zip(val_list,val_sample_list)): 
                            DataOut = sample.Data[:,lastInputIndex+1:]
                            sample_id = val_dict["id"].split(".")
                            
                            # insert just before its training variation
                            var_id_list.insert((i+1)*4-1,sample_id[0])
                            time_list.insert((i+1)*4-1,val_dict["time"])
                            gt_list.insert((i+1)*4-1,DataOut)
                            id_list.insert((i+1)*4-1,val_dict["id"])
                        time_list_per_var,gt_list_per_var,unique_var_id_list = split_plot_all(var_id_list,time_list,gt_list,id_list,rep_split=4,fig_label=f"Train GT {combi_num}-{sample_num}")
                        
                        # get the mean of training samples for each variation, repeat once for the comparator as there is only 1 validation sample
                        train_comparators_per_var = []
                        for samples_per_var in gt_list_per_var:
                            mid,max,min,mean,sem,moe,median,q1,q3,iqr,mad = compute_central_tendency(samples_per_var)
                            
                            comparator = {
                                "samples":samples_per_var,       
                                "mid":mid,
                                "max":max,
                                "min":min,
                                "mean":mean,
                                "sem":sem,
                                "moe":moe,
                                "median":median,
                                "q1":q1,
                                "q3":q3,
                                "iqr":iqr,
                                "mad":mad
                                }
                            train_comparators_per_var.append(comparator)
                            
                        """
                        perform reconstruction for validation task parameters
                        """
                        recon_list = []
                        recon_lut_list = []
                        gt_list = []
                        id_list = []
                        time_list = []
                        start = times.perf_counter()
                        for val_dict,sample in zip(val_list,val_sample_list): 
                            DataIn  = sample.Data[:,0:lastInputIndex+1]
                            DataOut = sample.Data[:,lastInputIndex+1:]
                            
                            gt_list.append(DataOut)
                            id_list.append(val_dict["id"])
                            time_list.append(val_dict["time"])
                    
                            expected_data,exp_sigma,_,_ = tpgmm.repro_condition_gmr(DataIn,sample,lastInputIndex,lastOutputIndex,DS=False,new_dt=0)
                            recon_list.append(expected_data)
                            
                            lut_output = lut.search_closest_output(val_dict["time"],DataIn)
                            recon_lut_list.append(lut_output)
                        
                        # save the results
                        save_results(session_data["patient_id"],session_data["subject_id"],tpgmm,time_list,recon_list,recon_lut_list,gt_list,train_comparators_per_var,id_list,f"{subject_path}/repro/val_{combi_num}_{sample_num}")
                        end = times.perf_counter()
                        total_time += (end - start)
                        print(f"Fast Deploy:Elapsed \t= {(end - start):.4f}s")
                        
                        """plot validation recon & gt with the training data's mean and CI"""
                        if plt_results:
                            stats_fig,stats_ax = plot_multi_dim(
                                x_list=time_list,
                                data_list=gt_list,
                                dim=10,
                                labels=[f"{i}.GT" for i in id_list],
                                xtype="time",
                                datatype="tau",
                                fig_label=f"{session_data['exp_id']} Val Compare {combi_num}-{sample_num}",
                                split=1,
                                legend=False,
                                figsize=(18.33333*0.393701,8.5*0.393701),
                                color_set="bright",
                            )
                            stats_fig,stats_ax = plot_multi_dim(
                                x_list=time_list,
                                data_list=recon_list,
                                dim=10,
                                labels=[f"{i}.TPGMM" for i in id_list],
                                xtype="time",
                                datatype="tau",
                                fig_label=f"{session_data['exp_id']} Val Compare {combi_num}-{sample_num}",
                                split=1,
                                legend=False,
                                prev_fig=stats_fig,prev_ax=stats_ax,
                                color_set="bright",
                                shuffle=5
                            )
                            stats_fig,stats_ax = plot_multi_dim(
                                x_list=time_list,
                                data_list=recon_lut_list,
                                dim=10,
                                labels=[f"{i}.LUT" for i in id_list],
                                xtype="time",
                                datatype="tau",
                                fig_label=f"{session_data['exp_id']} Val Compare {combi_num}-{sample_num}",
                                split=1,
                                legend=False,
                                prev_fig=stats_fig,prev_ax=stats_ax,
                                color_set="bright",
                                shuffle=100
                            )
                            plot_stats(
                                time_list_per_var,
                                gt_list_per_var,
                                fig=stats_fig,
                                axs=stats_ax,
                                labels=[f"{x}.GT" for x in unique_var_id_list],
                                relimit=True,
                                split=1,
                                color_set="bright"
                            )
                            interactive_plot(stats_fig,stats_ax)
                        
                        """
                        perform reconstruction for test task parameters
                        """
                        recon_list = []
                        recon_lut_list = []
                        gt_list = []
                        id_list = []
                        time_list = []
                        var_id_list = []
                        start = times.perf_counter()
                        for test_dict,sample in zip(test_list,test_sample_list): 
                            DataIn  = sample.Data[:,0:lastInputIndex+1]
                            DataOut = sample.Data[:,lastInputIndex+1:]
                            
                            sample_id = test_dict["id"].split(".")
                            gt_list.append(DataOut)
                            id_list.append(test_dict["id"])
                            var_id_list.append(sample_id[0])
                            time_list.append(test_dict["time"])
                            
                            expected_data,exp_sigma,_,_ = tpgmm.repro_condition_gmr(DataIn,sample,lastInputIndex,lastOutputIndex,DS=False,new_dt=0)
                            recon_list.append(expected_data)
                            
                            lut_output = lut.search_closest_output(test_dict["time"],DataIn)
                            recon_lut_list.append(lut_output)
                            
                        time_list_per_var,gt_list_per_var,unique_var_id_list = split_plot_all(var_id_list,time_list,gt_list,id_list,rep_split=4,fig_label=f"Test GT {combi_num}-{sample_num}")
                        
                        # get the mean of each test ground truth, and repeat for 4 repetitions
                        test_comparators_per_var = []
                        for samples_per_var in gt_list_per_var:
                            mid,max,min,mean,sem,moe,median,q1,q3,iqr,mad = compute_central_tendency(samples_per_var)
                            
                            comparator = {
                                "samples":samples_per_var,       
                                "mid":mid,
                                "max":max,
                                "min":min,
                                "mean":mean,
                                "sem":sem,
                                "moe":moe,
                                "median":median,
                                "q1":q1,
                                "q3":q3,
                                "iqr":iqr,
                                "mad":mad
                                }
                            
                            for i in range(session_data["num_rep"]):
                                test_comparators_per_var.append(comparator)
                        # save the results
                        save_results(session_data["patient_id"],session_data["subject_id"],tpgmm,time_list,recon_list,recon_lut_list,gt_list,test_comparators_per_var,id_list,f"{subject_path}/repro/test_{combi_num}_{sample_num}")
                        """ END OF PIPELINE"""
                        end = times.perf_counter()
                        print(f"Fast Deploy:Elapsed \t= {(end - start):.4f}s")
                        total_time += (end - start)
                        
                        """plot test gt's rep, mean and ci within each variation and their corresponding reconstructions"""
                        if plt_results:
                            stats_fig,stats_ax = plot_multi_dim(
                                x_list=time_list,
                                data_list=gt_list,
                                dim=10,
                                labels=[f"{x}.GT" for x in id_list],
                                xtype="time (s)",
                                datatype="tau",
                                fig_label=f"{session_data['exp_id']} Test Compare {combi_num}-{sample_num}",
                                split=4,
                                legend=False,
                                figsize=(18.33333*0.393701,18*0.393701),
                                color_set="bright",
                            )
                            stats_fig,stats_ax = plot_multi_dim(
                                x_list=time_list,
                                data_list=recon_list,
                                dim=10,
                                labels=[f"{x}.TPGMM" for x in id_list],
                                xtype="time",
                                datatype="tau",
                                fig_label=f"{session_data['exp_id']} Test Compare {combi_num}-{sample_num}",
                                split=4,
                                legend=False,
                                prev_fig=stats_fig,
                                prev_ax= stats_ax,
                                color_set="bright",
                                shuffle=200
                                
                            )
                            stats_fig,stats_ax = plot_multi_dim(
                                x_list=time_list,
                                data_list=recon_lut_list,
                                dim=10,
                                labels=[f"{i}.LUT" for i in id_list],
                                xtype="time",
                                datatype="tau",
                                fig_label=f"{session_data['exp_id']} Test Compare {combi_num}-{sample_num}",
                                split=4,
                                legend=False,
                                prev_fig=stats_fig,prev_ax=stats_ax,
                                shuffle=300,
                                color_set="bright",
                            )
                            
                            plot_stats(
                                time_list_per_var,
                                gt_list_per_var,
                                fig=stats_fig,
                                axs=stats_ax,
                                labels=[f"{x}.Test GT" for x in unique_var_id_list],
                                relimit=True,
                                split=1,
                                color_set="bright",
                            )
                            interactive_plot(stats_fig,stats_ax)
                            plt.show()
        print(f"\np{p} Total Train:Elapsed \t= {(total_time):.4f}s")

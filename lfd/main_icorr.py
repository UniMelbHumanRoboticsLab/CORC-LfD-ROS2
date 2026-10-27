import sys,os
import numpy as np
np.set_printoptions(suppress=True,precision=4) # suppress scientific notation
import time as times

from lut_pkg.LookUpTable import LookUpTable
from tpgmm_pkg.tpgmm_util import arrange_data,save_results

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_visual.plot_pkg import plot_multi_dim,plot_stats,interactive_plot,split_plot_all
from data_process.file_util_pkg import compile_train_val_test_data,create_dir
from data_analyse.stats_pkg import compute_central_tendency
import matplotlib.pyplot as plt

plt_results = True
deploy = True

GREEN = '\033[92m'

RED = '\033[91m'
RESET = '\033[0m'

p_start = 1
sub_start  = 11
input_ranges = [range(0,1),range(1,11),range(1,21),range(0,21)]
input_cases = ["t","q","qqd","qqdt"]

def save_results(patient_id,subject_id,time_list,gt_list,recon_dict,recon_type,comparators_per_var,id_list,filename):
    compile_data = []
    # save the results as follows (time,GT,recon,comparator,sample id)
    for i,(time,gt,comparator,sample_id) in enumerate(zip(time_list,gt_list,comparators_per_var,id_list)):
        sample = {
                "patient_id":patient_id,
                "subject_id":subject_id,
                "var-id-case":sample_id,
                "gt":gt,
                "compare":comparator,
                }
        for case_ in input_cases:
            sample[f"recon_lut_{case_}"] = recon_dict[case_][recon_type][i]
        compile_data.append(sample)
    np.save(filename, np.array(compile_data,dtype=dict))
def get_results(filename):
    # check if file exists
    if os.path.isfile(filename):
        exists = True
        save_data = np.load(filename,allow_pickle=True).tolist()
    else:
        exists = False
        save_data = []
    return exists,save_data

for p in range(p_start,4):
    total_time = 0
    if p == 1:
        sm_num = 2
    else:
        sm_num=4
    for sub in range(sub_start,25):
        session_data = {
            "exp_id":"exp1_icorr",
            "patient_id":f"p{p}",
            "subject_id":f"sub{sub}",
            "num_rep":4,
            "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] ,
            "sbmvmt_num":sm_num
        }
        subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
        files = os.listdir(subject_path+"/splits")
        total_combi = len(set(s.split('_')[0] for s in files))
        total_monte = len(set(s.split('_')[-1] for s in files if (len(s.split('_'))==4)))
        for combi_num in range(0,total_combi):
            for sample_num in range(0,total_monte):
                print(f"\n====={session_data['exp_id']}-{session_data['patient_id']}-{session_data['subject_id']}-{combi_num}-{sample_num} =================")
                train_list,val_list,test_list = compile_train_val_test_data(session_data,subject_path,combi_num,sample_num,False)
                all_data = []
                for train in train_list:
                    all_data.append(np.hstack((np.expand_dims(train["time"],axis=1),train["data"])))
                all_data = np.vstack(all_data)
                print(f"Number of samples:{all_data.shape[0]}")
        
                """
                Transform training and test data
                """
                num_of_frames = session_data["sbmvmt_num"]+1
                num_of_dims = 30
                lastInputIndex = 19  # always arrange data such that [input indices, output indices]
                lastOutputIndex = 29
                _,train_sample_list = arrange_data(train_list,num_of_frames,num_of_dims)
                _,val_sample_list = arrange_data(val_list,num_of_frames,num_of_dims)
                _,test_sample_list = arrange_data(test_list,num_of_frames,num_of_dims)
                
                """
                Initialize the LookUp Tables
                """  
                lut_dict = {}
                for case_,range_ in zip(input_cases,input_ranges):
                    lut = LookUpTable(np.concatenate([all_data[:, range_], all_data[:, -10:]], axis=1), len(range_)-1, len(range_)+9)
                    lut_dict[case_] = {"lut":lut,
                                       "recon":[],
                                       "recon_test":[]}
                
                """
                Deployment pipeline
                """
                if deploy:
                    """
                    compile training data for variability visualisation
                    """
                    gt_list = []
                    id_list = []
                    time_list = []
                    var_id_list = []
                    for train_dict,sample in zip(train_list,train_sample_list): 
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
                    perform reconstruction for validation samples
                    """
                    exists,save_data= get_results(f"{subject_path}/repro/val_{combi_num}_{sample_num}.npy")

                    gt_list = []
                    id_list = []
                    time_list = []
                    start = times.perf_counter()
                    for i,(val_dict,sample) in enumerate(zip(val_list,val_sample_list)): 
                        full_sample = np.hstack((np.expand_dims(val_dict["time"],axis=1),val_dict["data"]))
                        DataOut = sample.Data[:,lastInputIndex+1:]
                        
                        gt_list.append(DataOut)
                        id_list.append(val_dict["id"])
                        time_list.append(val_dict["time"])
                        correct = np.linalg.norm(DataOut-save_data[i]['gt']) <= 0 and exists
                        
                        for case_,range_ in zip(input_cases,input_ranges):
                            if correct and f"recon_lut_{case_}" in save_data[i]:
                                lut_dict[case_]["recon"].append(save_data[i][f"recon_lut_{case_}"])
                            else:
                                print(f"redoing {case_} for {subject_path}/repro/val_{combi_num}_{sample_num}_{i}")
                                DataIn = full_sample[:,range_]
                                lut = lut_dict[case_]["lut"]
                                lut_output = lut.search_closest_output(val_dict["time"],DataIn)
                                lut_dict[case_]["recon"].append(lut_output)
                    
                    # save the results
                    if not os.path.isdir(f"{subject_path}/repro"):
                        create_dir(f"{subject_path}/repro")
                    save_results(session_data["patient_id"],session_data["subject_id"],time_list,gt_list,lut_dict,"recon",train_comparators_per_var,id_list,f"{subject_path}/repro/val_{combi_num}_{sample_num}")
                    
                        
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
                            figsize=(18.33333*0.393701,6*0.393701),
                            color_set="bright",
                        )
                        for case_,shuffle in zip(input_cases,[5,100,300,400]):
                            stats_fig,stats_ax = plot_multi_dim(
                                x_list=time_list,
                                data_list=lut_dict[case_]["recon"],
                                dim=10,
                                labels=[f"{i}.{case_}" for i in id_list],
                                xtype="time",
                                datatype="tau",
                                fig_label=f"{session_data['exp_id']} Val Compare {combi_num}-{sample_num}",
                                split=1,
                                legend=False,
                                prev_fig=stats_fig,prev_ax=stats_ax,
                                color_set="bright",
                                shuffle=shuffle
                            )
                        plot_stats(
                            time_list_per_var,
                            gt_list_per_var,
                            fig=stats_fig,
                            axs=stats_ax,
                            labels=[f"{x}.GT" for x in unique_var_id_list],
                            relimit=True,
                            split=1,
                            color_set="bright",
                            legend=False
                        )
                        interactive_plot(stats_fig,stats_ax)

                    """
                    perform reconstruction for test samples
                    """
                    exists,save_data= get_results(f"{subject_path}/repro/test_{combi_num}_{sample_num}.npy")
                    gt_list = []
                    id_list = []
                    time_list = []
                    var_id_list = []
                    start = times.perf_counter()
                    for i,(test_dict,sample) in enumerate(zip(test_list,test_sample_list)): 
                        full_sample = np.hstack((np.expand_dims(test_dict["time"],axis=1),test_dict["data"]))
                        DataOut = sample.Data[:,lastInputIndex+1:]
                        
                        sample_id = test_dict["id"].split(".")
                        gt_list.append(DataOut)
                        id_list.append(test_dict["id"])
                        var_id_list.append(sample_id[0])
                        time_list.append(test_dict["time"])
                        correct = np.linalg.norm(DataOut-save_data[i]['gt']) <= 0 and exists
                        
                        for case_,range_ in zip(input_cases,input_ranges):
                            if correct and f"recon_lut_{case_}" in save_data[i]:
                                lut_dict[case_]["recon_test"].append(save_data[i][f"recon_lut_{case_}"])
                            else:
                                print(f"redoing {case_} for {subject_path}/repro/test_{combi_num}_{sample_num}_{i}")
                                DataIn = full_sample[:,range_]
                                lut = lut_dict[case_]["lut"]
                                lut_output = lut.search_closest_output(test_dict["time"],DataIn)
                                lut_dict[case_]["recon_test"].append(lut_output)
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
                    save_results(session_data["patient_id"],session_data["subject_id"],time_list,gt_list,lut_dict,"recon_test",test_comparators_per_var,id_list,f"{subject_path}/repro/test_{combi_num}_{sample_num}")
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
                            figsize=(18.33333*0.393701,12*0.393701),
                            color_set="bright",
                        )
                        for case_,shuffle in zip(input_cases,[5,100,300,400]):
                            stats_fig,stats_ax = plot_multi_dim(
                                x_list=time_list,
                                data_list=lut_dict[case_]["recon_test"],
                                dim=10,
                                labels=[f"{i}.{case_}" for i in id_list],
                                xtype="time",
                                datatype="tau",
                                fig_label=f"{session_data['exp_id']} Test Compare {combi_num}-{sample_num}",
                                split=4,
                                legend=False,
                                prev_fig=stats_fig,prev_ax=stats_ax,
                                color_set="bright",
                                shuffle=shuffle
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
                            legend=False
                        )
                        interactive_plot(stats_fig,stats_ax)
                        plt.show()
    # print(f"\np{p} Total Train:Elapsed \t= {(total_time):.4f}s")

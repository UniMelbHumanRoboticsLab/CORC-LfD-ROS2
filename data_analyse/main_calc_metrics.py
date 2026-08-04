import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from metrics_pkg import compute_norm_error,compute_norm_tau_peak_diff,q,compute_coverage
from data_process.file_util_pkg import load_npy,save_npy
import numpy as np
from stats_pkg import remove_outliers_iqr
def print_string(str_list):
    string = ""
    for i in str_list:
        string+=i
        string+=" "
    print(string)
def sanity_check(sample):
    # sanity check purposes   
    print_string(["-----",sample["patient_id"],sample["subject_id"],sample["var-id-case"],"-----"])
    # print()
    print_string([f"{'metric':10}",
                   f"{'joint':15}",
                   f"{'diff':10}"])
    for cur_case in ["gt","recon","recon_lut"]:
        for metric in ["avg_norm_error","norm_diff_tau","coverage"]:
            for i,joint in enumerate(q):
                case_metric = f"{cur_case}_{metric}"
                print_string([f"{case_metric:10}",
                               f"{joint:15}",
                               f"{sample[case_metric][i]:10.5f}"])
        print()
    print()

p_start = 1
sub_start = 11
total_monte = 4

remove_var = 2

for train_var_num,total_combi in zip([2,3],[2,10]):
    for p in range(p_start,4):
        total_time = 0
            
        for sub in range(sub_start,25):
            session_data = {
                "exp_id":f"exp1_trained_{train_var_num}",
                "patient_id":f"p{p}",
                "subject_id":f"sub{sub}",
                "num_rep":4,
                "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
            }
            subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
            
            val_samples_compile = []
            test_samples_compile = []
            for combi_num in range(0,total_combi):
                for sample_num in range(0,total_monte):
                    
                    import pandas as pd
                    train_test_split = pd.read_csv(f'{subject_path}/splits/{combi_num}_train_test.csv')[["split"]].values
                    train_val_df = pd.read_csv(f'{subject_path}/splits/{combi_num}_train_val_{sample_num}.csv')
                    train_val_split = dict(zip(train_val_df["repetition"], train_val_df["split"]))
                    
                    if p == 3 and sub == 20:
                        if train_test_split[1] == "train":
                            print(f"{session_data['exp_id']}_{session_data['patient_id']}_{session_data['subject_id']}_{combi_num}_{sample_num} discarded")
                            break
                        else:
                            print(f"{session_data['exp_id']}_{session_data['patient_id']}_{session_data['subject_id']}_{combi_num}_{sample_num}")
                    else:
                        print(f"{session_data['exp_id']}_{session_data['patient_id']}_{session_data['subject_id']}_{combi_num}_{sample_num}")
                        
                    # calculate metrics for validation set
                    val_samples = load_npy(f"{subject_path}/repro/val_{combi_num}_{sample_num}.npy")
                    for sample in val_samples:
                        for i in ["gt","recon","recon_lut"]:
                            sample[i+"_avg_norm_error"] = compute_norm_error(sample[i],sample["compare"])
                            sample[i+"_norm_diff_tau"] = compute_norm_tau_peak_diff(sample[i],sample["compare"])
                            sample[i+"_coverage"] = compute_coverage(sample[i],sample["compare"])
                            sample[i+"_avg_norm_error_mean"] =  np.array([np.mean(remove_outliers_iqr(sample[i+"_avg_norm_error"])[0])])
                            sample[i+"_norm_diff_tau_mean"] =   np.array([np.mean(remove_outliers_iqr(sample[i+"_norm_diff_tau"])[0])])
                            sample[i+"_coverage_mean"] =        np.array([np.mean(remove_outliers_iqr(sample[i+"_coverage"])[0])])
                        # sanity_check(sample)
                    val_samples_compile += val_samples
                    
                    # calculate metrics for test set
                    test_samples = load_npy(f"{subject_path}/repro/test_{combi_num}_{sample_num}.npy")
                    for sample in test_samples:
                        
                        for i in ["gt","recon","recon_lut"]:
                            sample[i+"_avg_norm_error"] = compute_norm_error(sample[i],sample["compare"])
                            sample[i+"_norm_diff_tau"] = compute_norm_tau_peak_diff(sample[i],sample["compare"])
                            sample[i+"_coverage"] = compute_coverage(sample[i],sample["compare"])
                            sample[i+"_avg_norm_error_mean"] =  np.array([np.mean(remove_outliers_iqr(sample[i+"_avg_norm_error"])[0])])
                            sample[i+"_norm_diff_tau_mean"] =   np.array([np.mean(remove_outliers_iqr(sample[i+"_norm_diff_tau"])[0])])
                            sample[i+"_coverage_mean"] =        np.array([np.mean(remove_outliers_iqr(sample[i+"_coverage"])[0])])
                        # sanity_check(sample)
                        
                        if p == 3 and sub == 20:
                            if "var_2" in sample["var-id-case"]:
                                print(f"{sample['patient_id']}_{sample['subject_id']}_{combi_num}_{sample_num}_{sample['var-id-case']} discarded")
                                continue
                            else:
                                test_samples_compile.append(sample)
                                print(f"{sample['patient_id']}_{sample['subject_id']}_{combi_num}_{sample_num}_{sample['var-id-case']}")
                        else:
                            test_samples_compile.append(sample)
                        #     
                    #     print(f"{sample['patient_id']}_{sample['subject_id']}_{combi_num}_{sample_num}_{sample['var-id-case']}")
                    # test_samples_compile += test_samples
            
            save_npy(f"{subject_path}/repro/val_processed",val_samples_compile)
            save_npy(f"{subject_path}/repro/test_processed",test_samples_compile)

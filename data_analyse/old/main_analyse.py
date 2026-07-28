# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_process.file_util_pkg import load_npy
from data_visual.plot_pkg import plot_violins
from stats_pkg import remove_outliers_iqr
from metrics_pkg import q
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_subject = True
plot_per_patient = False

avg_norm_labels = []
tau_diff_labels = []
coverage_labels = []
for i in ["Seen","Unseen"]:
    for j in ["TPGMM","LUT","THERAPIST"]:
        avg_norm_labels.append(j)
        tau_diff_labels.append(j)
        if j != "THERAPIST":
            coverage_labels.append(j)
         
methods = ["recon","recon_lut","gt",]
subjects = [
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
if all_subject:
    """
    Every subject
    """
    all_patients_error = []
    all_patients_tau_diff = []
    all_patients_coverage = []
    all_patients_gaussian = []
    for p in range(1,4):
        if p == 1:
            sm_num = 2
        else:
            sm_num=4
        
        all_val_avg_norm_error,all_val_norm_diff_tau,all_val_coverage,all_test_avg_norm_error,all_test_norm_diff_tau,all_test_coverage = {},{},{},{},{},{}
        for i in methods:
            all_val_avg_norm_error[i] = []
            all_val_norm_diff_tau[i] = []
            all_val_coverage[i] = []
            all_test_avg_norm_error[i] = []
            all_test_norm_diff_tau[i] = []
            all_test_coverage[i] = []
        
        num_of_gaussians = []
        for sub in subjects[p-1]:
            session_data = {
                "exp_id":"exp1_trained2",
                "patient_id":f"p{p}",
                "subject_id":f"sub{sub}",
                "sbmvmt_num":sm_num,
                "num_rep":4,
                "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
            }
            subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
            
            val_samples = load_npy(f"{subject_path}/repro/val_processed.npy")
            val_samples_df = pd.DataFrame(val_samples)
      
            test_samples = load_npy(f"{subject_path}/repro/test_processed.npy")
            test_samples_df = pd.DataFrame(test_samples)

            if len(val_samples) > 0:
                for i in methods:
                    # all combinations
                    val_avg_norm_error = np.array(val_samples_df[f'{i}_avg_norm_error_mean'].tolist())
                    val_norm_diff_tau = np.array(val_samples_df[f'{i}_norm_diff_tau_mean'].tolist())
                    # average over combination
                    all_val_avg_norm_error[i].append(np.mean(remove_outliers_iqr(val_avg_norm_error)[0]))
                    all_val_norm_diff_tau[i].append(np.mean(remove_outliers_iqr(val_norm_diff_tau)[0]))
                    if i != "gt":
                        val_coverage = np.array(val_samples_df[f'{i}_coverage_mean'].tolist())
                        all_val_coverage[i].append(np.mean(remove_outliers_iqr(val_coverage)[0]))
                num_of_gaussians.append(val_samples_df['tpgmm_param'].values[0::4].tolist())
            if len(test_samples) > 0:
                for i in methods:
                    # all combinations
                    test_avg_norm_error = np.array(test_samples_df[f'{i}_avg_norm_error_mean'].tolist())
                    test_norm_diff_tau = np.array(test_samples_df[f'{i}_norm_diff_tau_mean'].tolist())
                    # average over combination
                    all_test_avg_norm_error[i].append(np.mean(remove_outliers_iqr(test_avg_norm_error)[0]))
                    all_test_norm_diff_tau[i].append(np.mean(remove_outliers_iqr(test_norm_diff_tau)[0]))
                    
                    if i != "gt":
                        test_coverage = np.array(test_samples_df[f'{i}_coverage_mean'].tolist())
                        all_test_coverage[i].append(np.mean(remove_outliers_iqr(test_coverage)[0]))

        all_avg_norm_error_per_patient = []
        all_norm_diff_tau_per_patient = []
        all_coverage_per_patient = []
        for i in methods:
            all_val_avg_norm_error[i] = np.vstack(all_val_avg_norm_error[i])
            all_avg_norm_error_per_patient.append(all_val_avg_norm_error[i])
            all_val_norm_diff_tau[i] = np.vstack(all_val_norm_diff_tau[i])
            all_norm_diff_tau_per_patient.append(all_val_norm_diff_tau[i])
            if i != "gt":
                all_val_coverage[i] = np.vstack(all_val_coverage[i])
                all_coverage_per_patient.append(all_val_coverage[i])

        for i in methods:
            all_test_avg_norm_error[i] = np.vstack(all_test_avg_norm_error[i])
            all_avg_norm_error_per_patient.append(all_test_avg_norm_error[i])
            all_test_norm_diff_tau[i] = np.vstack(all_test_norm_diff_tau[i])
            all_norm_diff_tau_per_patient.append(all_test_norm_diff_tau[i])
        
            if i != "gt":
                all_test_coverage[i] = np.vstack(all_test_coverage[i])
                all_coverage_per_patient.append(all_test_coverage[i])
            
        all_patients_error.append(all_avg_norm_error_per_patient)
        all_patients_tau_diff.append(all_norm_diff_tau_per_patient)
        all_patients_coverage.append(all_coverage_per_patient)
        num_of_gaussians = np.hstack(num_of_gaussians)
        all_patients_gaussian.append([num_of_gaussians])
        
    # lump all personas into 1
    overall_patients_error = np.hstack(all_patients_error)
    overall_patients_error = [arr for arr in overall_patients_error]
    all_patients_error.insert(0,overall_patients_error)
    
    overall_patients_tau_diff = np.hstack(all_patients_tau_diff)
    overall_patients_tau_diff = [arr for arr in overall_patients_tau_diff]
    all_patients_tau_diff.insert(0,overall_patients_tau_diff)
    
    overall_patients_coverage = np.hstack(all_patients_coverage)
    overall_patients_coverage = [arr for arr in overall_patients_coverage]
    all_patients_coverage.insert(0,overall_patients_coverage)
    
    # plot 
    plot_violins(
        title="Normalized Average Error "+r"$\epsilon$",
        data_list=all_patients_error,
        axis_num = 4,
        x_labels=avg_norm_labels,
        axis_title=["All Personas","Persona 1","Persona 2","Persona 3"],
        split=2,
        figwidth=7,
        remove_outlier=False)
    plot_violins(
        title="Normalized "+r"$\Delta\tau_{peak}$",
        data_list=all_patients_tau_diff,
        axis_num = 4,
        x_labels=tau_diff_labels,
        axis_title=["All Personas","Persona 1","Persona 2","Persona 3"],
        split=2,figwidth=6.5,
        remove_outlier=False)
    plot_violins(
        title="Coverage "+r"$C~(\%)$",
        data_list=all_patients_coverage,
        axis_num = 4,
        x_labels=coverage_labels,
        axis_title=["All Personas","Persona 1","Persona 2","Persona 3"],
        split=2,figwidth=4.83,
        remove_outlier=False)
    plt.show()
else:
    """
    Single subject
    """
    session_data = {
        "exp_id":"exp1_trained2",
        "patient_id":"p2",
        "subject_id":"sub11",
        "sbmvmt_num":2,
        "num_rep":4,
        "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
    }
    subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
    
    val_samples = load_npy(f"{subject_path}/repro/val_processed.npy")
    val_samples_df = pd.DataFrame(val_samples)
    avg_norm_error = np.array(val_samples_df['avg_norm_error'].values.tolist())
    norm_diff_tau = np.array(val_samples_df['norm_diff_tau'].values.tolist())
    coverage = np.array(val_samples_df['coverage'].values.tolist())
    
    test_samples = load_npy(f"{subject_path}/repro/test_processed.npy")
    test_samples_df = pd.DataFrame(test_samples)
    avg_norm_error_test = np.array(test_samples_df['avg_norm_error'].values.tolist())
    check = np.mean(avg_norm_error_test,axis=0)
    norm_diff_tau_test = np.array(test_samples_df['norm_diff_tau'].values.tolist())
    coverage_test = np.array(test_samples_df['coverage'].values.tolist())
    
    plot_violins(
        title=r"Normalized Metric Dist.",
        data_list=[[avg_norm_error ,avg_norm_error_test ,norm_diff_tau ,norm_diff_tau_test ],
                   [np.mean(avg_norm_error,axis=1),np.mean(avg_norm_error_test,axis=1),np.mean(norm_diff_tau,axis=1),np.mean(norm_diff_tau_test,axis=1)]],
        axis_num = 2,
        x_labels=[r"Average Norm $\epsilon$ \n(Seen)",r"Average Norm $\epsilon$ \n(Unseen)",r"Norm $\Delta\tau_{peak}$ \n(Seen)",r"Norm $\Delta\tau_{peak}$ (Unseen)"],
        axis_title=["All Joints","Mean Joints"])
    
    plot_violins(
        title="Coverage Dist",
        data_list=[[coverage ,coverage_test ],
                   [np.mean(coverage,axis=1),np.mean(coverage_test,axis=1)]],
        axis_num = 2,
        x_labels=[r"Coverage % \n(Seen)",r"Coverage % \n(Unseen)"],
        axis_title=["All Joints","Mean Joints"])
    
    data_list_per_joint = []
    coverage_per_joint = []
    axis_title_list = []
    for i,joint in enumerate(q):
        data_list_per_joint.append([avg_norm_error[:,i],avg_norm_error_test[:,i],norm_diff_tau[:,i],norm_diff_tau_test[:,i]])
        coverage_per_joint.append([coverage[:,i],coverage_test[:,i]])
        axis_title_list.append(f"{joint}")
        
    plot_violins(
        data_list=data_list_per_joint,
        title="Joint Normalized Metric Dist.",
        axis_num = len(q),
        x_labels=[r"Average Norm $\epsilon$ (Seen)",r"Average Norm $\epsilon$ (Unseen)",r"Norm $\Delta\tau_{peak}$ (Seen)",r"Norm $\Delta\tau_{peak}$ (Unseen)"],
        axis_title=axis_title_list)
    plot_violins(
        data_list=coverage_per_joint,
        title="Joint Coverage Dist.",
        axis_num = len(q),
        x_labels=[r"Coverage % (Seen)",r"Coverage % (Unseen)"],
        axis_title=axis_title_list)
    
    import matplotlib.pyplot as plt
    plt.show()

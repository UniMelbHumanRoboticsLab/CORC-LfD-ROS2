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

test_ids = ["val","test"]
test_cases = ["Seen","Unseen"]
method_ids = ["recon","recon_lut","gt",]
method_names = ["TPGMM","LUT","THERAPIST"]
subjects = [
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24]]

secondary_stats = pd.read_csv('figures/secondary_stats.csv')
for metric_id,metric_name,figwidth in zip(["avg_norm_error_mean","norm_diff_tau_mean","coverage_mean"],[r"$\epsilon$",r"$\Delta\tau_{peak}$", r"$C~(\%)$" ],[7,6.5,4.83]):    
    metric_labels = []
    for i in test_cases:
        for j in method_names:
            if metric_id != "coverage_mean" or j != "THERAPIST":
                metric_labels.append(j)

    cur_metric_all_patients = []
    if metric_id == "avg_norm_error_mean":
        all_patients_gaussian = []
    """
    All Persona
    """
    for p in range(1,4):
        cur_metric_per_patient = []
        num_of_gaussians_per_patient = []
        for test_id,test_case in zip(test_ids,test_cases):
            cur_metric_per_patient_case = {}
            for i in method_ids:
                cur_metric_per_patient_case[i] = []
                
            """
            All Subjects
            """
            for sub in subjects[p-1]:
                session_data = {
                    "exp_id":"exp1_tnsre_trained4",
                    "patient_id":f"p{p}",
                    "subject_id":f"sub{sub}",
                    "num_rep":4,
                    "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
                }
                subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
                
                samples = load_npy(f"{subject_path}/repro/{test_id}_processed.npy")
                samples_df = pd.DataFrame(samples)
        
                if len(samples) > 0:
                    for i in method_ids:
                        if metric_id != "coverage_mean" or i != "gt":
                            # all train-val-generalisation combination per patient per subject
                            metric_per_patient_case_method = np.array(samples_df[f'{i}_{metric_id}'].tolist())
                            # average over train-val-generalisation combination per patient per subject
                            cur_metric_per_patient_case[i].append(np.mean(remove_outliers_iqr(metric_per_patient_case_method)[0]))
                    if test_id == "val":
                        num_of_gaussians_per_patient.append(samples_df['tpgmm_param'].values[0::4].tolist())
            
            for i in method_ids:
                if metric_id != "coverage_mean" or i != "gt":
                    # stack each participant 
                    cur_metric_per_patient_case[i] = np.vstack(cur_metric_per_patient_case[i])
                    cur_metric_per_patient.append(cur_metric_per_patient_case[i])
            
        cur_metric_all_patients.append(cur_metric_per_patient)
        if "val" in test_ids:
            num_of_gaussians_per_patient = np.hstack(num_of_gaussians_per_patient)
            all_patients_gaussian.append([num_of_gaussians_per_patient])
        
    # get lump sum of all personas (this lump sum should match the lump sum in main_stats)
    cur_metric_overall_patient = np.hstack(cur_metric_all_patients)
    cur_metric_overall_patient = [arr for arr in cur_metric_overall_patient]
    cur_metric_all_patients.insert(0,cur_metric_overall_patient)

    # get required stats
    required_stats = secondary_stats[[f"Seen {metric_name}",f"Unseen {metric_name}"]].values.tolist()
    # plot 
    plot_violins(
        title=metric_name,
        data_list=cur_metric_all_patients,
        axis_num = 4,
        x_labels=metric_labels,
        axis_title=["All Personas","Persona 1","Persona 2","Persona 3"],
        split=len(test_cases),
        figwidth=figwidth,
        remove_outlier=False,
        significance_set=required_stats)
    if metric_id == "avg_norm_error_mean" and "val" in test_ids:
        plot_violins(
            title="Gaussian Numbers",
            data_list=all_patients_gaussian,
            axis_num = 3,
            x_labels=["gaussian #"],
            axis_title=["Persona 1","Persona 2","Persona 3"],
            split=1,
            figwidth=figwidth,
            remove_outlier=False)
plt.show()

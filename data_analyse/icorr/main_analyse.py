# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_process.file_util_pkg import load_npy
from data_visual.plot_pkg import plot_violins
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from stats_pkg import remove_outliers_iqr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_ids = ["val","test"]
test_cases = ["Seen","Unseen"]
method_ids = ["recon_lut_q","recon_lut_qqd","recon_lut_qqdt","recon_lut_t","gt"]
method_names = [
                {"name":r"$LUT (q)$","id":"Q"},
                {"name":r"$LUT (q,\dot{q})$","id":"QQD"},
                {"name":r"$LUT (q,\dot{q},t)$","id":"QQDT"},
                {"name":r"$LUT (t)$","id":"T"},
                {"name":"THERAPIST"}]
subjects = [
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
# subjects = [
#     [11,12],
#     [11,12],
#     [11,12]]
width = 18.33
secondary_stats = pd.read_csv('figures/secondary_stats.csv')
for metric_id,metric_name,figwidth in zip(["avg_norm_error_mean","norm_diff_tau_mean","coverage_mean"],[r"$\epsilon$",r"$\Delta\tau_{peak}$", r"$C~(\%)$" ],[0.5*width,0.5*width,0.5*width]):    
    method_labels = []
    for i in test_cases:
        for j in method_names:
            if metric_id != "coverage_mean" or j["name"] != "THERAPIST":
                method_labels.append(j)

    cur_metric_all_patients = []
    """
    All Persona
    """
    for p in [1,2,3]:
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
                    "exp_id":"exp1_icorr",
                    "patient_id":f"p{p}",
                    "subject_id":f"sub{sub}",
                    "num_rep":4,
                    "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
                }
                subject_path = os.path.join(os.path.dirname(__file__), '../..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
                
                samples = load_npy(f"{subject_path}/repro/{test_id}_processed.npy")
                samples_df = pd.DataFrame(samples)
        
                if len(samples) > 0:
                    for i in method_ids:
                        if metric_id != "coverage_mean" or i != "gt": # dont collect for gt's coverage
                            # all train-val-generalisation combination per patient per subject
                            metric_per_patient_case_method = np.array(samples_df[f'{i}_{metric_id}'].tolist())
                            # average over train-val-generalisation combination per patient per subject
                            cur_metric_per_patient_case[i].append(np.mean(remove_outliers_iqr(metric_per_patient_case_method)[0]))
            
            for i in method_ids:
                if metric_id != "coverage_mean" or i != "gt": # dont collect for gt's coverage
                    # stack each participant 
                    cur_metric_per_patient_case[i] = np.vstack(cur_metric_per_patient_case[i])
                    cur_metric_per_patient.append(cur_metric_per_patient_case[i])
        cur_metric_all_patients.append(cur_metric_per_patient)
        
    # get lump sum of all personas (this lump sum should match the lump sum in main_stats)
    cur_metric_overall_patient = np.hstack(cur_metric_all_patients)
    cur_metric_overall_patient = [arr for arr in cur_metric_overall_patient]
    cur_metric_all_patients.insert(0,cur_metric_overall_patient)
    # cur_metric_all_patients = [cur_metric_overall_patient]

    # get required stats
    required_stats = secondary_stats[["id",f"Seen {metric_name}",f"Unseen {metric_name}"]].values.tolist()
    # plot 
    plot_violins(
        title=metric_name,
        data_list=cur_metric_all_patients,
        axis_num = 4,
        x_labels=method_labels,
        ax_titles=["All Personas","Persona 1","Persona 2","Persona 3"],
        split=len(test_cases),
        figwidth=figwidth,
        remove_outlier=False,
        significance_list=required_stats)

plt.show()

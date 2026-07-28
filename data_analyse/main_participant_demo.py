import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__)))
from stats_pkg import compute_central_tendency
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyCORC.pycorc_io.package_utils.unpack_json import get_subject_file
from data_process.file_util_pkg import get_raw_data
from lfd.tpgmm_pkg.TPGMM import TPGMM
sys.modules['tpgmm_pkg'] = sys.modules['lfd.tpgmm_pkg'] # because it was saved in another file directory - pickle is dumb

subject_stats_csv = "./figures/subject_stats.csv"
if os.path.exists(subject_stats_csv):
    subjects_df = pd.read_csv(subject_stats_csv)
else:
    subjects_dict_list = []
    for sub in range(1,25):
        session_data = {
            "exp_id":"exp1_tnsre_trained4",
            "subject_id":f"sub{sub}",
            "num_rep":4,
            "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
        }
        body_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/subject_measurements/{session_data["subject_id"]}')
        subject_param = get_subject_file(body_path)
        subject_param["sub_id"] = session_data["subject_id"]
        
        for p in range(1,4):
            subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/p{p}/{session_data["subject_id"]}')
            demo_time_per_patient = []
            # grab the times for each demonstration
            for var in session_data["variants"]:
                for rep in range(1,session_data["num_rep"]+1):
                    data_path = f'{subject_path}/{var}/raw/UBORecord{rep}Log.csv'
                    print("processing ",data_path)
                    _,time_data_unscaled,_,_ = get_raw_data(data_path)
                    demo_time_per_patient.append(time_data_unscaled[-1])
            # grab the times for each training
            training_time_per_patient = []
            for combi_num in range(0,6):
                for sample_num in range(0,4):
                    print(f"\n===== p{p}-{session_data['subject_id']}-{combi_num}-{sample_num} =================")
                    tpgmm_file_path = f'{subject_path}/repro/tpgmm_{combi_num}_{sample_num}.pkl'
                    exist = os.path.exists(tpgmm_file_path)
                    if exist: 
                        with open(tpgmm_file_path, 'rb') as outp:                        
                            tpgmm = pickle.load(outp)
                            assert tpgmm.__class__.__name__ == 'TPGMM'
                            num_of_gauss = tpgmm.num_of_gauss
                            training_time_per_patient.append(tpgmm.training_time+tpgmm.bic_time)
                    else:
                        training_time_per_patient.append(0)
                            
            demo_time_per_patient = np.array(demo_time_per_patient)
            training_time_per_patient = np.array(training_time_per_patient)
            subject_param[f"p{p}_demo"] = np.sum(demo_time_per_patient)
            subject_param[f"p{p}_train"] = np.mean(training_time_per_patient)
      
                
        required_keys = ['sub_id','age', 'gender','body_height']
        for p in range(1,4):
            required_keys.append(f"p{p}_demo")
            required_keys.append(f"p{p}_train")
            required_keys.append(f"p{p}_interaction")
        reduced_subject_param = {k: subject_param[k] for k in required_keys}
        subjects_dict_list.append(reduced_subject_param)
    subjects_df = pd.DataFrame(subjects_dict_list)
    subjects_df.to_csv(subject_stats_csv)

subjects_df[["p1_demo","p2_demo","p3_demo","p1_train","p2_train","p3_train"]]=subjects_df[["p1_demo","p2_demo","p3_demo","p1_train","p2_train","p3_train"]]/60
subjects_df[["p1_demo","p2_demo","p3_demo"]]=subjects_df[["p1_demo","p2_demo","p3_demo"]]/2
# Calculate mean and SEM for all numeric columns by gender
def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

# aggregrate information
result = subjects_df[['age','body_height',"p1_demo","p2_demo","p3_demo"]].agg(['mean','sem'])
print(result)
print()
# training informatin
oi = subjects_df[["p1_train","p2_train","p3_train"]][10:]
result = subjects_df[["p1_train","p2_train","p3_train"]][10:].agg(['mean','sem'])
print(result)
# gender aggregrate
percentages = subjects_df['gender'].value_counts()
print(percentages)


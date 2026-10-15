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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_visual.plot_pkg import plot_multi_source_spread,interactive_spread

for p in range(1,4):
    if p == 1:
        sm_num = 2
    else:
        sm_num=4
        
    all_data_dict = {}
    for sub in range(11,25):
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
        
        """ init xsens skeleton model """
        q = ['trunk_ie','trunk_aa','trunk_fe',
                'clav_dep_ev','clav_prot_ret',
                'shoulder_fe','shoulder_aa','shoulder_ie',
                'elbow_fe','elbow_ps']
        qdot = [f"{joint}_dot" for joint in q]
        tau = [f"tau_{joint}" for joint in q]
        
        for var in session_data["variants"]:
            for rep in range(1,5):
                exist = os.path.exists(f'{subject_path}/{var}/processed/UBORecord{rep}Log.csv')

                # check if data processed?
                if exist:
                    print(f'{subject_path}/{var}/processed/UBORecord{rep}Log.csv Processed')
                    full_df = pd.read_csv(f'{subject_path}/{var}/processed/UBORecord{rep}Log.csv')
                    time_data_norm = full_df["norm_time"].values
                    q_traj_rad = full_df[q].values
                    qdot_traj_rad = full_df[qdot].values
                    hand_pos_traj = full_df[["x","y","z"]].values
                    tau_total = full_df[tau].values
                    indices_traj = full_df['index'].values[:,np.newaxis]

                all_data_dict[f'{var}.Rep{rep}.{session_data["patient_id"]}.{session_data["subject_id"]}'] ={
                    "time_data":time_data_norm,
                    "q_traj":q_traj_rad,
                    "qdot_traj":qdot_traj_rad,
                    "tau_traj":tau_total,
                    }

    # arrange the compiled trajectories for compare_multi_dim
    time_list_all,tau_list_all= [],[]
    for sub in range(11,25):
        time_list_sub,tau_list_sub,label_list_sub = [],[],[]
        for i,var in enumerate(["var_1","var_2","var_3","var_4","var_5","var_6"]):
            time_list_var,tau_list_var = [],[]
            for rep in range(1,5):
                label = f'{var}.Rep{rep}.{f"p{p}"}.{f"sub{sub}"}'  
                time_list_var.append(all_data_dict[label]["time_data"][:,np.newaxis])
                tau_list_var.append(all_data_dict[label]["tau_traj"])
            time_list_sub.append(time_list_var)
            tau_list_sub.append(tau_list_var)

        time_list_all.append(time_list_sub)
        tau_list_all.append(tau_list_sub)

    if p==1:
        figwidth=7
    else:
        figwidth=(18.333333-7)/2
    fig,axs = plot_multi_source_spread(x_list=time_list_all,data_list=tau_list_all,dim=6*14,labels=tau,xtype="norm. time",figwidth=figwidth,datatype="tau",split=4,sharex=False,legend=True,fig_label=f"p{p}")
    interactive_spread(fig,axs)
    fig.savefig(f'figures/p{p}.svg')

plt.show()

    # assert 0 

                
                    
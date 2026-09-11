import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import numpy as np
realmin  = np.finfo(np.float64).tiny
from tpgmm_objects import Sample,TaskParams
from matplotlib.patches import Polygon
import pickle

def get_optim_nbGauss(data):
    from sklearn.mixture import GaussianMixture as GMM_sk
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV
    import seaborn as sns
    import pandas as pd

    from sklearn.preprocessing import MinMaxScaler

    # Standardize for proper model selection
    step = 1
    q_data_unscaled,qdot_data_unscaled,tau_data_unscaled = data[::step,0:10],data[::step,10:20],data[::step,20:-1]

    # Normalize q_data into unit vectors for each row
    q_data = q_data_unscaled / (np.linalg.norm(q_data_unscaled, axis=1, keepdims=True)+realmin)
    qdot_data = qdot_data_unscaled / (np.linalg.norm(qdot_data_unscaled, axis=1, keepdims=True)+realmin)
    tau_data = tau_data_unscaled / np.linalg.norm(tau_data_unscaled, axis=1, keepdims=True)
    data_scaled = np.hstack((q_data,qdot_data,tau_data))
    # data_scaled = np.hstack((q_data_unscaled,qdot_data_unscaled,tau_data_unscaled))

    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    optim_n_gauss = 1000
    n_components = 5
    current_best_score = 1e-20
    param_grid = {
        "n_components": range(1, n_components+1),
        "covariance_type": ["full"],
    }
    
    all_results_df = pd.DataFrame()
    idx = np.arange(len(data_scaled))
    one_split = [(idx, idx)]
    while n_components <= 100:
        grid_search = GridSearchCV(
            GMM_sk(init_params='k-means++',max_iter=200), 
            param_grid=param_grid, scoring=gmm_bic_score,
            verbose=3,n_jobs=-1,
            cv = KFold(n_splits=5, shuffle=True)
            )
        grid_search.fit(data_scaled)

        optim_n_gauss = grid_search.best_params_['n_components']
        optim_mean_score = grid_search.best_score_
        grid_search_df = pd.DataFrame(grid_search.cv_results_)[
            ["param_n_components", "param_covariance_type", "mean_test_score"]
        ]
        all_results_df = pd.concat([all_results_df, grid_search_df], ignore_index=True)
        bic_change = np.abs((optim_mean_score-current_best_score)/current_best_score*100)
        
        # either optimal number of gaussian is far from the max limit or the score has converged
        if optim_n_gauss < n_components and bic_change < 5:
            print(bic_change)
            print(f"\nPrevious best:{current_best_score:.4f}, Current best: {optim_mean_score:.4f}, BIC Change: {bic_change :.4f}")
            print(f"Selected GMM (scaled): {grid_search.best_params_['covariance_type']} model, {optim_n_gauss} components")
            break
        else:
            n_components += 5
            print(f"\n{optim_n_gauss} is too close to the limit.")
            print(f"Previous best:{current_best_score:.4f}, Current best: {optim_mean_score:.4f}, BIC Change: {bic_change :.4f}")
            if optim_n_gauss-3<=0:
                optim_n_gauss = 1
            else:
                optim_n_gauss -=3
            print(f"Optimizing from {optim_n_gauss} to {n_components}\n")
            current_best_score = optim_mean_score

            param_grid = {
                "n_components": range(optim_n_gauss, n_components+1),
                "covariance_type": ["full"],
            }

    all_results_df["mean_test_score"] = -all_results_df["mean_test_score"]
    all_results_df = all_results_df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )
    all_results_df.sort_values(by="BIC score").head()
    # sns.catplot(
    #     data=all_results_df,
    #     kind="bar",
    #     x="Number of components",
    #     y="BIC score",
    #     hue="Type of covariance",
    # )
    
    optim_n_gauss = grid_search.best_params_['n_components']
    return optim_n_gauss,all_results_df

"""
Function to transform / arrange data into samples / extract task parameters from each sample 
"""
def unpack_tp(tp,num_of_frames,num_of_dim):
    # extract every frame transformation for current sample
    A = np.array([tp[i,:(num_of_dim*num_of_dim)].reshape((num_of_dim,num_of_dim)).transpose() for i in range(num_of_frames)])
    invA = np.array([np.linalg.inv(A[i]) for i in range(num_of_frames)])
    b = np.array([tp[i,(num_of_dim*num_of_dim):].reshape((num_of_dim,1)) for i in range(num_of_frames)])
    paramsObj = TaskParams(A=A,b=b,invA=invA)
    return paramsObj
    
def transform_data(data,tp,num_of_frames,num_of_dim):
    paramsObj = unpack_tp(tp,num_of_frames,num_of_dim)
    
    # arrange the original trajectory referencing inertial frame
    if data is not None:
        num_of_data_points = data.shape[0]
        sampleObj = Sample(params=paramsObj,Data=data,nbData=num_of_data_points)
    
        tp_traj = np.empty((num_of_data_points,num_of_dim,num_of_frames))
        # task parameterize the current trajectory according to each frame
        for i in range(num_of_frames):
            displaced = np.expand_dims(data-paramsObj.b[i].T,axis=2)
            rotate = np.squeeze(np.matmul(paramsObj.invA[i],displaced),axis=2)
            tp_traj[:,:,i] = rotate
    else:
        sampleObj = Sample(params=paramsObj,Data=None,nbData=None)
        tp_traj = None
    return tp_traj,sampleObj
def arrange_data(data_dict_list,num_of_frames,num_of_dim):
    tp_data_list = []
    sample_list = []
    for data_dict in data_dict_list:
        if "data" in data_dict:
            data = data_dict["data"]
        else:
            data = None
        cur_tp = data_dict["tp"]

        tp_data,sampleObj = transform_data(data,cur_tp,num_of_frames,num_of_dim)
        tp_data_list.append(tp_data)
        sample_list.append(sampleObj)
    
    tp_data_arr = np.vstack(tp_data_list)
    return tp_data_arr,sample_list

"""
plot for 2D cartesian trajectory
"""
def plotPegs_n_Traj(num_of_frames,sampleParam,expected_data,ax):

    import random
    colors = "brcmyo"
    num = int(np.floor(random.random()*4))
    expected_data = expected_data.T

    # plot the pegs
    colPegs = ['k','g']
    pegMesh = np.array([[-4,-4,-1.5,-1.5,1.5,1.5,4,4,4],[-3.5,10,10,-1,-1,10,10,-3.5,-3.5]])*0.1
    
    for i in range(num_of_frames):
        colPeg = colPegs[i]
        dispMesh = np.matmul(sampleParam.params.A[i,1:,1:],pegMesh) + sampleParam.params.b[i,1:]
        poly = Polygon(dispMesh.T, facecolor = colPegs[i])
        ax.add_patch(poly)
        ax.scatter(sampleParam.params.b[i,1],sampleParam.params.b[i,2],c="g")

    # plot trajectory
    ax.scatter(expected_data[0,0],expected_data[1,0],c="b",label="Starting")
    ax.scatter(expected_data[0,-1],expected_data[1,-1],c="b",label="Ending")
    ax.plot(expected_data[0,:],expected_data[1,:], c = colors[num], lw=2,label="Ori")

"""
save results for future comparison
"""
def save_results(patient_id,subject_id,tpgmm,time_list,recon_list,recon_lut_list,gt_list,comparators_per_var,id_list,filename):
    compile_data = []
    # save the results as follows (time,GT,recon,comparator,sample id)
    for time,recon,recon_lut,gt,comparator,sample_id in zip(time_list,recon_list,recon_lut_list,gt_list,comparators_per_var,id_list):
        compile_data.append(
            {
                "patient_id":patient_id,
                "subject_id":subject_id,
                "tpgmm_param":tpgmm.num_of_gauss,
                "var-id-case":sample_id,
                "gt":gt,
                "recon":recon,
                "recon_lut":recon_lut,
                "compare":comparator,
                }
            )
    np.save(filename, np.array(compile_data,dtype=dict))


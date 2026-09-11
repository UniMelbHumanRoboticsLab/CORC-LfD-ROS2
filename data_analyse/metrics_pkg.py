import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import numpy as np
from scipy import integrate,signal
from dtaidistance import dtw_ndim,dtw
from stats_pkg import compute_central_tendency,remove_outliers_iqr,compute_CI_scale

q = ['trunk_ie','trunk_aa','trunk_fe',
	'clav_dep_ev','clav_prot_ret',
	'shoulder_fe','shoulder_aa','shoulder_ie',
	'elbow_fe','elbow_ps']

def compute_cross_correlate(arrA,arrB):
    corr = signal.correlate(arrA, arrB,'same')
    lags = signal.correlation_lags(len(arrA), len(arrB),'same')
    corr /= np.max(corr)
    corr = np.linalg.norm(corr, axis=1)
    max_lag = lags[np.argmax(corr)]
    
    return lags, corr,max_lag

def compute_norm_error(recon,comparator):
    # normalise using mid, min and max range
    max_med_range = comparator["max"]-comparator["mid"]
    min_med_range = comparator["min"]-comparator["mid"]
    
    denum = max_med_range.copy()
    d = recon-comparator["mid"]
    # if residual below median, use the min-med range
    d_neg = d<0
    denum[d_neg] = min_med_range[d_neg]
    norm_error = d/denum
    
    # # normalise using t-score normalisation
    # d = np.abs(recon-comparator["mean"])
    # t_crit = compute_CI_scale(0.95,len(comparator["samples"]))
    # norm_error = d/(t_crit*comparator["sem"])
    
    # # normalise using median-MAD
    # d = np.abs(recon-comparator["median"])
    # norm_error = d/(comparator["mad"]*3)
    
    average_norm_error = np.mean(norm_error,axis=0)
    return average_norm_error

def compute_norm_tau_peak_diff(recon,comparator):
    # normalise using mid, min and max range
    peak_tau_recon = compute_tau_peak(recon,0)
    p = np.array(comparator["samples"])
    peak_tau_samples = compute_tau_peak(p,1)
    mid_peak_tau,max_peak_tau,min_peak_tau,mean_peak_tau,sem_peak_tau,moe_peak_tau,median_peak_tau,q1_peak_tau,q3_peak_tau,iqr_peak_tau,mad_peak_tau = compute_central_tendency(peak_tau_samples)
     
    max_med_range = max_peak_tau-mid_peak_tau
    min_med_range = min_peak_tau-mid_peak_tau
      
    denum = max_med_range.copy()
    d = peak_tau_recon - mid_peak_tau
    # if residual below median, use the min-med range
    d_neg = d<0
    denum[d_neg] = min_med_range[d_neg]
    norm_error = d/denum
    
    # # normalise using t-score normalisation
    # d = np.abs(peak_tau_recon-mean_peak_tau)
    # t_crit = compute_CI_scale(0.95,len(comparator["samples"]))
    # norm_error = d/(t_crit*sem_peak_tau)
    
    # # normalise using median-MAD
    # d = np.abs(peak_tau_recon-median_peak_tau)
    # norm_error = d/(mad_peak_tau*3)
    return norm_error

def compute_coverage(recon,comparator):
    coverage = (recon >= comparator["min"]) & (recon <= comparator["max"])
    avg_coverage = np.mean(coverage,axis=0)*100
    return avg_coverage
    
def compute_impulse(time,torque):
    """ Compute impulse of each joint using trapezoidal integration """
    torque_abs = np.abs(torque)
    impulse = []
    for i in range(torque.shape[1]):
        impulse_q = integrate.simpson(torque_abs[:,i], x=time)
        # print("impulse",q[i],impulse_q)
        impulse.append(impulse_q)
    return np.array(impulse)

def compute_tau_peak(torque,axis):
    """ Compute peak torque of each joint """
    peak_tau = np.max(np.abs(torque),axis=axis)
    return peak_tau

def compute_dtw(ref_traj,target_traj):
    # convert to double data type
    ref_traj = ref_traj.astype(np.double)
    target_traj = target_traj.astype(np.double)
    
    if len(ref_traj.shape) == 1: # univariate
        distance, paths = dtw.warping_paths_fast(ref_traj, target_traj,keep_int_repr=True)
        best_path = dtw.best_path(paths)
        similarity_score = distance / len(best_path)
    elif len(ref_traj.shape) > 1: # multidim
        distance, paths = dtw_ndim.warping_paths_fast(ref_traj, target_traj,keep_int_repr=True)
        best_path = dtw.best_path(paths)
        similarity_score = distance / len(best_path)
        
    from dtaidistance import dtw_visualisation as dtwvis
    dtwvis.plot_warpingpaths(ref_traj, target_traj,paths,path=best_path,shownumbers=False,show_diagonal=True,showlegend=True)
    return similarity_score,distance

if __name__ == "__main__":
    # sanity test for all metrics
    time = 5*np.linspace(0,1,num=100)
    cs = np.array([np.cos(2*np.pi*1*time),np.sin(2*np.pi*1*time)]).T
    sc = np.array([np.sin(2*np.pi*1*time),np.cos(2*np.pi*1*time)]).T
    sc2 = 5*sc
    
    import sys,os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data_visual.plot_pkg import plot_multi_dim,interactive_plot
    
    fig,axis = plot_multi_dim(
        [time,time,time],
        [cs,sc,sc2],
        dim = 2,
        labels=["cs","sc","sc2"],xtype="time",datatype="sinu",
        split=1,
        fig_label="sanity_check_metric"
        )
    interactive_plot(fig,axis)
    
    print("COS-SIN set metrics")
    tau_peak = compute_tau_peak(cs) # expect 1
    for i,met in enumerate(["D1","D2"]):
        print("tau peak",met,tau_peak[i])
    print()
    impulse = compute_impulse(time,cs) # expected 3.18310506506369
    for i,met in enumerate(["D1","D2"]):
        print("tau peak",met,impulse[i])
    print()
    
    print("DTW check")
    sim1,dis1 = compute_dtw(cs,sc)
    print("CS-SC:",sim1,dis1 )
    sim2,dis2 = compute_dtw(cs,sc2)
    print("CS-SC*2:",sim2,dis2)
    sim3,dis3 = compute_dtw(sc,sc2)
    print("SC-SC*2:",sim3,dis3)
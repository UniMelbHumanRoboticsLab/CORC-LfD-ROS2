import os,sys
import numpy as np
import matplotlib.pyplot as plt
import FreeSimpleGUI as sg

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from data_visual.plot_pkg import plot_multi_dim,plot_3d_submovements
from scipy.interpolate import CubicSpline

from dtaidistance import dtw_ndim,dtw
import math
from scipy.signal import butter,bessel, sosfilt_zi, sosfilt

class JointVelEstimatorND:
    def __init__(self, dim, fs=62.5, fc=10):
        self.dim = dim
        self.b, self.a = butter(N=2, Wn=fc, btype='low', fs=fs)
        print(f"Order 2 butterworth initialised: b:{self.b} a:{self.a}")
        # self.sos = bessel(2, fc, btype='low', fs=fs, output='sos')
        # n_sections = self.sos.shape[0]
        # self.zi = np.zeros((12, n_sections, 2), dtype=float)
        # print(f"Order 2 bessel initialised: sos:{self.sos}")
        

        self.prev_t = None
        self.prev_q = None  # np.array shape (dim,)

        self.x1 = np.zeros(dim)
        self.x2 = np.zeros(dim)
        self.y1 = np.zeros(dim)
        self.y2 = np.zeros(dim)
    
    def reset_filter(self):
        self.prev_t = None
        self.prev_q = None  # np.array shape (dim,)

        self.x1 = np.zeros(self.dim)
        self.x2 = np.zeros(self.dim)
        self.y1 = np.zeros(self.dim)
        self.y2 = np.zeros(self.dim)
    
    # def update(self,timestamp,q):
    #     q = np.asarray(q, dtype=float)
    #     if self.prev_t is None:
    #         self.prev_t = timestamp
    #         self.prev_q = q.copy()
    #         return np.zeros(self.dim)
        
    #     y = np.empty(12, dtype=float)
    #     new_zi = np.empty_like(self.zi)
        
    #     dt = timestamp - self.prev_t
    #     if dt <= 0:
    #         return self.y1.copy()

    #     # element-wise raw velocity
    #     qd = (q - self.prev_q) / dt
    #     for d in range(12):
    #         yd, new_zi[d] = sosfilt(self.sos, [qd[d]], zi=self.zi[d])
    #         y[d] = yd[0]
    #     self.zi = new_zi
        
    #     self.prev_t = timestamp
    #     self.prev_q = q.copy()
    #     return y
        
    def update(self, timestamp, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dim,):
            raise ValueError(f"Expected q shape ({self.dim},), got {q.shape}")

        if self.prev_t is None:
            self.prev_t = timestamp
            self.prev_q = q.copy()
            return np.zeros(self.dim)

        dt = timestamp - self.prev_t
        if dt <= 0:
            return self.y1.copy()

        # element-wise raw velocity
        x0 = (q - self.prev_q) / dt

        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        y0 = b0*x0 + b1*self.x1 + b2*self.x2 - a1*self.y1 - a2*self.y2

        self.x2, self.x1 = self.x1.copy(), x0.copy()
        self.y2, self.y1 = self.y1.copy(), y0.copy()

        self.prev_t = timestamp
        self.prev_q = q.copy()

        return y0
    
"""low pass filter"""
def lpf(time_array,data,ts=0.01,fc=50,filter_type="low",datatype="trajectory",plot_results=False):
    from scipy.fft import fft,fftfreq
    from scipy.signal.windows import blackman
    from scipy.signal import butter,sosfreqz,sosfiltfilt

    num_dtps = data.shape[0]
    dim = data.shape[1]
    filtered_data = []
    fft_arr = []
    fft_filt_arr = []
    db_arr = []

    # Define the sos filter
    fs = 1/ts  # Sampling frequency
    nyquist = fs / 2  # Nyquist frequency
    wc = fc / nyquist # Normalize cut-off frequency by Nyquist frequency (fs/2)

    xf = fftfreq(num_dtps, ts)[:num_dtps//2] # convert time to frequency domain
    window_fft = blackman(num_dtps) # define fft window
    # if filter_type=="low":
    sos = butter(4, wc, 'low', analog=False, output='sos')
    # else:
    sos2 = butter(4, [0.001/nyquist ,wc], 'bandpass', analog=False, output='sos')
    w, h = sosfreqz(sos, worN=8000, fs=fs)

    for i in range(dim):
        disturb = 0 * np.cos(2 * np.pi * 25 * time_array) # just to check the lpf
        curr_dim_data = data[:,i] + disturb

        # FFT resampled data
        rja_fft = fft(curr_dim_data*window_fft)
        fft_arr.append(np.abs(rja_fft[0:num_dtps//2]))
        # filter the data
        # if i == 0 and "q_" in datatype: # currently only remove the bias in the trunk ie ()
        #     filtered_dim_data = sosfiltfilt(sos , curr_dim_data)
        # else:
        filtered_dim_data = sosfiltfilt(sos , curr_dim_data)
        filtered_data.append(filtered_dim_data)
        # FFT filtered data
        rja_fft_filt = fft(filtered_dim_data*window_fft)
        fft_filt_arr.append(np.abs(rja_fft_filt[0:num_dtps//2]))

        db_arr.append(np.abs(h))

    filtered_data = np.array(filtered_data).T
    fft_arr = np.array(fft_arr).T   
    fft_filt_arr = np.array(fft_filt_arr).T
    db_arr = np.array(db_arr).T

    if plot_results:
        # plot_multi_dim(
        #     [xf, xf],
        #     [fft_arr, fft_filt_arr],
        #     dim,
        #     ['noisy', 'filtered'],
        #     'Frequency',
        #     f'FFT {datatype}',
        #     semilogx=True,
        #     fig_label=f'FFT_{datatype}'
        # )
        # plot_multi_dim(
        #     [w],
        #     [db_arr],
        #     dim,
        #     ['FreqResp'],
        #     'Frequency (Hz)',
        #     f'Frequency Response {datatype}',
        #     semilogx=True,
        #     fig_label=f'FreqResp_{datatype}'
        # )
        plot_multi_dim(
            [time_array,time_array],
            [data, filtered_data],
            dim,
            ['noisy', 'filtered'],
            'Time',
            datatype,
            fig_label=f'LFP_{datatype}'    
        )
    return filtered_data

"""calculate fixed derivative using finite differences"""
def calc_fixed_diff(data, dt):
    derivative = np.diff(data, axis=0) / dt
    derivative = np.vstack([derivative, derivative[-1]])
    return derivative

"""calculate magnitude of vectors"""
def calc_mag(vectors):
    return np.linalg.norm(vectors, axis=1)[:,np.newaxis]

def closest_indices(time, selected):
    time = np.asarray(time)
    selected = np.asarray(selected)

    idx = np.empty(len(selected), dtype=int)
    for k, t in enumerate(selected):
        # index of smallest absolute difference
        idx[k] = int(np.argmin(np.abs(time - t)))
    return idx
"""segment submovements for current repetition"""
def segment_sbmvmts(time_array,hand_traj,hand_speed,submovement_num,data_path,redo=True,skeleton=False):
    data_path_split = data_path.split("/")
    
    if submovement_num == 1:
        segment_type = "StartEnd"
    else:
        segment_type = "Submovements"
        
    if os.path.isfile(data_path):
        print(f"{data_path_split[-1]} exists")
        # read sbmvmt_indices from txt file
        with open(data_path, 'r') as f:
            lines = f.readlines()
            sbmvmt_indices = np.array(lines[0].split(": ")[1].split(), dtype=int)
            
            if segment_type != "Submovements":
                indexed_time = np.array(lines[1].split(": ")[1].split(), dtype=float)
            else:
                p = lines[1].split(": ")[1].split("\n")[0]
                q = lines[2].split(": ")[1].split("\n")[0]
                import re
                indexed_time = np.array([float(x) for x in re.findall(r"\[([^\]]+)\]", p)])
                if len(indexed_time) == 0:
                    indexed_time = np.array(lines[1].split(": ")[1].split(), dtype=float)
                indexed_time_norm = np.array([float(x) for x in re.findall(r"\[([^\]]+)\]", q)])
                if len(indexed_time_norm) == 0:
                    indexed_time_norm = np.array(lines[2].split(": ")[1].split(), dtype=float)
            print(f"Saved {segment_type} Indices:", sbmvmt_indices)
            print(f"Saved {segment_type} Times:", indexed_time)
            if segment_type == "Submovements":
                print(f"Saved {segment_type} Norm Times:", indexed_time_norm)
                
            # reindex if given time array is different from previous iteration time array
            new_sbmvmt_indices = closest_indices(time_array, indexed_time)
            # reuse the first one
            new_sbmvmt_indices[0] = sbmvmt_indices[0]
            # new_sbmvmt_indices[-1] = sbmvmt_indices[-1]
            print(f"New {segment_type} Indices:", new_sbmvmt_indices)
            print(f"New {segment_type} Times:", np.array(time_array[new_sbmvmt_indices],dtype=float))
            
            sbmvmt_indices = new_sbmvmt_indices
            time_array_norm = (time_array-time_array[0])/(time_array[-1]-time_array[0])
        if redo:
            # placehold = sg.popup('',location=(1550,300),non_blocking=True)  
            plot_3d_submovements(hand_traj,sbmvmt_indices=sbmvmt_indices,skeleton=skeleton)
            plt.show(block=False)
            ok_or_not = sg.popup_yes_no('Ok or not',location=(3400,300),non_blocking=False)  
            # placehold.close()
            plt.close("all")
            if ok_or_not == "Yes":
                good = True
            else:
                good = False
        else:
            good = True
    else:
        print(f"{data_path_split[-1]} DNE")
        sbmvmt_indices = [0,-1]
        good = False
        
        
    while not good:
        from mpl_point_clicker import clicker
        placehold = sg.popup('',location=(1550,300),non_blocking=True)  # remember to FSG change source code to return window for non-blocking = True popups
        plot_3d_submovements(hand_traj,sbmvmt_indices=sbmvmt_indices,skeleton=skeleton)
        fig,axs = plot_multi_dim([time_array], [hand_speed], 1, ['speed'], 'Time', 'Hand Speed',fig_label= data_path)
        klicker = clicker(axs[0], ["event"], markers=["x"])
        
        from typing import Tuple
        def point_added_cb(position: Tuple[float, float], klass: str):
            x, y = position
            
            points = klicker.get_positions()['event']
    
            sbmvmt_indices = []
            for point in points:
                idx = (np.abs(time_array - point[0])).argmin()
                sbmvmt_indices.append(idx)
            if submovement_num != 1:
                sbmvmt_indices = [0] + sbmvmt_indices + [len(time_array)-2]
            sbmvmt_indices.sort()
            
            print(klicker.get_positions()['event'][-1])
            plt.close("Hand Traj")
            plot_3d_submovements(hand_traj,sbmvmt_indices=sbmvmt_indices,skeleton=skeleton)
            plt.show(block=False)
        
        def point_removed_cb(position: Tuple[float, float], klass: str, idx):
            points = klicker.get_positions()['event']
    
            sbmvmt_indices = []
            for point in points:
                idx = (np.abs(time_array - point[0])).argmin()
                sbmvmt_indices.append(idx)
            if submovement_num != 1:
                sbmvmt_indices = [0] + sbmvmt_indices + [len(time_array)-2]
            sbmvmt_indices.sort()
            
            # print(klicker.get_positions()['event'])
            plt.close("Hand Traj")
            plot_3d_submovements(hand_traj,sbmvmt_indices=sbmvmt_indices,skeleton=skeleton)
            plt.show(block=False)

        klicker.on_point_added(point_added_cb)
        klicker.on_point_removed(point_removed_cb)
        
        plt.show(block=False)
        sg.popup('Close all',location=(3400,300),non_blocking=False)  
        placehold.close()
        plt.close("all")
        points = klicker.get_positions()['event']

        sbmvmt_indices = []
        for point in points:
            idx = (np.abs(time_array - point[0])).argmin()
            sbmvmt_indices.append(idx)
        if submovement_num != 1:
            sbmvmt_indices = [0] + sbmvmt_indices + [len(time_array)-2]
        sbmvmt_indices.sort()

        if len(sbmvmt_indices) != submovement_num+1:
            print("Restart Selection:",len(sbmvmt_indices), "selected, need", submovement_num+1)
        else:
            print(f"{segment_type} Indices:", sbmvmt_indices)
            print(f"Selected {segment_type} Time:",np.array(time_array[sbmvmt_indices],dtype=float))
            plot_3d_submovements(hand_traj,sbmvmt_indices=sbmvmt_indices,skeleton=skeleton)
            plt.show(block=False)
            ok_or_not = sg.popup_yes_no('Ok or not',location=(3400,300),non_blocking=False)  
            plt.close("all")
            if ok_or_not == "Yes":
                good = True
                sbmvmt_indices = np.array(sbmvmt_indices,dtype=int)
                time_array_norm = (time_array-time_array[0])/(time_array[-1]-time_array[0])
            else:
                good = False

    if segment_type != "Submovements":
        indexed_time = np.array(time_array[sbmvmt_indices],dtype=float)
        # save to txt file
        with open(data_path, 'w') as f:
            f.write("sbmvmt_indices: " + " ".join(map(str, sbmvmt_indices)) + "\n")
            f.write("indexed_time: " + " ".join(map(str, indexed_time)) + "\n") 
    else:
        indexed_time = np.array(time_array[sbmvmt_indices],dtype=float)
        indexed_time_norm = np.array(time_array_norm[sbmvmt_indices],dtype=float)
        # save to txt file
        with open(data_path, 'w') as f:
            f.write("sbmvmt_indices: " + " ".join(map(str, sbmvmt_indices)) + "\n")
            f.write("indexed_time: " + " ".join(map(str, indexed_time)) + "\n") 
            f.write("indexed_time_norm: " + " ".join(map(str, indexed_time_norm)) + "\n") 
    
    if submovement_num != 1:
        indices_array = np.ones((time_array.shape[0],1))
        for i in range(len(sbmvmt_indices)-1):
            start_idx = sbmvmt_indices[i]
            end_idx = sbmvmt_indices[i+1]
            indices_array[start_idx:end_idx,0] = i+1
        indices_array[sbmvmt_indices[len(sbmvmt_indices)-1]:] = i+1
    else:
        sbmvmt_indices[-1] = sbmvmt_indices[-1]+1 # add one more index for the active movement segmentation
        indices_array = np.zeros((time_array.shape[0],1))
    print()
    return indices_array,sbmvmt_indices

""" up/down sample data to new time array"""
def rescale(t_old, data,t_new,datatype="trajectory",plot_results=False):
    interp = CubicSpline(t_old, data, axis=0,bc_type='natural')
    data_interp = interp(t_new)
    dim = data.shape[1]
    if plot_results:
        plot_multi_dim(
            [t_old,t_new],
            [data, data_interp],
            dim,
            ['old scale', 'new scale'],
            'Time',
            datatype,
            fig_label=f'rescale_{datatype}'    
        )
    return data_interp

""""alignment using DTW """
def align_dtw(ref_traj,target_traj):
    # convert to double data type
    ref_traj = ref_traj.astype(np.double)
    target_traj = target_traj.astype(np.double)
    
    norm_pos = np.linalg.norm(ref_traj[:,:10],axis=1)
    norm_ref_pos_traj = ref_traj[:,:10]/norm_pos[:,np.newaxis]
    norm_ref_vel_traj = ref_traj[:,10:]/np.linalg.norm(ref_traj[:,10:],axis=1)[:,np.newaxis]
    
    norm_target_pos_traj = target_traj[:,:10]/np.linalg.norm(target_traj[:,:10],axis=1)[:,np.newaxis]
    norm_target_vel_traj = target_traj[:,10:]/np.linalg.norm(target_traj[:,10:],axis=1)[:,np.newaxis]
    
    norm_ref_traj = np.hstack((norm_ref_pos_traj,norm_ref_vel_traj))
    norm_target_traj = np.hstack((norm_target_pos_traj,norm_target_vel_traj))
    
    if len(ref_traj.shape) == 1: # univariate
        distance, paths = dtw.warping_paths_fast(ref_traj, target_traj,keep_int_repr=True)
        best_path = dtw.best_path(paths)
    elif len(ref_traj.shape) > 1: # multidim
        distance, paths = dtw_ndim.warping_paths_fast(norm_ref_traj, norm_target_traj,keep_int_repr=True)
        best_path = dtw.best_path(paths)
        
    
    # from dtaidistance import dtw_visualisation as dtwvis
    # dtwvis.plot_warpingpaths(ref_traj, target_traj,paths,path=best_path,shownumbers=False,show_diagonal=True,showlegend=True)
    return best_path

def warp_target_to_ref(ref, target, path):
    """
    Warp series B to align with series A
    Returns a warped version of B with same length as A
    """
    warped_target = np.zeros((len(ref),target.shape[1]))
    
    # Create mapping from A indices to B indices
    a_to_b_mapping = {}
    for a_idx, b_idx in path:
        if a_idx not in a_to_b_mapping:
            a_to_b_mapping[a_idx] = []
        a_to_b_mapping[a_idx].append(b_idx)
    
    # Map values from B to A using the warping path
    for a_idx in range(len(ref)):
        if a_idx in a_to_b_mapping:
            # Average if multiple B indices map to same A index
            b_indices = a_to_b_mapping[a_idx]
            warped_target[a_idx] = np.mean([target[b_idx,:] for b_idx in b_indices],axis=0)
    
    return warped_target
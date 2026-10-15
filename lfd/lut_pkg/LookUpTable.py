import numpy as np
np.set_printoptions(suppress=True,precision=4) # suppress scientific notation
realmin  = np.finfo(np.float64).tiny
realmax  = np.finfo(np.float64).max

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..',".."))
from data_process.post_process_pkg import lpf
from sklearn.metrics.pairwise import euclidean_distances

class LookUpTable:
    def __init__(self, all_data,input_index,output_index):
        self.all_data = all_data
        self.all_input_data = all_data[:,0:input_index+1]
        self.all_output_data = all_data[:,input_index+1:]
        hi = self.all_input_data
        hi2 = self.all_output_data
        pass
        
    def search_closest_output(self,time,input_data):
        closest_output = []
        p = self.all_output_data
        for t in input_data:
            # Calculate Euclidean distances between the query vector and the dataset
            distances = euclidean_distances(self.all_input_data, [t])
            
            # Find the closest vector
            closest_index = np.argmin(distances)
            closest_vector = self.all_output_data[closest_index,:]
            # closest_vector = self.all_output_data[closest_index]
            closest_output.append(closest_vector)
        closest_output = np.array(closest_output)
        filtered_output   = lpf(time,closest_output,ts=1/100,fc=3,filter_type="low",datatype="LUT", plot_results=False)
        
        return filtered_output
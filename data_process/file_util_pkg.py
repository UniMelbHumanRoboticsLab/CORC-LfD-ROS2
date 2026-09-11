import pandas as pd
import numpy as np
import os 

""" compile raw data"""
def get_raw_data_with_vel(data_path):
    data = pd.read_csv(data_path)
    time_data = data["time"].values
    joints = ['trunk_ie','trunk_aa','trunk_fe',
				'clav_dep_ev','clav_prot_ret',
				'shoulder_fe','shoulder_aa','shoulder_ie',
				'elbow_fe','elbow_ps',
				'wrist_fe','wrist_dev']
    q_right = data[[f"q_{joint}" for joint in joints]].values
    qdot_right = data[[f"qdot_{joint}" for joint in joints]].values
    
    return data,time_data,q_right,qdot_right

""" select which variation for train and for test """
def level_sub(sample_id: int) -> tuple[int, int]:
    """
    Map an ID to (level, sublevel).

    IDs:
      1,2,3 -> level 1, sublevels 1,2,3
      4,5,6 -> level 2, sublevels 1,2,3
    """
    level = 1 if sample_id <= 3 else 2
    sub = ((sample_id - 1) % 3) + 1
    return level, sub
def valid_test_combis(n=6,num_train=5) -> list[tuple[int, int]]:
    from itertools import combinations
    ids = range(1, n + 1)
    valid_tests = []
    valid_trains = []
    for combi in combinations(ids, num_train):
        
        selected_var = []
        lvl_subs = []
        for item in combi:
            selected_var.append(item)
            lvl_subs.append(level_sub(item))
        # (first,second,third) = combi
        # lvl_sublvl1 = level_sub(first)
        # lvl_sublvl2 = level_sub(second)
        # lvl_sublvl3 = level_sub(third)
        
        boundB = {1,3}
        # B levels (sublevels) present in this combination
        covered_subs = set()
        # A levels (levels) counts in this triple
        covered_levels = []
        for lvl_sub in lvl_subs:
            covered_subs.add(lvl_sub[1])
            covered_levels.append(lvl_sub[0])
                   
        from collections import Counter
        level_counts = Counter(covered_levels)
        balance = num_train/2 if num_train%2 == 0 else (num_train+1)/2

        if (boundB.issubset(covered_subs)) and max(level_counts.values()) <= balance: 
            # covers endpoints b1,b3
            # A levels is not dominant on 1 A level
            # train conditions satisfy, take the rest as test variations
            all_numbers = {1, 2, 3, 4, 5, 6}
            remaining = all_numbers - set(combi)
            valid_tests.append(list(remaining))
            valid_trains.append(list(combi))
            
    # valid train combinations for each variation
    var_train_combis = {}
    for i in range(1,n+1):
        var_train_combis[f"var_{i}"] = []
        for combi in valid_trains:
            if i not in combi:
                var_train_combis[f"var_{i}"].append(combi)
    return valid_tests,valid_trains,var_train_combis
def separate_train_test_val(variant_list,path,num_train=5,num_val=1,num_monte=4):
    # Given numbers
    numbers = len(variant_list)

    valid_tests,valid_trains,_ = valid_test_combis(numbers,num_train)
        
    # iterate through every valid test combination
    for i,test in enumerate(valid_tests):
        case_id = []
        train_var = []
        test_var = []
        
        # Iterate through the files in the source folder
        for var in variant_list:
            var_id = int(var[-1])
            # Check if its test or train 
            if var_id in test:
                case_id.append("test")
                test_var.append(var)
            else:
                case_id.append("train")
                train_var.append(var)
    
        # save train/test split to csv
        train_test_df = pd.DataFrame({
            "variant": variant_list,
            "split": case_id
        })
        train_test_df.to_csv(f'{path}/splits/{i}_train_test.csv',index=True)
        
        # select 1 repetition out of 4 for validation
        selected_reps = np.zeros((num_monte,len(train_var)))
        for j in range(num_monte):
            train_reps = []
            val_reps = []
            case_id = []
            repetitions = []
            # randomly select one rep for validation, rest for training
            for k,var in enumerate(train_var):
                val_rep_num = np.random.randint(1, 4 + 1)
                
                if j != 0:
                    while np.isin(val_rep_num,selected_reps[:,k]):
                        val_rep_num = np.random.randint(1, 4 + 1)

                selected_reps[j,k] = val_rep_num
                for rep in range(1, 4 + 1):
                    rep_path = f"{var}/processed/UBORecord{rep}Log.csv"
                    repetitions.append(rep_path)
                    if rep == val_rep_num:
                        val_reps.append(rep_path)
                        case_id.append("val")
                    else:
                        train_reps.append(rep_path)
                        case_id.append("train")
                
            # save train/val split to csv
            train_val_df = pd.DataFrame({
                "repetition": repetitions,
                "split": case_id
            })
            train_val_df.to_csv(f'{path}/splits/{i}_train_val_{j}.csv', index=False)

        all_numbers = {1, 2, 3, 4, 5, 6}
        remaining = all_numbers - set(test)
        print("Train Var: ",remaining)
        print("Test Var: ",test)
        for sample,val_rep in enumerate(selected_reps.astype('int')):
            print_text = f"Sample {sample}: "
            for var,rep in zip(train_var,val_rep):
                print_text += f"{var}-Rep{rep}, "
            print(print_text,"\n")

""" compile raw data"""
def get_raw_data(data_path):
    data = pd.read_csv(data_path)
    time_data = data["elapsed_time"].values
    corc_data = data[["F1x", "F1y", "F1z", "T1x", "T1y", "T1z",
			"F2x", "F2y", "F2z", "T2x", "T2y", "T2z",
			"F3x", "F3y", "F3z", "T3x", "T3y", "T3z"]].values
    joints = ['trunk_ie','trunk_aa','trunk_fe',
				'clav_dep_ev','clav_prot_ret',
				'shoulder_fe','shoulder_aa','shoulder_ie',
				'elbow_fe','elbow_ps',
				'wrist_fe','wrist_dev']
    right_xsens_data = data[[f"{joint}_right" for joint in joints]].values
    joint_rad = np.radians(right_xsens_data)
    unwrapped_deg = np.degrees(np.unwrap(joint_rad, axis=0))
    
    # standardize discontinuity unwrapping to -pi to 2pi
    corrected = unwrapped_deg.copy().astype(float)
    # Check each joint (column)
    for joint_idx in range(unwrapped_deg.shape[1]):
        joint_values = unwrapped_deg[:, joint_idx]
        
        # Check if any value is less than -180 (unwrapped opposite direction)
        if np.any(joint_values < -180):
            # This joint was unwrapped in opposite direction
            # Add 360 to bring it to [0, 360] range
            corrected[:, joint_idx] = joint_values + 360
    
    return data,time_data,corc_data,corrected

""" compile processed data into train test val"""
def get_processed_data(data_path,degree_flag=False):
    q = ['trunk_ie','trunk_aa','trunk_fe','clav_dep_ev','clav_prot_ret','shoulder_fe','shoulder_aa','shoulder_ie','elbow_fe','elbow_ps']
    qdot = [f"{joint}_dot" for joint in q]
    tau = [f"tau_{joint}" for joint in q]
    
    data = pd.read_csv(data_path)
    time_data = data["norm_time"].values
    indices = data["index"].values
    if degree_flag:
        joint_kinematics = np.rad2deg(data[q+qdot].values) # change back to degrees to check
    else:
        joint_kinematics = data[q+qdot].values # change back to degrees to check
    joint_torques = data[tau].values

    return time_data,indices,joint_kinematics,joint_torques
def compile_train_val_test_data(session_data,task_path,combi,sample,degree_flag=False):
	train_test_split = pd.read_csv(f'{task_path}/splits/{combi}_train_test.csv')[["split"]].values
	train_val_df = pd.read_csv(f'{task_path}/splits/{combi}_train_val_{sample}.csv')
	train_val_split = dict(zip(train_val_df["repetition"], train_val_df["split"]))

	train_list = []
	val_list = []
	test_list = []

	for case,var in zip(train_test_split,session_data["variants"]):
		reps = range(1,session_data["num_rep"]+1)
		for rep in reps:
			time,_,q_qdot,tau	= get_processed_data(f'{task_path}/{var}/processed/UBORecord{rep}Log.csv',degree_flag)
			tp 				= pd.read_csv(f'{task_path}/{var}/processed/tp/UBOTP{rep}Log.csv').values
			if degree_flag:
				tp[:,900:920] = np.rad2deg(tp[:,900:920])  # change back to degrees to check
			var_rep = {
			"time":time,
			"data":np.hstack((q_qdot,tau)),
			"tp":tp
			}

			if case[0] == "train":
				# check if var_rep is train or val
				split = train_val_split[f"{var}/processed/UBORecord{rep}Log.csv"]
				var_rep["id"] = f"{var}.{rep}.{split}"
				if split == "train":
						train_list.append(var_rep)
				elif split == "val":
						val_list.append(var_rep)
			elif case[0] == "test":
				# test list
				var_rep["id"] = f"{var}.{rep}.{case[0]}"
				test_list.append(var_rep)
	return train_list,val_list,test_list

""" create directory if needed"""
def create_dir(data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

""" split all variations / repetitions into a list of variations of a list of repetitions"""
def split_reps(data_list:list=[],n=4):
	result = [data_list[i:i+n] for i in range(0, len(data_list), n)]
	return result

""" load / save npy files for each samples and convert them to list"""
def load_npy(data_path):
    return  np.load(data_path,allow_pickle=True).tolist()
def save_npy(data_path,compile_data):
    np.save(data_path, np.array(compile_data,dtype=dict))
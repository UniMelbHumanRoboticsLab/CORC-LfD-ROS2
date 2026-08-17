
"""
upper torso class definition for upper torso model handling and inverse kinematics

@author: JQ
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from data_visual.plot_pkg import create_custom_3d_fig


# Set NumPy print options to show decimal format instead of exponential
np.set_printoptions(
    precision=3,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__)))
from ub_models import *
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

class ub():
    """Init Functions"""
    def __init__(self, body_params={'torso':0.5,'clav':0.2,'ua_l': 0.3, 'fa_l': 0.25, 'ha_l': 0.1, 'm_ua': 2.0, 'm_fa':1.1+0.23+0.6,"shoulder_aa_offset":[16],"ft_offsets": [0.1,0.1,0.1]},model='xsens',arm_side="right"):
        self.model = model
        if self.model == 'ubo':
            self.ub_model = ubo_robot(
                torso=body_params['torso'],
                clav=body_params['clav'],
                ua_l=body_params['ua_l'],
                fa_l=body_params['fa_l'],
                ha_l=body_params['ha_l'],
                m_ua=body_params['m_ua'],
                m_fa=body_params['m_fa'],
                shoulder_aa_offset=body_params['shoulder_aa_offset'],
                ft_offsets =body_params['ft_offsets'],
                arm_side=arm_side
            )
            self.ee_names = ["hand","clavicle","upper_arm","forearm"]
        elif self.model == 'xsens':
            self.ub_model = [xsens_ub_12dof(
                torso=body_params['torso'],
                clav=body_params['clav'],
                ua_l=body_params['ua_l'],
                fa_l=body_params['fa_l'],
                ha_l=body_params['ha_l'],
                m_ua=body_params['m_ua'],
                m_fa=body_params['m_fa'],
                shoulder_aa_offset=body_params['shoulder_aa_offset'],
                arm_side=arm_side
            )]
            self.ee_names = ["hand"]
        # dominant hand
        self.cur_side = arm_side
        
        #Overall arm mass
        self.Marm=0
        for l in self.ub_model[0].links:
            self.Marm+=l.m

        #Default gravity vector (can be changed)
        self.ub_model[0].gravity=[0,0,-9.81]
    def ArmMassFromBodyMass(self, body_mass: float):
        '''Calculate arm mass from overall body mass based on anthropomorphic rules
        from Drillis et al., Body Segment Parameters, 1964. Table 7'''
        UA_percent_m = 0.053
        FA_percent_m = 0.036
        hand_percent_m = 0.013
        self.ub_model[0][2].m = UA_percent_m*body_mass
        self.ub_model[0][4].m = (FA_percent_m+hand_percent_m)*body_mass
        self.Marm=0
        for l in self.ub_model[0].links:
            self.Marm+=l.m

    def SetGravity(self,g_vector: np.array =[0,0,-9.81]):
        '''Define (set) gravitational vector of the model'''
        self.ub_model[0].gravity=g_vector
    """
    Plotting
    """
    def plot_skeleton(self,ax,joints_pose,color='k'):
        # plot skeleton
        x = np.array([joints_pose[0].t[0],joints_pose[4].t[0],joints_pose[6].t[0],joints_pose[8].t[0],joints_pose[10].t[0]])
        y = np.array([joints_pose[0].t[1],joints_pose[4].t[1],joints_pose[6].t[1],joints_pose[8].t[1],joints_pose[10].t[1]])
        z = np.array([joints_pose[0].t[2],joints_pose[4].t[2],joints_pose[6].t[2],joints_pose[8].t[2],joints_pose[10].t[2]])
        ax.plot(x,y,z,marker='.', linestyle='-', color=color)
    def plot_joints_ees_frame(self,joints_pose,ees_pose,show_joint_frames=True,show_ee_frames=True):
        fig,ax = create_custom_3d_fig(num="UB")
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1,projection='3d')
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)
        ax.set_zlim(-0.5,0.5)
        ax.set_box_aspect([1, 1, 1])
        
        self.plot_skeleton(ax,joints_pose)

        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())+list(mcolors.TABLEAU_COLORS.values())
        if show_joint_frames:
            for i,robot_joint_pose in enumerate(joints_pose):
                if i == 0 :
                    off = np.array([0.0,0.0,0.0])
                else:   
                    off = np.array([0.02,-0.02,-0.02])
                offset = SE3.Rt(robot_joint_pose.R, robot_joint_pose.t+robot_joint_pose.R@off)
                offset.plot(frame=f"{i}", length=0.05, ax=ax,color=colors[i],flo=(0.005,0.005,0.005))
        if show_ee_frames:
            for i, (ee_pose, frame_name) in enumerate(zip(ees_pose,self.ee_names)):
                ee_pose.plot(frame=frame_name, length=0.05, ax=ax,color='k',flo=(0.01,0.01,0.01))
    """
    Inverse Kinematics
    Estimate new joint angles (task parameters) from new task points
    - TODO: develop an IK that can recreate that without giving the explicit joint angles 
    """
    def IK_task_params(self,task_points,joints_traj,point_index,visual_ik):
        qq_new = []
        qq_new.append(joints_traj.values[point_index[0]]) # initial configuration
        for i,task_point in enumerate(task_points):
            TT=SE3(task_point)
            iter = 0
            success=0
            while (success == 0 and iter < 3):
                iter = iter+1
                # To replace with a custom IK that follows healthy body constraints
                q0 = np.array(joints_traj.values[point_index[i+1]].tolist())
                qq, success, iterations, searches, residual=self.ub_model[0].ik_GN(TT, q0=q0, mask=[1, 1, 1, 1, 1, 1],tol=1e-6,pinv=True)
            qq_new.append(qq)
            if success == 0:
                assert 0
            print(f"Success / Iterations/ Searches:",success,iterations,searches)
            print("Intended Task Param:",q0*180/np.pi,"\nCalculated Task Param:",qq*180/np.pi)
            print("Actual Task Point:",task_point.t)
            print("Assumed Task Point:",self.ub_model[0].fkine(q0).t)
            print("Estimated Task Point:",self.ub_model[0].fkine(qq).t)
            print()

        qq_new.append(joints_traj.values[point_index[-1]-1]) # take the one just before resting submovement
        if visual_ik:
            fig = plt.figure()
            self.ub_model[0].plot(q = np.array(qq_new),
                        backend='pyplot',block=False,loop=False,jointaxes=True,eeframe=True,shadow=False,fig=fig,dt=3)
            for qq in qq_new:
                joints_pose=self.ub_model[0].fkine_all(np.array(qq)) # this has all the frames of the robot joints
                ee_pose = [self.ub_model[0].fkine(np.array(qq))]
                self.plot_joints_ees_frame(joints_pose,ee_pose)
            
        qq_new = pd.DataFrame(qq_new, columns=list(joints_traj.columns))
        return qq_new
    """
    Forward Kinematics
    """
    def fkine(self,joints_config:list):
        # supports both single joints config or joints trajectory
        ees_pose = []
        joints_config = np.array(joints_config)
        if joints_config.ndim == 1:
            joints_config = joints_config[np.newaxis, :]

        for i, robot in enumerate(self.ub_model): 
            posture = joints_config[:,:(robot.n)] 
            ees_pose.append(robot.fkine(np.deg2rad(posture)))
        return ees_pose
    def fkine_all(self,joints_config:list):
        joints_pose = self.ub_model[0].fkine_all(np.deg2rad(np.array(joints_config))) # this has all the frames of the robot joints
        return joints_pose
    def ub_fkine(self,joints_config:list):
        # for single joints configuration atm
        joints_pose,ees_pose = self.fkine_all(joints_config),self.fkine(joints_config)
        return joints_pose,ees_pose
    """
    Inverse Dynamics
    """
    def get_joints_torque(self,joints_config:np.ndarray,ees_wrench:np.ndarray):
        taus = []
        total_tau = np.zeros(self.ub_model[-1].n)
        # self.ub_model[0].jacobe()
        for i, robot in enumerate(self.ub_model[1:]): 
            ee_wrench = ees_wrench[i*6:i*6+6]
            posture = joints_config[:(robot.n)]
            J = robot.jacobe(np.deg2rad(np.array(posture)))
            tau = J.T @ ee_wrench

            tau_padded = np.zeros(self.ub_model[-1].n)
            tau_padded[:robot.n] = tau
            total_tau += tau_padded
            taus.append(tau)
        return taus, total_tau
    
    def get_joints_torques_traj(self,joints_config_traj:np.ndarray,ees_wrench_traj:np.ndarray):
        num_samples = joints_config_traj.shape[0]
        taus_dict = {"total":
                     {
                         "raw": np.empty((num_samples,self.ub_model[-1].n))
                     }}
        for robot in self.ub_model[1:]: 
            taus_dict[f'{robot.name}'] = {
                "raw": np.empty((num_samples,robot.n))
            }

        for j,(joints_config, ees_wrench) in enumerate(zip(joints_config_traj, ees_wrench_traj)):
            taus, total_tau = self.get_joints_torque(joints_config, ees_wrench)
            for tau, robot in zip(taus,self.ub_model[1:]):
                taus_dict[f'{robot.name}']["raw"][j] = tau
            taus_dict["total"]["raw"][j] = total_tau
        return taus_dict

if __name__ == "__main__":
    #Define ISB rtb arm model
    body_params = {
        "body_height": 1770.0,
        "shoulder_height": 1520,
        "shoulder_width": 390.0,
        "elbow_span": 800.0,
        "wrist_span": 1320.0,
        "arm_span": 1710.0,
        "torso": 520.0,
        "clav": 200.0,
        "ua_l": 260.0,
        "fa_l": 250.0,
        "ha_l": 70.0,
        "ft_offsets": {
            "clav": [0.0,100.0],
            "ua":   [0.0,130.0],
            "fa":   [0.0,200.0]
        },
        "ft_grav": {
            "clav": 0,
            "ua":   0,
            "fa":   0
        },
        "shoulder_aa_offset": [0,0]
        }
    body_params_rbt = {'torso': body_params["torso"]/1000,
                    'clav': body_params["clav"]/1000,
                    'ua_l': body_params["ua_l"]/1000,
                    'fa_l': body_params["fa_l"]/1000,
                    'ha_l': body_params["ha_l"]/1000,
                    'm_ua': 2.0,
                    'm_fa': 1.1+0.23+0.6,
                    "shoulder_aa_offset": np.array(body_params["shoulder_aa_offset"]),
                    "ft_offsets": body_params["ft_offsets"]}
    ub_xsens = ub(body_params_rbt,model="ubo",arm_side="right")
    ub_xsens_left = ub(body_params_rbt,model="ubo",arm_side="left")

    """
    Sanity check your UA model here
    """
    ub_postures_to_test = [[0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0]]

    for joints_config in tqdm(ub_postures_to_test):
        joints_pose, ee_pose = ub_xsens.ub_fkine(joints_config)
        ub_xsens.ub_model[0].plot(q=np.deg2rad(np.array(joints_config)))
        ub_xsens.plot_joints_ees_frame(joints_pose,ee_pose)

        # wrenches = np.array([[0,0,-10,0,0,0,
        #             10,0,0,0,0,0,
        #             0,0,-0,3,2.5,0]])
        # tau_dict = ub_xsens.get_joints_torques_traj(np.array([joints_config]),wrenches)
        # print()
        # for i,(key,tau) in enumerate(zip(tau_dict.keys(),tau_dict.values())):
        #     if key != "total":
        #         i = i-1
        #         print(f"{key} Wrenches\t:{wrenches[0,i*6:i*6+6]}")
        #         print(f"{key} joint torques (Nm)\t:{tau['raw']}")
        #         print()

        # joints_pose, ee_pose = ub_xsens_left.ub_fkine(joints_config)
        # ub_xsens_left.plot_joints_ees_frame(joints_pose,ee_pose)
        plt.show(block=True)
        
        
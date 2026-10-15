# -*- coding: utf-8 -*-
"""
torso model functions that defines a rbt-toolbox robot

supported models
- 12 DOF upper torso model based on XSENS Biomechanical model
- 5 DOF upper limb model based on ISB recommendations using Vive trackers

@author: JQ
"""
from tqdm import tqdm
import numpy as np
from spatialmath import SO3, SE3
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from data_visual.plot_pkg import create_custom_3d_fig
import roboticstoolbox as rbt
from spatialmath import SO3, SE3

def emu() -> rbt.Robot:
    """
    xsens_ub_12dof Create a Robot of robotic toolbox Xsens compatible upper body (pelvis to hand)
    torso: torso length
    clav: clavicle length
    ua_l: upper-arm length
    fa_l: forearm length
    ha_l: hand length
    m_ua and m_fa: upper-arm and forearm masses, centered in middle of segment
    - internal/external rotation are not following ISBUL standard for rehabilitation application
    """
    L = [] #Links list
    # ROM: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7549223/#:~:text=Normal%20range%20of%20active%20movement,for%20external%20rotation%20%5B6%5D.
    L.append(rbt.RevoluteMDH(d=0,a=0,alpha=0,offset=0,name='q1'))
    L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=np.pi/2,name='q2'))
    L.append(rbt.RevoluteMDH(d=0,a=0.5,alpha=0,offset=np.pi/2,name='q3'))
    L.append(rbt.RevoluteMDH(d=0,a=0.615,alpha=0,offset=0,name='q4'))

    emu = rbt.DHRobot(L,name="emu")
    # xsens.base=SE3([-0.5,-0.5,-0.5]) * SE3(SO3.Rz(np.pi/2))
    
    #Add hand transformation (tool) to match XSENS model wrist offset
    emu.tool= SE3([0,-0,0]) * SE3(SO3.Ry(-np.pi/2))#*SE3(SO3.Rx(0)) # for intrinsic rotation (rotation about local axis), always use post multiply
    
    return emu

class emu_robot():
    """Init Functions"""
    def __init__(self):
        self.model_name = "emu"
        if self.model_name == 'emu':
            self.model = [emu()]
            self.ee_names = ["ft"]
    """
    Plotting
    """
    def plot_skeleton(self,ax,joints_pose,color='k'):
        # plot skeleton
        x = np.array([joints_pose[0].t[0],joints_pose[1].t[0],joints_pose[2].t[0],joints_pose[3].t[0],joints_pose[4].t[0]])
        y = np.array([joints_pose[0].t[1],joints_pose[1].t[1],joints_pose[2].t[1],joints_pose[3].t[1],joints_pose[4].t[1]])
        z = np.array([joints_pose[0].t[2],joints_pose[1].t[2],joints_pose[2].t[2],joints_pose[3].t[2],joints_pose[4].t[2]])
        ax.plot(x,y,z,marker='.', linestyle='-', color=color)
    def plot_joints_ees_frame(self,joints_pose,ees_pose,show_joint_frames=True,show_ee_frames=True,ax=None):
        if ax == None:
            fig,ax = create_custom_3d_fig(num="UB")
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1,projection='3d')
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)
        ax.set_zlim(-0.5,0.5)
        ax.set_box_aspect([1, 1, 1])
        
        self.plot_skeleton(ax,joints_pose)

        import matplotlib.colors as mcolors
        base = SE3.Rt(np.eye(3), np.array([0,0,0]))
        base.plot(frame="emu", length=0.05, ax=ax,color="k",flo=(0.005,0.005,0.005))
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
    Forward Kinematics
    """
    def fkine(self,joints_config:list):
        # supports both single joints config or joints trajectory
        ees_pose = []
        joints_config = np.array(joints_config)
        if joints_config.ndim == 1:
            joints_config = joints_config[np.newaxis, :]

        for i, robot in enumerate(self.model): 
            posture = joints_config[:,:(robot.n)] 
            ees_pose.append(robot.fkine(np.deg2rad(posture)))
        return ees_pose
    def fkine_all(self,joints_config:list):
        joints_pose = self.model[0].fkine_all(np.deg2rad(np.array(joints_config))) # this has all the frames of the robot joints
        return joints_pose
    def emu_fkine(self,joints_config:list):
        # for single joints configuration atm
        joints_pose,ees_pose = self.fkine_all(joints_config),self.fkine(joints_config)

        return joints_pose,ees_pose


if __name__ == "__main__":
    #Define ISB rtb arm model
    emu_model = emu_robot()
    print(emu_model.model)

    """
    Sanity check your UA model here
    """
    ub_postures_to_test = [[0,0,0,0]]

    for joints_config in tqdm(ub_postures_to_test):
        joints_pose, ee_pose = emu_model.emu_fkine(joints_config)
        emu_model.model[0].plot(q=np.deg2rad(np.array(joints_config)))
        emu_model.plot_joints_ees_frame(joints_pose,ee_pose)

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

        # joints_pose, ee_pose = ub_xsens_left.emu_fkine(joints_config)
        # ub_xsens_left.plot_joints_ees_frame(joints_pose,ee_pose)
        plt.show(block=True)
        
        
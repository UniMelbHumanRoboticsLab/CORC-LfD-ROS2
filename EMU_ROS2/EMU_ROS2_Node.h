/**
 * \file EMU_ROS2_Node.h
 * \author Jia Quan Loh
 * \version 0.1
 * \date 2022-10-24
 * \copyright Copyright (c) 2022
 * \brief An example ROS2 node that also holds a reference to the robot object.
 */
#ifndef EMU_ROS2_Node_H
#define EMU_ROS2_Node_H

#include "RobotM3.h"
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

using std::placeholders::_1;

class EMU_ROS2_Node : public rclcpp::Node
{
public:
    EMU_ROS2_Node(const std::string &name, RobotM3 *robot);

    void lfd_traj_callback(const sensor_msgs::msg::JointState::SharedPtr  msg);
    void publish_joint_states();

    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr get_interface();
    bool return_lfd_ready() { return lfd_ready;}
    bool lfd_ready;
    std::vector<Eigen::VectorXd> lfd_trajectory;

private:
    RobotM3 *m_Robot;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_Sub;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr m_Pub;
    
};

#endif//EMU_ROS2_Node_H
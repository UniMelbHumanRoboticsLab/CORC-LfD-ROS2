/**
 * \file FT_EMU_ROS2_Node.h
 * \author Jia Quan Loh
 * \version 0.1
 * \date 2022-10-24
 * \copyright Copyright (c) 2022
 * \brief An example ROS2 node that also holds a reference to the robot object.
 */
#ifndef FT_EMU_ROS2_Node_H
#define FT_EMU_ROS2_Node_H

#include "FT_RobotM3.h"
#include <string>
#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int16.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include "corc_lfd_interfaces/msg/task_demo.hpp"

using std::placeholders::_1;

class FT_EMU_ROS2_Node : public rclcpp::Node
{
public:
    FT_EMU_ROS2_Node(const std::string &name, FT_RobotM3 *robot);

    void wrench_callback(const sensor_msgs::msg::JointState::SharedPtr  msg);
    void publish_task_dynamics(uint16_t sbmvmtNum);
    void publish_state_id(uint16_t id);

    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr get_interface();

private:
    FT_RobotM3 *m_Robot;
    
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr wrench_sub;
    rclcpp::Publisher<corc_lfd_interfaces::msg::TaskDemo>::SharedPtr task_pub;
    rclcpp::Publisher<std_msgs::msg::UInt16>::SharedPtr state_pub;
};

#endif//FT_EMU_ROS2_Node_H
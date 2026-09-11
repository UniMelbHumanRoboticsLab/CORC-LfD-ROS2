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
#include "corc_interfaces/msg/task_demo.hpp"
#include "corc_interfaces/msg/session_param.hpp"
#include "corc_interfaces/msg/state_var.hpp"
#include "corc_interfaces/msg/desired_wrench.hpp"

using std::placeholders::_1;

class FT_EMU_ROS2_Node : public rclcpp::Node
{
public:
    FT_EMU_ROS2_Node(const std::string &name, FT_RobotM3 *robot); 
    
    void wrench_callback(const corc_interfaces::msg::DesiredWrench::SharedPtr  msg);
    void session_param_callback(const corc_interfaces::msg::SessionParam::SharedPtr  msg);
    
    void publish_task_dynamics(uint16_t repNum,uint16_t sbmvmtNum);
    void publish_state_var(std::string id,uint16_t varNum);
    
    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr get_interface();
    std::vector<std::string> get_session_param();
    std::vector<uint16_t> get_session_param_ints();
    Eigen::VectorXd get_desired_wrench();
    void reset_desired_wrench();

private:
    FT_RobotM3 *m_Robot;
    
    rclcpp::Subscription<corc_interfaces::msg::SessionParam>::SharedPtr ses_param_sub;
    rclcpp::Subscription<corc_interfaces::msg::DesiredWrench>::SharedPtr wrench_sub;
    
    rclcpp::Publisher<corc_interfaces::msg::TaskDemo>::SharedPtr task_pub;
    rclcpp::Publisher<corc_interfaces::msg::StateVar>::SharedPtr state_pub;
    
    // session params
    std::string expID = "unknown";
    std::string patientID = "unknown";
    std::string subID = "unknown";
    uint16_t maxSbmvmt = 0;
    std::string learnOutput = "unknown";
    
    // learning status (1st-learning finish?, 2nd-nth leanring succeed?)
     std::vector<uint16_t> learnStatus;
     
     // desired commands from LfD
     Eigen::VectorXd desired_wrench;
    

};

#endif//FT_EMU_ROS2_Node_H4
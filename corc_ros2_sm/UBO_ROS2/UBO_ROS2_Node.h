/**
 * \file UBO_ROS2_Node.h
 * \author Jia Quan Loh
 * \version 0.1
 * \date 2022-10-24
 * \copyright Copyright (c) 2022
 * \brief An example ROS2 node that also holds a reference to the robot object.
 */
#ifndef UBO_ROS2_Node_H
#define UBO_ROS2_Node_H

#include "UBORobot.h"
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

using std::placeholders::_1;

class UBO_ROS2_Node : public rclcpp::Node
{
public:
    UBO_ROS2_Node(const std::string &name, UBORobot *robot);

    void publish_state(std::string status);
    void publish_wrenches(Eigen::VectorXd wrenches,Eigen::VectorXd olStatus);

    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr get_interface();


private:
    UBORobot *m_Robot;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench1_pub;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench2_pub;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench3_pub;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr rft_stats;
    
};

#endif//UBO_ROS2_Node_H
#include "EMU_ROS2_Node.h"

EMU_ROS2_Node::EMU_ROS2_Node(const std::string &name, RobotM3 *robot)
    : Node(name), m_Robot(robot)
{
    // Create an example subscription
    m_Sub = create_subscription<sensor_msgs::msg::JointState>(
        "lfd_traj", 10, std::bind(&EMU_ROS2_Node::lfd_traj_callback, this, _1)
    );
    // Create a joint state publisher
    m_Pub = create_publisher<sensor_msgs::msg::JointState>(
        "demo_traj", 10
    );
    lfd_ready = false;
}

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr
EMU_ROS2_Node::get_interface()
{
    // Must have this method to return base interface for spinning
    return this->get_node_base_interface();
}

void
EMU_ROS2_Node::lfd_traj_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    // Callback to receive Learned Trajectory from LfD Node
    // Convert std::vector to Eigen::VectorXd
    Eigen::VectorXd lfd_position = Eigen::Map<Eigen::VectorXd>(msg->position.data(), msg->position.size());
    lfd_trajectory.push_back(lfd_position);

    std::cout << std::setprecision(3) << std::fixed << std::showpos;
    std::cout << "X=[ " << lfd_position.transpose() <<" ]\t";
    std::cout <<  std::endl;
    std::cout <<  std::noshowpos;
}

void
EMU_ROS2_Node::publish_joint_states()
{
    // Instantiate joint state message
    // spdlog::debug("publishing");
    sensor_msgs::msg::JointState msg;

    // Assign current header time stamp
    msg.header.stamp = this->now();

    // Use this naming scheme for robot state publisher to recognise
    msg.name = {
        "left_hip_joint",
        "left_knee_joint",
        "right_hip_joint",
        "right_knee_joint",
        "world_to_backpack"
    };

    // Copy position, velocity and toroque from M3
    msg.position.assign(
        m_Robot->getEndEffPosition().data(),
        m_Robot->getEndEffPosition().data() + m_Robot->getEndEffPosition().size()
    );

    msg.velocity.assign(
        m_Robot->getEndEffVelocity().data(),
        m_Robot->getEndEffVelocity().data() + m_Robot->getEndEffVelocity().size()
    );
    msg.effort.assign(
        m_Robot->getTorque().data(),
        m_Robot->getTorque().data() + m_Robot->getTorque().size()
    );

    // Publish the joint state message
    m_Pub->publish(msg);
}

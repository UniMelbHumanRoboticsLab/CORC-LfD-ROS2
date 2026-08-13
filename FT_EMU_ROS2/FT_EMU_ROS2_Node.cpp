#include "FT_EMU_ROS2_Node.h"

FT_EMU_ROS2_Node::FT_EMU_ROS2_Node(const std::string &name, FT_RobotM3 *robot)
    : Node(name), m_Robot(robot)
{
    // Create an example subscription
    m_Sub = create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", 10, std::bind(&FT_EMU_ROS2_Node::humanJA_callback, this, _1)
    );
    // Create a joint state publisher
    m_Pub = create_publisher<sensor_msgs::msg::JointState>(
        "demo_traj", 10
    );
    lfd_ready = false;
}

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr
FT_EMU_ROS2_Node::get_interface()
{
    // Must have this method to return base interface for spinning
    return this->get_node_base_interface();
}

void
FT_EMU_ROS2_Node::humanJA_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    // Callback to receive human joints from XSENS
    // Map joint names to indices
    std::map<std::string, size_t> joint_map;
    for (size_t i = 0; i < msg->name.size(); ++i) {
        joint_map[msg->name[i]] = i;
    }
    
    // Desired joints in order
    std::vector<std::string> right_joints = {
        "skeleton_pelvis_t8_NA_y", "skeleton_pelvis_t8_NA_x", "skeleton_pelvis_t8_NA_z",
        "skeleton_right_c7_shoulder_z", "skeleton_right_c7_shoulder_y",
        "skeleton_right_shoulder_z", "skeleton_right_shoulder_x", "skeleton_right_shoulder_y",
        "skeleton_right_elbow_z", "skeleton_right_elbow_y",
        "skeleton_right_wrist_z", "skeleton_right_wrist_x"
    };
    
    // Extract values
    Eigen::VectorXd right_joint_pos(right_joints.size());
    for (size_t i = 0; i < right_joints.size(); ++i) {
        right_joint_pos(i) = msg->position[joint_map[right_joints[i]]];
    }
    
    std::vector<std::string> left_joints = {
        "skeleton_pelvis_t8_NA_y", "skeleton_pelvis_t8_NA_x", "skeleton_pelvis_t8_NA_z",
        "skeleton_left_c7_shoulder_z", "skeleton_left_c7_shoulder_y",
        "skeleton_left_shoulder_z", "skeleton_left_shoulder_x", "skeleton_left_shoulder_y",
        "skeleton_left_elbow_z", "skeleton_left_elbow_y",
        "skeleton_left_wrist_z", "skeleton_left_wrist_x"
    };
    
    // Extract values
    Eigen::VectorXd left_joint_pos(left_joints.size());
    for (size_t i = 0; i < left_joints.size(); ++i) {
        left_joint_pos(i) = msg->position[joint_map[left_joints[i]]];
    }

    std::cout << std::setprecision(3) << std::fixed << std::showpos;
    std::cout << "q_right\t=[ " << right_joint_pos.transpose() <<" ]\t";
    std::cout <<  std::endl;
    std::cout << "q_left\t=[ " << left_joint_pos.transpose() <<" ]\t";
    std::cout <<  std::endl;
    std::cout <<  std::noshowpos;
}

void
FT_EMU_ROS2_Node::publish_joint_states()
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

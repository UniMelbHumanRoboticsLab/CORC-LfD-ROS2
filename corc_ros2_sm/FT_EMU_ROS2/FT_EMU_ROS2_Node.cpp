#include "FT_EMU_ROS2_Node.h"

FT_EMU_ROS2_Node::FT_EMU_ROS2_Node(const std::string &name, FT_RobotM3 *robot)
    : Node(name), m_Robot(robot)
{
    // match xsens QOS profile
    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable();
    qos.transient_local();
    
    // Create an session parameter subscriber ande desired wrench subscriber
    ses_param_sub = create_subscription<corc_interfaces::msg::SessionParam>(
        "session_param", qos, std::bind(&FT_EMU_ROS2_Node::session_param_callback, this, _1)
    );
    wrench_sub = create_subscription<corc_interfaces::msg::DesiredWrench>(
        "desired_wrench", 10, std::bind(&FT_EMU_ROS2_Node::wrench_callback, this, _1)
    );
    // Create a task dynamics publisher
    task_pub = create_publisher<corc_interfaces::msg::TaskDemo>(
        "emu_task_dynamics", qos
    );
    state_pub = create_publisher<corc_interfaces::msg::StateVar>(
        "state_var", qos
    );
}

void
FT_EMU_ROS2_Node::session_param_callback(const corc_interfaces::msg::SessionParam::SharedPtr msg){
    expID = msg->exp_id;
    patientID = msg->patient_id;
    subID = msg->sub_id;
    learnOutput = msg->learn_output;
    maxSbmvmt = msg->max_sbmvmt;
    learnStatus = msg->learn_status;
}

void
FT_EMU_ROS2_Node::wrench_callback(const corc_interfaces::msg::DesiredWrench::SharedPtr msg)
{
//     // Callback to receive human joints from XSENS
//     // Map joint names to indices
//     std::map<std::string, size_t> joint_map;
//     for (size_t i = 0; i < msg->name.size(); ++i) {
//         joint_map[msg->name[i]] = i;
//     }
//     
//     std::vector<std::string> left_joints = {
//         "skeleton_pelvis_t8_NA_y", "skeleton_pelvis_t8_NA_x", "skeleton_pelvis_t8_NA_z",
//         "skeleton_left_c7_shoulder_z", "skeleton_left_c7_shoulder_y",
//         "skeleton_left_shoulder_z", "skeleton_left_shoulder_x", "skeleton_left_shoulder_y",
//         "skeleton_left_elbow_z", "skeleton_left_elbow_y",
//         "skeleton_left_wrist_z", "skeleton_left_wrist_x"
//     };
//     
//     // Extract values
//     VX left_joint_pos(left_joints.size());
//     for (size_t i = 0; i < left_joints.size(); ++i) {
//         left_joint_pos(i) = msg->position[joint_map[left_joints[i]]];
//     }
//     
    // Extract values
    this->desired_wrench = Eigen::Map<VM3>(
                            msg->desired_wrench.data(), 
                            msg->desired_wrench.size()
                            );  

//     std::cout << std::setprecision(3) << std::fixed << std::showpos;
//     std::cout << "desired_wrench\t=[ " << desired_wrench.transpose() <<" ]\t";
//     std::cout <<  std::endl;
//     std::cout <<  std::noshowpos;
}


void FT_EMU_ROS2_Node::publish_task_dynamics(uint16_t repNum,uint16_t sbmvmtNum,VM3 commanded_wrench)
{
    // Instantiate joint state message
    corc_interfaces::msg::TaskDemo msg;

    // Assign current header time stamp
    msg.header.stamp = this->now();

    msg.rep_num = repNum;
    msg.sbmvmt_num = sbmvmtNum;

    msg.name = {
        "x", "y", "z",           // Position (Cartesian)
        "xd", "yd", "zd",        // Velocity (Cartesian)
        "q1", "q2", "q3",           // Position (Joint)
        "q1d", "q2d", "q3d",           // Position (Joint)
        "Fx", "Fy", "Fz",        // Forces
        "Mx", "My", "Mz"         // Moments/Torques
        "Fx_int", "Fy_int", "Fz_int",        // Forces
    };
    
    // Copy position, velocity and torque from M3
    msg.position.assign(
        m_Robot->getEndEffPosition().data(),
        m_Robot->getEndEffPosition().data() + m_Robot->getEndEffPosition().size()
    );
    msg.velocity.assign(
        m_Robot->getEndEffVelocityFiltered().data(),
        m_Robot->getEndEffVelocityFiltered().data() + m_Robot->getEndEffVelocityFiltered().size()
    );
    msg.joint_position.assign(
        m_Robot->getPosition().data(),
        m_Robot->getPosition().data() + m_Robot->getPosition().size()
    );
    msg.joint_velocity.assign(
        m_Robot->getVelocity().data(),
        m_Robot->getVelocity().data() + m_Robot->getVelocity().size()
    );
    msg.wrench_m.assign(
        m_Robot->getWrenches().data(),
        m_Robot->getWrenches().data() + m_Robot->getWrenches().size()
    );
    msg.wrench_int.assign(
        m_Robot->getInteractionForce().data(),
        m_Robot->getInteractionForce().data() + m_Robot->getInteractionForce().size()
    );
    msg.wrench_com.assign(
        commanded_wrench.data(),
        commanded_wrench.data() + commanded_wrench.size()
    );

    // Publish the joint state message
    task_pub->publish(msg);
}

void FT_EMU_ROS2_Node::publish_state_var(std::string state_id,uint16_t var_num)
{
    // Instantiate UInt16 message
    corc_interfaces::msg::StateVar msg;
    msg.state_id = state_id;
    msg.var_num = var_num;
    
    // Publish the state message
    state_pub->publish(msg);
}

// getter functions for the private variables
rclcpp::node_interfaces::NodeBaseInterface::SharedPtr
FT_EMU_ROS2_Node::get_interface()
{
    // Must have this method to return base interface for spinning
    return get_node_base_interface();
}

std::vector<std::string>
FT_EMU_ROS2_Node::get_session_param()
{
    return {expID, patientID, subID,learnOutput};
}

std::vector<uint16_t>
FT_EMU_ROS2_Node::get_session_param_ints()
{
    std::vector<uint16_t> session_param_ints = {maxSbmvmt};
    session_param_ints.insert(session_param_ints.end(), learnStatus.begin(), learnStatus.end());
    return session_param_ints;
}

VM3 
FT_EMU_ROS2_Node::get_desired_wrench()
{
    return desired_wrench;
}

void 
FT_EMU_ROS2_Node::reset_desired_wrench()
{
    desired_wrench = VM3::Zero();
}

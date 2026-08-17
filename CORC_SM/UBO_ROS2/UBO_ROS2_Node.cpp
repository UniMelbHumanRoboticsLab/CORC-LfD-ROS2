#include "UBO_ROS2_Node.h"

UBO_ROS2_Node::UBO_ROS2_Node(const std::string &name, UBORobot *robot)
    : Node(name), m_Robot(robot)
{
    // Create an example subscription
    // m_Sub = create_subscription<sensor_msgs::msg::JointState>(
    //     "lfd_traj", 1000, std::bind(&UBO_ROS2_Node::lfd_traj_callback, this, _1)
    // );

    // Create a status publisher
    status_pub = create_publisher<std_msgs::msg::String>(
        "rbt_state", 500    );
    wrench1_pub = create_publisher<geometry_msgs::msg::WrenchStamped>(
        "rft1_wrench", 500    );
    wrench2_pub = create_publisher<geometry_msgs::msg::WrenchStamped>(
        "rft2_wrench", 500    );
    wrench3_pub = create_publisher<geometry_msgs::msg::WrenchStamped>(
        "rft3_wrench", 500    );
    rft_stats = create_publisher<geometry_msgs::msg::Vector3>(
        "rft_stats", 500    );

}

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr
UBO_ROS2_Node::get_interface()
{
    // Must have this method to return base interface for spinning
    return this->get_node_base_interface();
}

void
UBO_ROS2_Node::publish_state(std::string status)
{
    // Instantiate joint state message
    // spdlog::debug("publishing");
    std_msgs::msg::String msg;

    // Assign current header time stamp
    msg.data = status;

    // Publish the joint state message
    status_pub->publish(msg);
}

void
UBO_ROS2_Node::publish_wrenches(Eigen::VectorXd wrenches,Eigen::VectorXd olStatus)
{
    // Instantiate joint state message
    // spdlog::debug("publishing");
    int num_rft = wrenches.size()/6;
    geometry_msgs::msg::Vector3 msg2;

    for (int i=0; i< num_rft; i++)
    {

        geometry_msgs::msg::WrenchStamped msg;

        msg.header.stamp = this->now();
        msg.header.frame_id= std::to_string(olStatus[i]);

        msg.wrench.force.x = wrenches[i*6+0];
        msg.wrench.force.y = wrenches[i*6+1];
        msg.wrench.force.z = wrenches[i*6+2];
        msg.wrench.torque.x = wrenches[i*6+3];
        msg.wrench.torque.y = wrenches[i*6+4];
        msg.wrench.torque.z = wrenches[i*6+5];

        if (i == 0)
        {
            wrench1_pub->publish(msg);
            msg2.x = olStatus[i];
        }
        else if (i == 1)
        {
            wrench2_pub->publish(msg);
            msg2.y = olStatus[i];
        }
        else if (i == 2)
        {
            wrench3_pub->publish(msg);
            msg2.z = olStatus[i];
        }
    }
    

    rft_stats->publish(msg2);
    
    
}




/**
 * \file FT_EMU_ROS2.h
 * \author Jia Quan Loh
 * \brief The FT_EMU_ROS2 class is a state machine
 * \date 2024-05-20
 *
 * \copyright Copyright (c) 2024
 *
 */
#ifndef FT_M3_SM_H
#define FT_M3_SM_H

#include <cstdint>
#include "StateMachine.h"
#include "FT_RobotM3.h"
#include "FLNLHelper.h"

// State Classes
#include "FT_EMU_ROS2_States.h"
#include "FT_EMU_ROS2_Node.h"

/**
 * @brief Example implementation of a StateMachine for the M3Robot class. States should implemented M3DemoState
 *
 */
class FT_EMU_ROS2 : public StateMachine {

   public:
        FT_EMU_ROS2(int argc, char **argv) ;
        ~FT_EMU_ROS2();
        void init();
        void end();
        
        void hwStateUpdate();
        
        FT_RobotM3 *robot() { return static_cast<FT_RobotM3*>(_robot.get()); } //!< Robot getter with specialised type (lifetime is managed by Base StateMachine)
        const std::shared_ptr<FT_EMU_ROS2_Node> &get_node(){ return m_Node;} // get the ros 2 node to access ROS2 functions and attributes
        std::shared_ptr<FLNLHelper> UIserver = nullptr;     //!< Pointer to communication server
        
        double MassComp = 0;         //!< Mass comp value used for standard operations
        bool robotVerbose = false;
        uint16_t stateID = 0;
        unint16_t sbmvmtNum = 0;

    private:
        std::shared_ptr<FT_EMU_ROS2_Node> m_Node;
        
};

#endif /*FT_M3_SM_H*/

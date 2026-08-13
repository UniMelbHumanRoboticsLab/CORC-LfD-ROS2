/**
 * \file EMU_ROS2.h
 * \author Jia Quan Loh
 * \brief The EMU_ROS2 class is a state machine aims to evaluate deweighting algorithms on EMU device
 * \date 2024-05-20
 *
 * \copyright Copyright (c) 2024
 *
 */
#ifndef M3_SM_H
#define M3_SM_H


#include "StateMachine.h"
#include "RobotM3.h"
#include "FLNLHelper.h"

// State Classes
#include "EMU_ROS2_States.h"
#include "EMU_ROS2_Node.h"

/**
 * @brief Example implementation of a StateMachine for the M3Robot class. States should implemented M3DemoState
 *
 */
class EMU_ROS2 : public StateMachine {

   public:
    EMU_ROS2(int argc, char **argv) ;
    ~EMU_ROS2();
    void init();
    void end();

    void hwStateUpdate();

    RobotM3 *robot() { return static_cast<RobotM3*>(_robot.get()); } //!< Robot getter with specialised type (lifetime is managed by Base StateMachine)
    const std::shared_ptr<EMU_ROS2_Node> &get_node(){ return m_Node;}
    std::shared_ptr<FLNLHelper> UIserver = nullptr;     //!< Pointer to communication server

    //TODO: place in struct and pass to states (instead of whole state machine)
    double Command = 0;         //!< Command (state) currently applied
    double MvtProgress = 0;     //!< Progress (status) along mvt
    double Contribution = 0;    //!< User contribution to mvt
    double MassComp =0;         //!< Mass comp value used for standard operations
    Deweight_s DwData;

    private:
        std::shared_ptr<EMU_ROS2_Node> m_Node;
};

#endif /*M3_SM_H*/

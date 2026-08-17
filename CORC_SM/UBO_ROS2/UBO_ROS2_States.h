/**
 * \file UBO_ROS2_States.h
 * \author Jia Quan Loh
 * \brief The UBO_ROS2 State class is a compilation of states for UBO state machine
 * \date 2025-10-29
 *
 * \copyright Copyright (c) 2024
 *
 */

#ifndef UBOSTATES_H_DEF
#define UBOSTATES_H_DEF

#include "State.h"
#include "UBORobot.h"
#include "LogHelper.h"
#include "FLNLHelper.h"

#define NUM_CALIBRATE_READINGS 1000
class UBO_ROS2; // declare empty class for UBO_ROS2 state machine for forward inclusion

/**
 * \brief Generic state type to be used for UBO_ROS2 StateMachine, providing running time and iterations number
 *
 */
class UBO_ROS2_State : public State {
    protected:
        UBORobot * robot;                               //!< Pointer to state machines robot object

        UBO_ROS2_State(UBORobot* ubo, UBO_ROS2 *sm_, const char *name_ = NULL): State(name_), robot(ubo), sm(sm_),state_name(name_){spdlog::debug("Created UBO_ROS2_State {}", name_);};
    private:
        void entry(void) final;
        void during(void) final;
        void exit(void) final;

    public:
        virtual void entryCode(){};
        virtual void duringCode(){};
        virtual void exitCode(){};

    protected:
        int entry_num = 0;
        std::string state_name;
        UBO_ROS2 *sm;
        LogHelper stateLogger;
};


/**
 * \brief UBO initialization
 *
 */
class UBOInitState : public UBO_ROS2_State {
    public:
        UBOInitState(UBORobot * ubo, UBO_ROS2 *sm, const char *name = "UBO Init"):UBO_ROS2_State(ubo, sm, name){};

        void entryCode(void) {spdlog::info("InitState Entry");robot->setUBO_FTSensorsFilter();}
        void duringCode(void) {}
        void exitCode(void) {spdlog::info("InitState Exit");}
};

/**
 * \brief UBO calibration
 *
 */
class UBOCalibState : public UBO_ROS2_State {
    public:
        UBOCalibState(UBORobot * ubo, UBO_ROS2 *sm, const char *name = "UBO Calibrate"):UBO_ROS2_State(ubo, sm, name){};

        void entryCode(void);
        void duringCode(void);
        void exitCode(void);

        bool isCalibDone() {return calibDone;}

    private:
        Eigen::ArrayXXd readings;
        bool calibDone=false;
        int currReading = 0;
};

/**
 * \brief UBO idle
 *
 */
class UBOIdleState : public UBO_ROS2_State {
    public:
        UBOIdleState(UBORobot * ubo, UBO_ROS2 *sm, const char *name = "UBO Idle"):UBO_ROS2_State(ubo, sm, name){};

        void entryCode(void);
        void duringCode(void);
        void exitCode(void);
};

/**
 * \brief UBO record
 *
 */
class UBORecordState : public UBO_ROS2_State {
    private:
        Eigen::VectorXd lastRFTReadings;
        int ticker = 0;

    public:
        UBORecordState(UBORobot * ubo, UBO_ROS2 *sm, const char *name = "UBO Record"):UBO_ROS2_State(ubo, sm, name){};

        void entryCode(void);
        void duringCode(void);
        void exitCode(void);
};


#endif

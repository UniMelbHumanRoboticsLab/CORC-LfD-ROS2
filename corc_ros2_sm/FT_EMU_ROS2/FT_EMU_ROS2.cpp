#include "FT_EMU_ROS2.h"

using namespace std;

/**
 * \Calibration transitions
 *
 */
bool goToCalib(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //only transition when session parameters is confirmed
    if ( sm.robot()->keyboard->getKeyUC()=='C' )
        if (sm.expID != "unknown" and sm.patientID != "unknown" and sm.subID != "unknown" and sm.maxSbmvmt != 0) 
            return true;
        else{
            spdlog::warn("Restart Session ID node");
        }

    //Otherwise false
    return false;
}
bool initGoToCalib(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    if (sm.expID != "unknown" and sm.patientID != "unknown" and sm.subID != "unknown" and sm.maxSbmvmt != 0) 
        return true;

    //Otherwise false
    return false;
}
bool endM3Calib(StateMachine & sm) {
    return (sm.state<M3CalibState>("CalibState"))->isCalibDone();
}
bool endFTCalib(StateMachine & sm) {
    return (sm.state<M3FTCalibState>("FTCalibState"))->isCalibDone();
}

/**
 * \transitions between standby
 *
 */
bool goToLock(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //keyboard press
    if (sm.robot()->keyboard->getKeyUC()=='P' )
        return true;

    //Otherwise false
    return false;
}
bool goToReset(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //keyboard press
    if ( sm.robot()->keyboard->getKeyUC()=='R' )
        return true;

    //Otherwise false
    return false;
}
bool goToStandby(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    if ( sm.robot()->keyboard->getKeyUC()=='S' )
        return true;

    return false;
}
bool goToDemo(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    if ( sm.robot()->keyboard->getKeyUC()=='D' or sm.robot()->joystick->isButtonTransition(1)>0)
        return true;

    return false;
}
bool goToLearn(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //keyboard press
    if (sm.robot()->keyboard->getKeyUC()=='L' )
        return true;

    //Otherwise false
    return false;
}
bool finishLearn(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    if (sm.learn_finish) 
        return true;

    //Otherwise false
    return false;
}
bool goToNewTP(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    if (sm.robot()->keyboard->getKeyUC()=='N' or sm.robot()->joystick->isButtonTransition(3)>0) 
    {
        if (sm.learn_success[sm.varNum-1])
        {
            return true;
        }
        else
        {
            spdlog::warn("Have not learned for Var {}",sm.varNum);
        }
    }


    //Otherwise false
    return false;
}
bool goToRepro(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    if (sm.robot()->joystick->isButtonTransition(1)>0){
        if (sm.lfd_repro_ready){
            return true;
        }
        else{
            spdlog::warn("TPs not enough");
        }
    }

    return false;
}
/**
 * \quitting transitions
 *
 */
bool quit(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    if ( sm.robot()->keyboard->getKeyUC()=='Q' ) {
        std::raise(SIGTERM); //Clean exit
        return true;
    }

    return false;
}

//Fake transition (return false all the time) used to update the mass parameter
bool updateMass(StateMachine & SM) {
    FT_EMU_ROS2 & sm = static_cast<FT_EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    std::vector<double> params;

    return false;
}

FT_EMU_ROS2::FT_EMU_ROS2(int argc, char **argv)  {
    //Create a Robot and set it to generic state machine
    setRobot(std::make_unique<FT_RobotM3>("EMU_FOURIER", "M3_params.yaml"));
    // setRobot(std::make_unique<FT_RobotM3>("EMU_MELB", "M3_params.yaml"));

    // Configure ROS2 initialisation options and disable SIGINT capture (handled by CORC)
    rclcpp::InitOptions ros_init = rclcpp::InitOptions();
    ros_init.shutdown_on_signal = false;
    rclcpp::init(argc, argv, ros_init);

    // Create the ROS2 node and pass a reference to the X2 Robot object
    m_Node = std::make_shared<FT_EMU_ROS2_Node>("FT_EMU", robot());

    //Create state instances and add to the State Machine
    addState("InitState", std::make_shared<M3InitState>(robot(), this));
    addState("ResetState", std::make_shared<M3InitState>(robot(), this));
    addState("CalibState", std::make_shared<M3CalibState>(robot(), this));
    addState("FTCalibState", std::make_shared<M3FTCalibState>(robot(), this));
    addState("StandbyState", std::make_shared<M3StandbyState>(robot(), this));
    addState("LockState", std::make_shared<M3LockState>(robot(), this));
    addState("DemoState", std::make_shared<M3DemoState>(robot(), this));
    addState("LearnState", std::make_shared<M3LearnState>(robot(), this));
    addState("NewTPState", std::make_shared<M3NewTPState>(robot(), this));
    addState("ReproLfDState", std::make_shared<M3ReproLfDState>(robot(), this));
    
    //state machine initialisation and quitting 
    setInitState("InitState");
    addTransitionFromAny(&quit, "InitState");
    addTransition("InitState", &quit, "InitState"); //From any does not apply to self (destination state)

    //Define transitions between functional states
    // initial transitions to always go through
    addTransition("InitState", &initGoToCalib, "CalibState");
    addTransition("CalibState", &endM3Calib, "FTCalibState");
    addTransition("FTCalibState", &endFTCalib, "StandbyState");
    addTransition("StandbyState", &updateMass, "StandbyState"); //Fake transition never returning true
    
    // transtions between standby
    addTransition("StandbyState", &goToCalib, "FTCalibState");
    addTransition("StandbyState", &goToReset, "ResetState");
    addTransition("ResetState", &goToStandby, "StandbyState");
    
    addTransition("StandbyState", &goToLock, "LockState");
    addTransition("LockState", &goToLock, "StandbyState");
    
    addTransition("StandbyState", &goToDemo, "DemoState");
    addTransition("DemoState", &goToStandby, "StandbyState");
    
    addTransition("StandbyState", &goToLearn, "LearnState");
    addTransition("LearnState", &finishLearn, "StandbyState");
    
    addTransition("StandbyState", &goToNewTP, "NewTPState");
    addTransition("NewTPState", &goToRepro, "ReproLfDState");
    addTransition("ReproLfDState", &goToStandby, "StandbyState");
}

FT_EMU_ROS2::~FT_EMU_ROS2() {
}

/**
 * \brief start function for running any designed statemachine specific functions
 * for example initialising robot objects.
 *
 */
void FT_EMU_ROS2::init() {
    spdlog::debug("FT_EMU_ROS2::init()");
    if(robot()->initialise()) {
        logHelper.initLogger("FT_EMU_ROS2_Log", "logs/corc_recordings/FT_EMU/exp2/FT_EMU_ROS2.csv", LogFormat::CSV, true);
        logHelper.add(runningTime(), "Time (s)");
        logHelper.add(robot()->getEndEffPosition(), "X");
        logHelper.add(robot()->getEndEffVelocity(), "dX");
        logHelper.add(robot()->getInteractionForce(), "F");
        logHelper.add(robot()->getEndEffAcceleration(), "ddX");
        logHelper.add(robot()->getEndEffVelocityFiltered(), "dXFilt");
        #ifdef NOROBOT
            UIserver = std::make_shared<FLNLHelper>(*robot(), "127.0.0.1");
            // UIserver = std::make_shared<FLNLHelper>(*robot(), "192.168.7.2");
        #else
            UIserver = std::make_shared<FLNLHelper>(*robot(), "127.0.0.1");
            // UIserver = std::make_shared<FLNLHelper>(*robot(), "192.168.7.2");
        #endif // NOROBOT
    }
    else {
        spdlog::critical("Failed robot initialisation. Exiting...");
        std::raise(SIGTERM); //Clean exit
    }
}

void FT_EMU_ROS2::end() {
    if(running())
        UIserver->closeConnection();
    FT_EMU_ROS2::stateID = "quit";
    FT_EMU_ROS2::get_node()->publish_state_var(FT_EMU_ROS2::stateID,0);
    StateMachine::end();
}


/**
 * \brief Statemachine to hardware interface method. Run any hardware update methods
 * that need to run every program loop update cycle.
 *
 */
void FT_EMU_ROS2::hwStateUpdate() {
    StateMachine::hwStateUpdate();
    //Also send robot state over network
    UIserver->sendState();
    //Attempt to reconnect (if not already waiting for connection)
    UIserver->reconnect();
    // Allow for the ROS2 node to execute callbacks (e.g., subscriptions)
    rclcpp::spin_some(get_node()->get_interface());
}



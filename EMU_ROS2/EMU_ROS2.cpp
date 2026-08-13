#include "EMU_ROS2.h"

using namespace std;

bool goToCalib(StateMachine & SM) {
    EMU_ROS2 & sm = static_cast<EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //keyboard or joystick press
    if ( sm.robot()->keyboard->getKeyUC()=='C' )
        return true;

    //Check incoming command requesting state change
    if ( sm.UIserver->isCmd("GOCA") ) {
        sm.UIserver->sendCmd(string("OKCA"));
        spdlog::debug("goToCalib");
        return true;
    }

    //Otherwise false
    return false;
}

bool endCalib(StateMachine & sm) {
    return (sm.state<M3CalibState>("CalibState"))->isCalibDone();
}

bool goToLock(StateMachine & SM) {
    EMU_ROS2 & sm = static_cast<EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //keyboard press
    if ( sm.robot()->joystick->isButtonTransition(3)>0 || sm.robot()->keyboard->getKeyUC()=='L' )
        return true;

    //Check incoming command requesting state change
    if ( sm.UIserver->isCmd("GOLO") ) {
        sm.UIserver->sendCmd(string("OKLO"));
        spdlog::debug("goToLock");
        return true;
    }

    //Otherwise false
    return false;
}

bool goToUnlock(StateMachine & SM) {
    EMU_ROS2 & sm = static_cast<EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //keyboard press
    if ( sm.robot()->keyboard->getKeyUC()=='U' )
        return true;

    //Check incoming command requesting state change
    if ( sm.UIserver->isCmd("GOUN") ) {
        sm.UIserver->sendCmd(string("OKUN"));
        spdlog::debug("goToUnlock");
        return true;
    }

    //Otherwise false
    return false;
}

bool goToReset(StateMachine & SM) {
    EMU_ROS2 & sm = static_cast<EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //keyboard press
    if ( sm.robot()->keyboard->getKeyUC()=='R' )
        return true;

    //Check incoming command requesting state change
    if ( sm.UIserver->isCmd("GORE") ) {
        sm.UIserver->sendCmd(string("OKRE"));
        spdlog::debug("goToReset");
        return true;
    }

    //Otherwise false
    return false;
}

bool goToReproduce(StateMachine & SM) {
    EMU_ROS2 & sm = static_cast<EMU_ROS2 &>(SM); //Cast to specific StateMachine type
    std::shared_ptr<EMU_ROS2_Node> node = sm.get_node(); //Cast to specific ROS2Node  type

    //keyboard press
    if ( sm.robot()->keyboard->getKeyUC()=='D' )
    {
        if (node->lfd_ready)
        {
            spdlog::debug("LFD Generalization is ready. Starting reproduction");
            return true;
        }
        else
        {
            spdlog::debug("LFD Generalization is not ready");
        }
    }

    //Check incoming command requesting state change
    if ( sm.UIserver->isCmd("GORE") ) {
        sm.UIserver->sendCmd(string("OKRE"));
        spdlog::debug("goToReset");
        return true;
    }

    //Otherwise false
    return false;
}

bool goToGravity(StateMachine & SM) {
    EMU_ROS2 & sm = static_cast<EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //TODO: differentiate (for logging) from standby state
    std::vector<double> params;
    if ( sm.UIserver->isCmd("GOGR", params) ) {
        if(params.size() == 1) {
            std::shared_ptr<M3StandbyPublishState> s = sm.state<M3StandbyPublishState>("StandbyState");
            s->setMass(params[0]);
            sm.UIserver->sendCmd(string("OKGR"));
            spdlog::debug("goToGravity");
            return true;
        }
        else {
            spdlog::warn("goToGravity: Error: number of command parameters.");
            sm.UIserver->sendCmd(string("ERGR"));
            return false;
        }
    }
    else {
        if ( sm.robot()->keyboard->getKeyUC()=='S' )
            return true;
    }

    return false;
}

//Exit CORC app properly
bool quit(StateMachine & SM) {
    EMU_ROS2 & sm = static_cast<EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    //keyboard press
    if ( sm.robot()->keyboard->getKeyUC()=='Q' ) {
        std::raise(SIGTERM); //Clean exit
        return true;
    }

    //Check incoming command requesting state change
    if ( sm.UIserver->isCmd("QUIT") ) {
        sm.UIserver->sendCmd(string("OKQU"));
        spdlog::debug("goToQuit");
        std::raise(SIGTERM); //Clean exit
        return true;
    }

    return false;
}


//Fake transition (return false all the time) used to update the mass parameter
bool updateMass(StateMachine & SM) {
    EMU_ROS2 & sm = static_cast<EMU_ROS2 &>(SM); //Cast to specific StateMachine type

    std::vector<double> params;

    //Path update assistance
    if ( sm.UIserver->isCmd("UDMA", params) ) {
        std::shared_ptr<M3StandbyPublishState> s = sm.state<M3StandbyPublishState>("StandbyState");

        if(params.size() == 1) {
            if(params[0]>=0. &&  params[0]<=3.) {
                s->setMass(params[0]);
            }
            sm.UIserver->sendCmd(string("OKUM"));
            spdlog::debug("updateMass: {}", params[0]);
        }
        else {
            spdlog::warn("updateMass: Error: number of command parameters.");
            sm.UIserver->sendCmd(string("ERUD"));
        }
    }

    return false;
}



EMU_ROS2::EMU_ROS2(int argc, char **argv)  {
    spdlog::debug("Am I here 2");
    //Create a Robot and set it to generic state machine
    setRobot(std::make_unique<RobotM3>("EMU_FOURIER", "M3_params.yaml"));
    // setRobot(std::make_unique<RobotM3>("EMU_MELB", "M3_params.yaml"));

    // Configure ROS2 initialisation options and disable SIGINT capture (handled by CORC)
    rclcpp::InitOptions ros_init = rclcpp::InitOptions();
    ros_init.shutdown_on_signal = false;
    rclcpp::init(argc, argv, ros_init);

    // Create the ROS2 node and pass a reference to the X2 Robot object
    m_Node = std::make_shared<EMU_ROS2_Node>("EMU", robot());

    //TODO: include new FLNL state and associated transitions
    //TODO: proper logic and transitions

    //Create state instances and add to the State Machine
    addState("DoNothingState", std::make_shared<M3NothingState>(robot(), this));
    addState("ResetState", std::make_shared<M3NothingState>(robot(), this));
    addState("CalibState", std::make_shared<M3CalibState>(robot(), this));
    addState("StandbyState", std::make_shared<M3StandbyPublishState>(robot(), this));
    addState("LockState", std::make_shared<M3LockState>(robot(), this));
    addState("ReproduceState", std::make_shared<M3ReproduceState>(robot(), this));

    //Define transitions between states

    // transition to standby states
    addTransition("DoNothingState", &goToCalib, "CalibState");
    addTransition("CalibState", &endCalib, "StandbyState");
    addTransition("StandbyState", &updateMass, "StandbyState"); //Fake transition never returning true

    // transition to lock state
    addTransition("StandbyState", &goToLock, "LockState");
    addTransition("LockState", &goToUnlock, "StandbyState");

    // transition to reproduce state
    addTransition("StandbyState", &goToReproduce, "ReproduceState");
    addTransition("ReproduceState", &goToGravity, "StandbyState");

    // transition to the reset state
    addTransition("StandbyState", &goToReset, "ResetState");
    addTransition("ReproduceState", &goToReset, "ResetState");
    addTransition("LockState", &goToReset, "ResetState");
    addTransition("ResetState", &goToGravity, "StandbyState");

    addTransitionFromAny(&quit, "StandbyState");
    addTransition("StandbyState", &quit, "StandbyState"); //From any does not apply to self (destination state)
}
EMU_ROS2::~EMU_ROS2() {
}

/**
 * \brief start function for running any designed statemachine specific functions
 * for example initialising robot objects.
 *
 */
void EMU_ROS2::init() {
    spdlog::debug("EMU_ROS2::init()");
    if(robot()->initialise()) {
        logHelper.initLogger("EMU_ROS2Log", "logs/EMU_ROS2.csv", LogFormat::CSV, true);
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
        UIserver->registerState(Command);
        UIserver->registerState(MvtProgress);
        UIserver->registerState(Contribution);
    }
    else {
        spdlog::critical("Failed robot initialisation. Exiting...");
        std::raise(SIGTERM); //Clean exit
    }
}

void EMU_ROS2::end() {
    if(running())
        UIserver->closeConnection();
    StateMachine::end();
}


/**
 * \brief Statemachine to hardware interface method. Run any hardware update methods
 * that need to run every program loop update cycle.
 *
 */
void EMU_ROS2::hwStateUpdate() {
    StateMachine::hwStateUpdate();
    //Also send robot state over network
    UIserver->sendState();
    //Attempt to reconnect (if not already waiting for connection)
    UIserver->reconnect();
    // Allow for the ROS2 node to execute callbacks (e.g., subscriptions)
    rclcpp::spin_some(get_node()->get_interface());
}



#include "FT_EMU_ROS2_States.h"
#include "FT_EMU_ROS2.h"
#include <fmt/format.h>

using namespace std;

/** @defgroup PrintingFunctions Convenience progress bar printing functions
 *  @{
 */
//Print a progress bar for a value from 0 to 1 and additional pre and post text
void printProgress(double val, std::string pre_txt, std::string post_txt, int l) {
    val = fmin(fmax(val, 0.), 1.0);
    std::cout << pre_txt << " |";
    for(int i=0; i<round(val*l); i++)
        std::cout << "=";
    for(int i=0; i<round((1.-val)*l); i++)
        std::cout << "-";
    std::cout << "| ";
    if(post_txt.empty())
        std::cout << "(" << val*100 << "%)  ";
    else
        std::cout << post_txt;
}
//Print a progress bar, centered at 0 for a value from -1 to 1 and additional pre and post text
void printProgressCenter(double val, std::string pre_txt, std::string post_txt, int l) {
    val = fmin(fmax(val, -1.0), 1.0);
    std::cout << pre_txt << " |";
    if(val>=0) {
        for(int i=0; i<round(l/2.); i++)
            std::cout << "-";
        for(int i=0; i<round(val*l/2.); i++)
            std::cout << "=";
        for(int i=0; i<round((1.-val)*l/2.); i++)
            std::cout << "-";
    }
    if(val<0) {
        val=-val;
        for(int i=0; i<round((1.-val)*l/2.); i++)
            std::cout << "-";
        for(int i=0; i<round(val*l/2.); i++)
            std::cout << "=";
        for(int i=0; i<round(l/2.); i++)
            std::cout << "-";
    }
    std::cout << "| ";
    if(post_txt.empty())
        std::cout << "(" << val*100 << "%)  ";
    else
        std::cout << post_txt;
}
/** @} */ // end of PrintingFunctions

/** @defgroup GenericEMUFunctions Generic EMU control functions
 *  @{
 */
//! Impedance force for given stiffness and damping matrices and target position and velocities
VM3 impedance(Eigen::Matrix3d K, Eigen::Matrix3d D, VM3 X0, VM3 X, VM3 dX, VM3 dXd) {
    return K*(X0-X) + D*(dXd-dX);
}
/** @} */ // end of GenericEMUFunctions


/**
 * \brief Generic state type for used with M3DemoMachine, providing running time and iterations number: been superseeded by default state, not very much useful anymore.
 *
 */
void FT_EMU_ROS2_State::during(void) {
    //Actual state during
    duringCode();

    //Manage state logger if used
    if(stateLogger.isInitialised()) {
        stateLogger.recordLogData();
    }
    if(robot->keyboard->getKeyUC()=='V' ){
        sm->robotVerbose = !sm->robotVerbose;
    }
}
void FT_EMU_ROS2_State::printFullStates(const std::string& extras) {
    if(iterations()%200==1) {
        robot->printJointStatus();
        robot->printStatus();    
        robot->printWrenches();
        if (!extras.empty()) {
            std::cout << extras;
        }
        std::cout <<  std::endl;
        std::cout <<  std::noshowpos;
    }
}

/**
 * \brief Initialises everthing then waits for a calib command. Set drives in torque control mode.
 *
 */
void M3InitState::entryCode(void) { 
    if (sm->stateID != "quit"){
        sm->stateID = "init";
        sm->get_node()->publish_state_var(sm->stateID,0);
        
        robot->initTorqueControl(); robot->setJointTorque(VM3(0,0,0));robot->setFT_SensorsFilter(); 
    }
}
void M3InitState::duringCode(void){ 
    robot->setJointTorque(VM3(0,0,0)); 
    sessionParam = sm->get_node()->get_session_param();
    sessionParamInts = sm->get_node()->get_session_param_ints();

    sm->expID = sessionParam[0];
    sm->patientID = sessionParam[1];
    sm->subID = sessionParam[2];
    sm->maxSbmvmt = sessionParamInts[0];
    sm->learnOutput = sessionParam[3];
}
void M3InitState::exitCode(void) { 
    robot->setJointTorque(VM3(0,0,0));
    spdlog::info("Session Params: [ expID: {}, patientID: {}, subID: {}, maxSbmvmt: {}, Learned Output: {} ]",sm->expID,sm->patientID,sm->subID,sm->maxSbmvmt,sm->learnOutput);
}
    
/**
 * \brief Position calibration of M3. Go to the bottom left stops of robot at constant torque for absolute position calibration. Set drives in torque control mode.
 *
 */
void M3CalibState::entryCode(void) {
    sm->stateID = "calib";
    sm->get_node()->publish_state_var(sm->stateID,0);
    calibDone=false;
    for(unsigned int i=0; i<3; i++) {
        stop_reached_time[i] = .0;
        at_stop[i] = false;
    }
    robot->decalibrate();
    robot->initTorqueControl();
    robot->printJointStatus();
    qi=robot->getPosition();
    std::cout << "Calibrating EMU (keep clear)..." << std::flush;
    std::cout <<  std::endl;
}
//Move slowly on each joint until max force detected
void M3CalibState::duringCode(void) {
    VM3 tau(0, 0, 0);

    //Apply constant torque (with damping) unless stop has been detected for more than 0.5s
    VM3 vel=robot->getVelocity();
    double b = 7.;
    for(unsigned int i=0; i<3; i++) {
        tau(i) = std::min(std::max(8 - b * vel(i), .0), 8.);
        #ifndef NOROBOT
            if(stop_reached_time(i)>0.5 && robot->getPosition()!=qi ) {
                at_stop[i]=true;
            }
        #else
            at_stop[i]=true;
        #endif
        if(vel(i)<0.01) {
            stop_reached_time(i) += dt();
        }
    }

    //Switch to gravity control when done
    if(robot->isCalibrated()) {
        robot->setEndEffForceWithCompensation(VM3(0,0,0));
        calibDone=true; //Trigger event
    }
    else {
        //If all joints are calibrated
        if(at_stop[0] && at_stop[1] && at_stop[2]) {
            robot->applyCalibration();
            std::cout << "OK." << std::endl;
        }
        else {
            robot->setJointTorque(tau);
            if(iterations()%100==1) {
                std::cout << "." << std::flush;
            }
        }
    }
}
void M3CalibState::exitCode(void) {
    robot->setEndEffForceWithCompensation(VM3(0,0,0));
}

/**
 * \brief FT calibration
 * 
 */
void M3FTCalibState::entryCode(void) {
    sm->stateID = "calib";
    sm->get_node()->publish_state_var(sm->stateID,0);
    
    calibDone = false;
    std::cout << "Calibrating RFT (keep clear)..." << std::flush;
    Eigen::VectorXd wrenches = robot->getWrenches();
    readings = Eigen::ArrayXXd::Zero(NUM_CALIBRATE_READINGS, wrenches.size());

    // Take average of the matrices
    Eigen::VectorXd offsets = Eigen::VectorXd::Zero(readings.cols());

    if(spdlog::get_level()<=spdlog::level::debug) {
        rcpputils::fs::path logDir = rcpputils::fs::path("logs/corc_recordings/FT_EMU") / sm->expID / sm->patientID / sm->subID;
        rcpputils::fs::create_directories(logDir);
        std::string logPath = (logDir / ("FT_EMU_Calib.csv")).string();
                
        // Initialize logger
        stateLogger.initLogger("FT_EMU_Calib_Log", logPath, LogFormat::CSV, true);
        stateLogger.add(running(), "%Time(s)");
        stateLogger.add(robot->getWrenches(), "W");
        stateLogger.startLogger();
    }
    robot->startFT_Sensors();
    readingCount = 0;
    std::cout << "Calibrating RFT (keep clear)..." << std::flush;
    std::cout <<  std::endl;
}
//collect offsets to the RFT sensors
void M3FTCalibState::duringCode(void) {
    // Collect data and save
    if (readingCount< NUM_CALIBRATE_READINGS){
        if(iterations()%100==1) {
            std::cout << "." << std::flush;
        }
        readings.row(readingCount) = robot->getWrenches();
    }
    else
    {
        std::cout << "OK." << std::endl;
        calibDone = true;
    }
    readingCount = readingCount+1;
}
void M3FTCalibState::exitCode(void) {
    // Take average of the matrices
    Eigen::VectorXd offsets = Eigen::VectorXd::Zero(readings.cols());

    // Set offsets for crutches
    for (int i = 0; i < readings.cols(); i++) {
        offsets[i] = readings.col(i).sum()/NUM_CALIBRATE_READINGS;
    }
    spdlog::info("FT Calibration Complete, setting offsets");
    for (int i = 0; i < readings.cols()/6; i++){
        Eigen::VectorXd cur_offsets = offsets.segment(i*6, 6).transpose();
        spdlog::debug("RFT {} Offset [{}, {}, {}, {}, {}, {}]", i,cur_offsets[0],cur_offsets[1],cur_offsets[2],cur_offsets[3],cur_offsets[4],cur_offsets[5]);
        if (offsets.segment(i*6, 6).isApprox(Eigen::VectorXd::Zero(6))){
            spdlog::warn("RFTs may not be connected");
        }
    }
    robot->setWrenchesOffset(offsets);
    spdlog::info("CalibrateState Exit");
    robot->stopFT_Sensors();
}

/**
 * \brief Standby State for therapist to set variation number
 * \details Mass is controllable through keyboard inputs. Assumes drives in torque control already.
 */
 
void M3StandbyState::entryCode(void) {  
    sm->stateID = "standby";
    sm->get_node()->publish_state_var(sm->stateID,0);
    
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), false);
//     std::cout << "Press S to decrease mass (-100g), W to increase (+100g)." << std::endl;
    
//     lastWrenches = robot->getWrenches();
//     if(robot->startFT_Sensors())
//     {
//         spdlog::info("Starting RFT");
//     }
    spdlog::info("Current Var: {}. Add Var ('2'). Start Demo ('1'). Start Repro ('3').",sm->varNum);
}
void M3StandbyState::duringCode(void) {

    /**
     * \ad hoc mass compensation
     */
     
    //Mass controllable through keyboard inputs
    if(robot->keyboard->getS()) {
        mass -=0.2;
    }
    if(robot->keyboard->getW()) {
        mass +=0.2;
    }
    //Bound mass to +-10kg
    if(mass>mass_limit) {
        mass = mass_limit;
    }
    if(mass<-mass_limit) {
        mass = -mass_limit;
    }

    //Calculate effective applied mass based on possible transition (change mass
    sm->MassComp += sign(mass - sm->MassComp)*change_mass_rate*dt();

    //If after transitioning dampin time
    if(running()>transition_t) {
        //Apply corresponding deweighting force
        robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    }
    else {
        //Apply corresponding deweighting force w/o friction comp
        robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), false);
    }
   
//     // Check if some sensors are not responding properly every one second
//     Eigen::VectorXd curWrenches = robot->getWrenches();
//     if(iterations() % 100 == 99){
//         bool ok = true;
//         if (lastWrenches.isApprox(curWrenches)){
//             ok = false;
//         }
//         lastWrenches = curWrenches;
//     }
    
    if(robot->joystick->isButtonTransition(2)>0 and sm->varNum < 6)
    {
        sm->varNum = sm->varNum+1;
        spdlog::info("Current Var: {}. Add Var ('2'). Start Demo ('1'). Start Repro ('3'). ",sm->varNum);
    }
    else if ((robot->joystick->isButtonTransition(2)>0 and sm->varNum == 6))
    {
        sm->varNum = 1;
        spdlog::info("Current Var: {}. Add Var ('2'). Start Demo ('1'). Start Repro ('3').",sm->varNum);
    }
    
    if (sm->robotVerbose){
        std::ostringstream oss;
        oss << std::setprecision(3) << std::fixed << std::showpos;
        oss << "MassComp:"<< sm->MassComp << "\t\t";
        oss <<  "MassReq:"<<  mass         << "\t\t";
        oss <<  "ForceReq:"<< mass * 9.8   << "\n";
        printFullStates(oss.str());
    }
}
void M3StandbyState::exitCode(void) {
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    sm->robotVerbose = false;
//     if(robot->stopFT_Sensors())
//     {
//         spdlog::info("Stopping RFT");
//     }
}

/**
 * \brief Lock in place: position control around current point. Assumes drives in torque control already.
 *
 */
void M3LockState::entryCode(void) {
    sm->stateID = "lock";
    sm->get_node()->publish_state_var(sm->stateID,0);
    robot->setEndEffForceWithCompensation(VM3::Zero(), false);
    X0 = robot->getEndEffPosition();
    if(robot->startFT_Sensors())
    {
        spdlog::info("Starting RFT");
    }
}
void M3LockState::duringCode(void) {
    //Impedance on point
    Eigen::Matrix3d K = k*Eigen::Matrix3d::Identity();
    Eigen::Matrix3d D = d*Eigen::Matrix3d::Identity();
    VM3 Fd = impedance(K, D, X0, robot->getEndEffPosition(), robot->getEndEffVelocity());
    robot->setEndEffForceWithCompensation(Fd, false);
    
    if (sm->robotVerbose){
        printFullStates();
    }
}
void M3LockState::exitCode(void) {
    robot->setEndEffForceWithCompensation(VM3::Zero(), false);
    sm->robotVerbose = false;
    if(robot->stopFT_Sensors())
    {
        spdlog::info("Stopping RFT");
    }
}

/**
 * \brief Demo
 * \Starts publishing emu task dynamics to subscribers when user starts the repetition
 */
void M3DemoState::entryCode(void) {
    sm->stateID = "demo";
    sm->get_node()->publish_state_var(sm->stateID,sm->varNum);
    
    // set EE force
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    if(robot->startFT_Sensors())
    {
        spdlog::info("Starting RFT");
    }
    
    // Initialize logger
    rcpputils::fs::path logDir = rcpputils::fs::path("logs/corc_recordings/FT_EMU") 
                                  / sm->expID 
                                  / sm->patientID 
                                  / sm->subID 
                                  / fmt::format("var{}", sm->varNum);
    rcpputils::fs::create_directories(logDir);
    std::string logPath = (logDir / ("FT_EMU_Demo.csv")).string();
    stateLogger.initLogger("FT_EMU_Demo", logPath, LogFormat::CSV, true);
    stateLogger.add(running(), "%Time(s)");
    stateLogger.add(sm->repNum, "rep");
    stateLogger.add(sm->sbmvmtNum, "sbmvmt");
    stateLogger.add(robot->getEndEffPosition(), "X");
    stateLogger.add(robot->getEndEffVelocityFiltered(), "dXFilt");
    stateLogger.add(robot->getWrenches(), "F");
    stateLogger.startLogger();

    sm->repNum = 1;
    sm->sbmvmtNum = 0;
    spdlog::info("Var {}, Start Rep {} ('2'). End Var {} ('S').",sm->varNum,sm->repNum,sm->varNum);
}
void M3DemoState::duringCode(void) {

    if(robot->joystick->isButtonTransition(2)>0 and sm->sbmvmtNum <= sm->maxSbmvmt)
    {
        sm->sbmvmtNum = sm->sbmvmtNum+1;
        if (sm->sbmvmtNum <sm->maxSbmvmt){
        spdlog::info("Var {}, Rep {}, Sbmvmt {}/{}. Add Sbmvmt ('2'). Redo Rep('3').",sm->varNum, sm->repNum,sm->sbmvmtNum,sm->maxSbmvmt);
        }
        else if (sm->sbmvmtNum == sm->maxSbmvmt){
        spdlog::info("Var {}, Rep {}, Sbmvmt {}/{}. Stop Rep {} ('2'). Redo Rep('3').",sm->varNum, sm->repNum,sm->sbmvmtNum,sm->maxSbmvmt,sm->repNum);
        }
        else{
        spdlog::info("Var {}, Rep {}. Sbmvmts finished. Go to next rep ('1') or redo current rep('3')",sm->varNum, sm->repNum);
        }
    }
    if(robot->joystick->isButtonTransition(1)>0 and sm->sbmvmtNum > sm->maxSbmvmt)
    {
        sm->repNum = sm->repNum+1; sm->sbmvmtNum = 0;
        spdlog::info("Var {}, Start Rep {} ('2'). End Var {} ('S').",sm->varNum,sm->repNum,sm->varNum);
    }
    if(robot->keyboard->getKeyUC()=='R' and sm->repNum > 1)
    {
        sm->repNum = sm->repNum-1; sm->sbmvmtNum = 0;
        spdlog::info("Redoing Rep {}",sm->repNum);
    }
    if(robot->joystick->isButtonTransition(3)>0 and sm->sbmvmtNum > 0)
    {
        sm->sbmvmtNum = 0;
        spdlog::info("Var {}, Redo Rep {} ('2'). End Var {} ('S').",sm->varNum,sm->repNum,sm->repNum,sm->varNum);
    }

    // publish emu task dynamics before end of sbmvmts to ROS2 network
    if (sm->sbmvmtNum <= sm->maxSbmvmt)
    {
        sm->get_node()->publish_task_dynamics(sm->repNum,sm->sbmvmtNum);
    }
    
    if (sm->robotVerbose){
        printFullStates();
    }
}
void M3DemoState::exitCode(void) {
    // set EE force
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    sm->repNum = 1;
    sm->sbmvmtNum = 0;
    sm->robotVerbose = false;
    if(robot->stopFT_Sensors())
    {
        spdlog::info("Stopping RFT");
    }
}


/**
 * \brief Learn state
 * \start the learning on lfd SM, waits for the learning complete signal from lfd SM
 */
void M3LearnState::entryCode(void) { 
    sm->stateID = "learn";
    sm->get_node()->publish_state_var(sm->stateID,sm->varNum);
    
    sm->learn_finish = false;
    
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), false);
}
void M3LearnState::duringCode(void){ 
    sessionParamInts = sm->get_node()->get_session_param_ints();
    
    if (sessionParamInts[1]>0){
        sm->learn_finish=true;
        // Extract from 3rd element to end and convert to bool
        std::vector<bool> learn_success_bool;
        std::transform(sessionParamInts.begin() + 2, sessionParamInts.end(),
                       std::back_inserter(learn_success_bool),
                       [](uint16_t val) { return val != 0; });
        sm->learn_success = learn_success_bool;
    }
    
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
}
void M3LearnState::exitCode(void) { 
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    sm->robotVerbose = false;

       
    std::string status_str = "Status=[ ";
    for (bool val : sm->learn_success) {
        status_str += (val ? "true " : "false ");
    }
    status_str += "]";
    
    spdlog::info("{}", status_str);
    spdlog::info("Learning Complete. {}",status_str);
}

/**
 * \brief NewTP state
 * \set the TP for the current var to be reproduced by robot
 */
void M3NewTPState::entryCode(void) { 
    sm->stateID = "new_tp";
    sm->get_node()->publish_state_var(sm->stateID,sm->varNum);

    // set EE force
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    if(robot->startFT_Sensors())
    {
        spdlog::info("Starting RFT");
    }
    
    sm->sbmvmtNum = 0;
    spdlog::info("Var {}. Add TP {}/{} ('2'). Redo Var {} TP ('3').",sm->varNum,sm->sbmvmtNum+1,sm->maxSbmvmt+1,sm->varNum);
}
void M3NewTPState::duringCode(void){   
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    
    if(robot->joystick->isButtonTransition(2)>0 and sm->sbmvmtNum <= sm->maxSbmvmt)
    {
        sm->sbmvmtNum = sm->sbmvmtNum+1;
        if (sm->sbmvmtNum <=sm->maxSbmvmt){
            spdlog::info("Var {}. TP {}/{} Added. Add TP ('2'). Redo Var {} TP ('3')",sm->varNum, sm->sbmvmtNum, sm->maxSbmvmt+1,sm->varNum);
        }
        else{
            spdlog::info("Var {}. TP {}/{} Complete. Redo Var {} TP ('3'). Go to Repro_LfD ('1')",sm->varNum, sm->sbmvmtNum, sm->maxSbmvmt+1,sm->varNum);
            sm->lfd_repro_ready = true;
        }
        
        // publish TP
        sm->get_node()->publish_task_dynamics(sm->repNum,sm->sbmvmtNum);
    }
    
    if(robot->joystick->isButtonTransition(3)>0 and sm->sbmvmtNum > 0)
    {
        sm->sbmvmtNum = 0;
        sm->lfd_repro_ready = false;
        sm->get_node()->publish_task_dynamics(sm->repNum,sm->sbmvmtNum);
        spdlog::info("Var {}. Add TP {}/{} ('2'). Redo Var {} TP ('R','3'').",sm->varNum,sm->sbmvmtNum+1,sm->maxSbmvmt+1,sm->varNum);
    }
}
void M3NewTPState::exitCode(void) { 
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    sm->robotVerbose = false;
    sm->lfd_repro_ready = false;
    sm->sbmvmtNum = 0;
}

/**
 * \brief ReproLfD state
 * \ reproduce interaction from the LfD process
 */
void M3ReproLfDState::entryCode(void) { 
    sm->stateID = "repro_lfd";
    sm->get_node()->publish_state_var(sm->stateID,sm->varNum);

    // set EE force
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    if(robot->startFT_Sensors())
    {
        spdlog::info("Starting RFT");
    }
    
    sm->repNum = 1;
    sm->sbmvmtNum = 0;
    spdlog::info("Var {}. Start Interaction {} ('2'). End Var {} ('S')",sm->varNum,sm->repNum,sm->repNum,sm->varNum);
}
void M3ReproLfDState::duringCode(void){   
    // put this here so we can get the latest information 
    if (start_interaction)
    {
        // to subscribe to the commanded wrenches
        desired_wrench = sm->get_node()->get_desired_wrench();
        if (desired_wrench.size() > 0)
        {
            std::cout << std::setprecision(3) << std::fixed << std::showpos;
            std::cout << "desired_wrench\t=[ " << desired_wrench.transpose() <<" ]\t";
            std::cout <<  std::endl;
            std::cout <<  std::noshowpos;
            robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
        }
    }
     
    if(robot->joystick->isButtonTransition(2)>0 and !start_interaction)
    {
        sm->sbmvmtNum = 1;
        start_interaction = true;
        spdlog::info("Var {}. Interaction {} Started. Stop Interaction {} ('1'). End Var {} ('S')",sm->varNum,sm->repNum,sm->repNum,sm->varNum);
    }
    if(robot->joystick->isButtonTransition(1)>0 and start_interaction)
    {
        sm->sbmvmtNum = 0;
        sm->repNum = sm->repNum+1;
        start_interaction = false;
        sm->get_node()->reset_desired_wrench();
        spdlog::info("Var {}. Interaction {} Stopped. Start Interaction {} ('2'). End Var {} ('S')",sm->varNum,sm->repNum-1,sm->repNum,sm->varNum);
    }
    
    sm->get_node()->publish_task_dynamics(sm->repNum,sm->sbmvmtNum);

    if (sm->robotVerbose){
        printFullStates();
    }
}
void M3ReproLfDState::exitCode(void) { 
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), true);
    sm->robotVerbose = false;
    sm->repNum = 1;
    sm->sbmvmtNum = 0;
    if(robot->stopFT_Sensors())
    {
        spdlog::info("Stopping RFT");
    }
}
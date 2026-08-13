#include "FT_EMU_ROS2_States.h"
#include "FT_EMU_ROS2.h"

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
    sm->get_node()->publish_joint_states();
}

/**
 * \brief Position calibration of M3. Go to the bottom left stops of robot at constant torque for absolute position calibration. Set drives in torque control mode.
 *
 */
void M3CalibState::entryCode(void) {
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
    calibDone = false;
    

    Eigen::VectorXd force = robot->getFT_readings();
    readings = Eigen::ArrayXXd::Zero(NUM_CALIBRATE_READINGS, force.size());

    // Take average of the matrices
    Eigen::VectorXd offsets = Eigen::VectorXd::Zero(readings.cols());
    robot->setFTOffsets(offsets);

    if(spdlog::get_level()<=spdlog::level::debug) {
        stateLogger.initLogger("FTCalib", "logs/FTCalibLog.csv", LogFormat::CSV, true);
        stateLogger.add(running(), "%Time(s)");
        stateLogger.add(robot->getFT_readings(), "F");
        stateLogger.startLogger();
    }
    robot->startFT_Sensors();
    curReading =0;
    std::cout << "Calibrating RFT (keep clear)..." << std::flush;
}
//collect offsets to the RFT sensors
void M3FTCalibState::duringCode(void) {
    // Collect data and save
    if (curReading< NUM_CALIBRATE_READINGS){
        if(iterations()%100==1) {
            std::cout << "." << std::flush;
        }
        readings.row(curReading) = robot->getFT_readings();
    }
    else
    {
        std::cout << "OK." << std::endl;
        calibDone = true;
    }
    curReading = curReading+1;
}
void M3FTCalibState::exitCode(void) {
    // Take average of the matrices
    Eigen::VectorXd offsets = Eigen::VectorXd::Zero(readings.cols());

    // Set offsets for crutches
    for (int i = 0; i < readings.cols(); i++) {
        offsets[i] = readings.col(i).sum()/NUM_CALIBRATE_READINGS;
        spdlog::debug("RFT Offset {}", offsets[i]);
    }
    spdlog::info("FT Calibration Complete, setting offsets");

    for (int i = 0; i < readings.cols()/6; i++){
        if (offsets.segment(i*6, 6).isApprox(Eigen::VectorXd::Zero(6))){
            spdlog::warn("RFTs may not be connected");
        }
    }

    robot->setFTOffsets(offsets);
    spdlog::info("CalibrateState Exit");
    robot->stopFT_Sensors();
}

/**
 * \brief Start publishing robot state. Provide end-effector mass compensation on M3. 
 * \details Mass is controllable through keyboard inputs. Assumes drives in torque control already.
 */
void M3StandbyPublishState::entryCode(void) {
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), false);
    std::cout << "Press S to decrease mass (-100g), W to increase (+100g)." << std::endl;
    
    lastRFTReadings = robot->getFT_readings();
    if(robot->startFT_Sensors())
    {
        spdlog::info("Starting RFT");
    }
}
void M3StandbyPublishState::duringCode(void) {
    
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

    //Mass controllable through keyboard inputs
    if(robot->keyboard->getS()) {
        mass -=0.2;
    }
    if(robot->keyboard->getW()) {
        mass +=0.2;
    }
    
    Eigen::VectorXd curReadings = robot->getCorrectedFT_readings();
    // Check if some sensors are not responding properly every one second
    if(iterations() % 100 == 99){
        bool ok = true;
        if (lastRFTReadings.isApprox(curReadings)){
            spdlog::error("Crutches Not Updating");
            ok = false;
        }
        lastRFTReadings = curReadings;
    }
    
    if(iterations()%200==1) {
        robot->printJointStatus();
        robot->printStatus();
        robot->printFT_readings(curReadings);
        std::cout << std::setprecision(3) << std::fixed << std::showpos;
        std::cout << "MassComp"  << "=[ " << sm->MassComp << " ]\t";
        std::cout << "MassReq"  << "=[ " << mass << " ]\t";
        std::cout << "ForceReq"  << "=[ " << mass*9.8 << " ]\t";
        std::cout <<  std::endl;
        std::cout <<  std::endl;
        std::cout <<  std::noshowpos;
    }
    
}
void M3StandbyPublishState::exitCode(void) {
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), false);
    if(robot->stopFT_Sensors())
    {
        spdlog::info("Stopping RFT");
    }
}

/**
 * \brief Lock in place: position control around current point. Assumes drives in torque control already.
 *
 */
void M3LockState::entryCode(void) {
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
    Eigen::VectorXd curReadings = robot->getCorrectedFT_readings();
    
    if(iterations()%200==1) {
        robot->printJointStatus();
        robot->printStatus();    
        robot->printFT_readings(curReadings);
        std::cout <<  std::endl;
        std::cout <<  std::noshowpos;
    }
}
void M3LockState::exitCode(void) {
    robot->setEndEffForceWithCompensation(VM3::Zero(), false);
    if(robot->stopFT_Sensors())
    {
        spdlog::info("Stopping RFT");
    }
}



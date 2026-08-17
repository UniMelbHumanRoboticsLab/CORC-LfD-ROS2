#include "UBO_ROS2_States.h"
#include "UBO_ROS2.h"
#include <fmt/format.h>

using namespace std;

/**
 * \brief Generic state type to be used for UBO_ROS2 StateMachine, providing running time and iterations number
 *
 */
void UBO_ROS2_State::entry(void) {
    entry_num++;
    //Actual state entry
    entryCode();
    sm->get_node()->publish_state(state_name);
    
};
void UBO_ROS2_State::during(void) {
    //Actual state during
    duringCode();

    //Manage state logger if used
    if(stateLogger.isInitialised()) {
        stateLogger.recordLogData();
    }
    // if(iterations()%50==1)
    // sm->get_node()->publish_joint_states();
}
void UBO_ROS2_State::exit(void) 
{
    //Actual state exit
    exitCode(); 
    if(stateLogger.isInitialised())
        stateLogger.endLog();
}

/**
 * \brief UBO calibration
 * 
 */
void UBOCalibState::entryCode(void) {
    calibDone = false;
    spdlog::info("CalibrateState entry");
    spdlog::info("Calibrating....");

    Eigen::VectorXd force = robot->getUBO_readings();
    readings = Eigen::ArrayXXd::Zero(NUM_CALIBRATE_READINGS, force.size());

    // Take average of the matrices
    Eigen::VectorXd offsets = Eigen::VectorXd::Zero(readings.cols());
    robot->setUBOOffsets(offsets);

    if(spdlog::get_level()<=spdlog::level::debug) {
        stateLogger.initLogger("UBOCalib", "logs/UBOCalibLog.csv", LogFormat::CSV, true);
        stateLogger.add(running(), "%Time(s)");
        stateLogger.add(robot->getUBO_readings(), "F");
        stateLogger.startLogger();
    }
    robot->startUBO_FTSensors();
    currReading =0;
}
//collect offsets to the RFT sensors
void UBOCalibState::duringCode(void) {
    // Collect data and save
    if (currReading< NUM_CALIBRATE_READINGS){
        if (currReading%50==1)
        {
            spdlog::info("Iter {}",currReading);
        }
        readings.row(currReading) = robot->getUBO_readings();
        robot->printUBO_readings(readings.row(currReading));
    }
    else
    {
        calibDone = true;
    }
    currReading = currReading+1;
}
void UBOCalibState::exitCode(void) {
    // Take average of the matrices
    Eigen::VectorXd offsets = Eigen::VectorXd::Zero(readings.cols());

    // Set offsets for crutches
    for (int i = 0; i < readings.cols(); i++) {
        offsets[i] = readings.col(i).sum()/NUM_CALIBRATE_READINGS;
        spdlog::debug("RFT Offset {}", offsets[i]);
    }
    spdlog::info("UBO Calibration Complete, setting offsets");

    for (int i = 0; i < readings.cols()/6; i++){
        if (offsets.segment(i*6, 6).isApprox(Eigen::VectorXd::Zero(6))){
            spdlog::warn("RFTs may not be connected");
        }
    }

    robot->setUBOOffsets(offsets);

    spdlog::info("CalibrateState Exit");
    robot->stopUBO_FTSensors();
}

/**
 * \brief UBO Idle
 *
 */
void UBOIdleState::entryCode(void) {
    
    spdlog::info("IdleState entry");
    spdlog::info("S to start logging");
    
};

void UBOIdleState::duringCode(void){
    
    // Do nothing
};

void UBOIdleState::exitCode(void) {
    spdlog::info("IdleState Exit");
};

/**
 * \brief UBO Record
 *
 */
void UBORecordState::entryCode(void) {
    spdlog::info("RecordState entry num {}",entry_num);
    spdlog::info("S to Stop");
    lastRFTReadings = robot->getUBO_readings();
    if(robot->startUBO_FTSensors())
    {
        spdlog::info("Starting RFT");
    }
    sm->UIserver->sendCmd(string("rec"));

    if(spdlog::get_level()<=spdlog::level::debug) {
        std::string recordLogName = fmt::format("logs/corc_recordings/UBORecord{}Log.csv", entry_num);
        stateLogger.initLogger("UBORecord", recordLogName, LogFormat::CSV, true);

        if (stateLogger.vectorOfLogElements.size() == 0)
        {
            stateLogger.add(running(), "%Time (s)");
            stateLogger.add(robot->getUBO_readings(), "F");
        }
        stateLogger.startLogger();
    }
};

void UBORecordState::duringCode(void){
    Eigen::VectorXd curReadings = robot->getCorrectedUBO_readings();
    Eigen::VectorXd olStatus = robot->getUBO_OLStatus();

    // Check if some sensors are not responding properly every one second
    if(ticker % 100 == 99){
        bool ok = true;
        if (lastRFTReadings.isApprox(curReadings)){
            spdlog::error("Crutches Not Updating");
            ok = false;
        }
        lastRFTReadings = curReadings;
    }

    robot->printUBO_readings(curReadings);
    sm->UIserver->sendState();
    sm->get_node()->publish_wrenches(curReadings,olStatus);
    ticker++;
};

void UBORecordState::exitCode(void) {
    if(robot->stopUBO_FTSensors())
    {
        spdlog::info("Stopping RFT");
    }
    spdlog::info("RecordState Exit");
};



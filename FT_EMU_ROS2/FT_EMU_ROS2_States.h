/**
 * \file FT_EMU_ROS2_States.h
 * \author Jia Quan Loh
 * \date 2024-03-26
 *
 * \copyright Copyright (c) 2024
 *
 */

#ifndef M3STATE_H_DEF
#define M3STATE_H_DEF

#include "State.h"
#include "FT_RobotM3.h"
#include "LogHelper.h"
#include "FLNLHelper.h"

#define NUM_CALIBRATE_READINGS 1000

class FT_EMU_ROS2; // declare empty class for FT_EMU_ROS2 state machine for forward inclusion

/** @defgroup PrintingFunctions Convenience progress bar printing functions
 *  @{
 */
//! Print a progress bar for a value from 0 to 1 and additional pre and post text
void printProgress(double val, std::string pre_txt="", std::string post_txt="", int l=80 /*nb char long*/);
//! Print a progress bar, centered at 0 for a value from -1 to 1 and additional pre and post text
void printProgressCenter(double val, std::string pre_txt="", std::string post_txt="", int l=80 /*nb char long*/);
/** @} */ // end of PrintingFunctions


/** @defgroup GenericEMUFunctions Generic EMU control functions
 *  @{
 */
//! Impedance force for given stiffness and damping matrices and target position and velocities
VM3 impedance(Eigen::Matrix3d K, Eigen::Matrix3d D, VM3 X0, VM3 X, VM3 dX, VM3 dXd=VM3::Zero());
/** @} */ // end of GenericEMUFunctions

/**
 * \brief Generic state type for used with M3DemoMachine, providing running time and iterations number: been superseeded by default state, not very much useful anymore.
 *
 */
class FT_EMU_ROS2_State : public State {
   protected:
    FT_RobotM3 * robot;                               //!< Pointer to state machines robot object

    FT_EMU_ROS2_State(FT_RobotM3* M3, FT_EMU_ROS2 *sm_, const char *name = NULL): State(name), robot(M3), sm(sm_){spdlog::debug("Created FT_EMU_ROS2_State {}", name);};
   private:
    void entry(void) final {
        //Actual state entry
        entryCode();
    };
    void during(void) final;
    void exit(void) final {
        exitCode();

        if(stateLogger.isInitialised())
            stateLogger.endLog();
    };

   public:
    virtual void entryCode(){};
    virtual void duringCode(){};
    virtual void exitCode(){};
    void printFullStates(const std::string& extras = "");

   protected:
    FT_EMU_ROS2 *sm;
    LogHelper stateLogger;

};


/**
 * \brief Initialises everthing then waits for a calib command. Set drives in torque control mode.
 *
 */
class M3InitState : public FT_EMU_ROS2_State {

   public:
    M3InitState(FT_RobotM3 * M3, FT_EMU_ROS2 *sm, const char *name = "M3 Init State"):FT_EMU_ROS2_State(M3, sm, name){};
    
    void entryCode(void);
    void duringCode(void);
    void exitCode(void);
};

/**
 * \brief Position calibration of M3. Go to the bottom left stops of robot at constant torque for absolute position calibration. Set drives in torque control mode.
 *
 */
class M3CalibState : public FT_EMU_ROS2_State {

   public:
    M3CalibState(FT_RobotM3 * M3, FT_EMU_ROS2 *sm, const char *name = "M3 Calib"):FT_EMU_ROS2_State(M3, sm, name){};

    void entryCode(void);
    void duringCode(void);
    void exitCode(void);
    bool isCalibDone() {return calibDone;}

   private:
    VM3 qi;
    VM3 stop_reached_time;
    bool at_stop[3];
    bool calibDone=false;
    
};

/**
 * \brief Force Torque calibration of M3. Collects the first 100 samples and sets that as static bias
 *
 */
class M3FTCalibState : public FT_EMU_ROS2_State {

   public:
    M3FTCalibState(FT_RobotM3 * M3, FT_EMU_ROS2 *sm, const char *name = "M3FT Calib"):FT_EMU_ROS2_State(M3, sm, name){};

    void entryCode(void);
    void duringCode(void);
    void exitCode(void);

    bool isCalibDone() {return calibDone;}

   private:
   Eigen::ArrayXXd readings;
    bool calibDone=false;
    int readingCount = 0;
};

/**
 * \brief Start publishing robot state. Provide end-effector mass compensation on M3. 
 * \details Mass is controllable through keyboard inputs. Assumes drives in torque control already.
 */
class M3StandbyPublishState : public FT_EMU_ROS2_State {

   public:
    M3StandbyPublishState(FT_RobotM3 * M3, FT_EMU_ROS2 *sm, const char *name = "M3 Standby Publish"):FT_EMU_ROS2_State(M3, sm, name){};

    void entryCode(void);
    void duringCode(void);
    void exitCode(void);

   private:
    std::vector<double> accel;
    const double transition_t = 1.;        //!< Time to apply progressive transition (no friction comp)
    const double mass_limit = 10;          //!< Maximum applicable mass (+ and -)
    double mass = 0;                       //!< Desired mass to apply: might differ from applied_mass during transition (i.e. setMass)
    double change_mass_rate = 2.;          //!< Rate at which mass will increase/decrease during change mass transition (in kg/s)
    
    Eigen::VectorXd lastWrenches;
};

/**
 * \brief Lock in place: position control around current point. Assumes drives in torque control already.
 *
 */
class M3LockState : public FT_EMU_ROS2_State {

   public:
    M3LockState(FT_RobotM3 * M3, FT_EMU_ROS2 *sm, const char *name = "M3 Lock"):FT_EMU_ROS2_State(M3, sm, name){};

    void entryCode(void);
    void duringCode(void);
    void exitCode(void);

   private:
    VM3 X0;
    double k = 2000.;                //! Impedance proportional gain (spring)
    double d = 3.;                   //! Impedance derivative gain (damper)
};


#endif

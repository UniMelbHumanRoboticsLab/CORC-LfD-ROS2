/**
 * \file EMU_ROS2_States.h
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

class EMU_ROS2; // declare empty class for EMU_ROS2 state machine for forward inclusion

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
//! Current status (0-1 parametrisation and expected position Xd and velocity dXd) for a given min
//!jerk path between X0 and Xf and a given total time T and current time t
double JerkIt(VM3 X0, VM3 Xf, double T, double t, VM3 &Xd, VM3 &dXd);
/** @} */ // end of GenericEMUFunctions


/**
 * \brief EMU trajectory point (i.e. 3d point) with associated reaching time and stop (pause) time
 *
 */
typedef struct M3TrajPt
{
    M3TrajPt (double x, double y, double z, double tf, double tp=.0): X(VM3(x, y, z)), T(tf), Tpause(tp) {};
    VM3 X; //! Point position
    double T; //! Time to reach point
    double Tpause; //! Pause time when reached point
} M3TrajPt;


/**
 * \brief Small data structure holding mass compensations tuning parameters
 *
 */
typedef struct Deweight_s
{
    double Mass = 0;            //!< Current effective mass compensation
    double MassTar = 0;         //!< Current mass target, dsired mass to apply: might differ from applied_mass during transition)
    double MassSet = 1.0;       //!< Expected mass comp to apply when in the appropiate state(cstt for each indiv.)
    const double MassLimit = 10;   //!< Maximum applicable mass (+ and -)
    double VelThresh = 0.1;         //!<Velocity threshold to trigger mass change state
    double AccThresh = 0.05;         //!<Acceleration threshold to trigger mass change state
    double ForceThresh = 0.1;         //!<Force threshold to trigger mass change state
    double MassChangeRate = 4.;     //!< Rate at which mass will increase/decrease during change mass transition (in kg/s)
    int Algorithm = 0;

} Deweight_s;

/**
 * \brief Generic state type for used with M3DemoMachine, providing running time and iterations number: been superseeded by default state, not very much useful anymore.
 *
 */
class EMU_ROS2_State : public State {
   protected:
    FT_RobotM3 * robot;                               //!< Pointer to state machines robot object

    EMU_ROS2_State(FT_RobotM3* M3, EMU_ROS2 *sm_, const char *name = NULL): State(name), robot(M3), sm(sm_){spdlog::debug("Created EMU_ROS2_State {}", name);};
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

   protected:
    EMU_ROS2 *sm;
    LogHelper stateLogger;

};


/**
 * \brief Does nothing waiting for a calib command. Set drives in torque control mode.
 *
 */
class M3NothingState : public EMU_ROS2_State {

   public:
    M3NothingState(FT_RobotM3 * M3, EMU_ROS2 *sm, const char *name = "M3 Do Nothing"):EMU_ROS2_State(M3, sm, name){};

    void entryCode(void) { robot->initTorqueControl(); robot->setJointTorque(VM3(0,0,0)); }
    void duringCode(void) 
    { 
        robot->setJointTorque(VM3(0,0,0)); 
        
        // try to get orientation data
    }
        
    void exitCode(void) { robot->setJointTorque(VM3(0,0,0)); }
};

/**
 * \brief Position calibration of M3. Go to the bottom left stops of robot at constant torque for absolute position calibration. Set drives in torque control mode.
 *
 */
class M3CalibState : public EMU_ROS2_State {

   public:
    M3CalibState(FT_RobotM3 * M3, EMU_ROS2 *sm, const char *name = "M3 Calib"):EMU_ROS2_State(M3, sm, name){};

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
 * \brief Lock in place: position control around current point. Assumes drives in torque control already.
 *
 */
class M3LockState : public EMU_ROS2_State {

   public:
    M3LockState(FT_RobotM3 * M3, EMU_ROS2 *sm, const char *name = "M3 Lock"):EMU_ROS2_State(M3, sm, name){};

    void entryCode(void);
    void duringCode(void);
    void exitCode(void);

   private:
    VM3 X0;
    double k = 1100.;                //! Impedance proportional gain (spring)
    double d = 3.;                   //! Impedance derivative gain (damper)
};


/**
 * \brief Start publishing robot state. Provide end-effector mass compensation on M3. 
 * \details Mass is controllable through keyboard inputs. Assumes drives in torque control already.
 */
class M3StandbyPublishState : public EMU_ROS2_State {

   public:
    M3StandbyPublishState(FT_RobotM3 * M3, EMU_ROS2 *sm, const char *name = "M3 Standby Publish"):EMU_ROS2_State(M3, sm, name){};

    void entryCode(void);
    void duringCode(void);
    void exitCode(void);

    void setMass(double m) {mass=m; std::cout << "Mass: " << mass << std::endl;}

   private:
    std::vector<double> accel;
    const double transition_t = 1.;        //!< Time to apply progressive transition (no friction comp)
    const double mass_limit = 10;          //!< Maximum applicable mass (+ and -)
    double mass = 0;                       //!< Desired mass to apply: might differ from applied_mass during transition (i.e. setMass)
    double applied_mass = 0;               //!< Currently appied mass
    double change_mass_rate = 2.;          //!< Rate at which mass will increase/decrease during change mass transition (in kg/s)
};


/**
 * \brief Generic pt to pt state: to be derived.
 *
 */
class M3PtToPt: public EMU_ROS2_State {

   public:
    M3PtToPt(FT_RobotM3 * M3, EMU_ROS2 *sm, const char *name = "M3 Pt to Pt"):EMU_ROS2_State(M3, sm, name) { };

    virtual void entryCode(void) = 0;
    virtual void duringCode(void) = 0;
    virtual void exitCode(void) = 0;

    void clearPts() {
        trajPts.clear();
        stop=true;
    }

    bool addPt(double x, double y, double z, double tf, double tp=.0) {
        //Lock robot: don't use pts while fed in
        stop=true;
        if(tf<0) {
            return false;
        }
        if(tp<0) {
            return false;
        }
        trajPts.push_back(M3TrajPt(x,y,z,tf, tp));
        //Unlock
        stop=false;

        return true;
    }

    unsigned int getPtsIdx() { return trajPtIdx; }
    double getStatus() { return status; }

   protected:
    double status=0;            //!< Represents the progress (0-1) along the path/trajectory
    bool stop;                  //!< Flag to stop when feeding pts
    bool loop_through_points;
    unsigned int trajPtIdx=0;
    double startTime;
    std::vector<M3TrajPt> trajPts;
    VM3 Xi, Xf;
    double T, Tpause;
    bool paused;
};


/**
 * \brief Path contraint with viscous assistance. Assumes drives in torque control already.
 *
 */
class M3ReproduceState : public M3PtToPt {

   public:
    M3ReproduceState(FT_RobotM3 * M3, EMU_ROS2 *sm, const char *name = "M3 Reproduce"):M3PtToPt(M3, sm, name){};

    void entryCode(void);
    void duringCode(void);
    void exitCode(void);

    short int sign(double val) {return (val>0)?1:((val<0)?-1:0); };

    void setAssistanceLevel(double a) { viscous_assistance = fmax(-30., fmin(30., a)); }

   private:
    double k = 1400;                //! Impedance proportional gain (spring)
    double d = 6.;                  //! Impedance derivative gain (damper)
    double viscous_assistance = 0;  //! Viscous assistance along path
    double veryFirstPt = true;
};


#endif

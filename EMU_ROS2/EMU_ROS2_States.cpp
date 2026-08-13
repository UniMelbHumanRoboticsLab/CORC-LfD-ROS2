#include "EMU_ROS2_States.h"
#include "EMU_ROS2.h"

using namespace std;


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



VM3 impedance(Eigen::Matrix3d K, Eigen::Matrix3d D, VM3 X0, VM3 X, VM3 dX, VM3 dXd) {
    return K*(X0-X) + D*(dXd-dX);
}

double JerkIt(VM3 X0, VM3 Xf, double T, double t, VM3 &Xd, VM3 &dXd) {
    t = std::max(std::min(t, T), .0001); //Bound time
    double tn=std::max(std::min(t/T, 1.0), .0);//Normalised time bounded 0-1
    double tn3=pow(tn,3.);
    double tn4=tn*tn3;
    double tn5=tn*tn4;
    Xd = X0 + ( (X0-Xf) * (15.*tn4-6.*tn5-10.*tn3) );
    dXd = (X0-Xf) * (4.*15.*tn4-5.*6.*tn5-10.*3*tn3)/t;
    return tn;
}

void EMU_ROS2_State::during(void) {
    //Actual state during
    duringCode();

    //Manage state logger if used
    if(stateLogger.isInitialised()) {
        stateLogger.recordLogData();
    }
    sm->get_node()->publish_joint_states();

}

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
    sm->Command = 1;
    sm->Contribution = .0;
    sm->MvtProgress = .0;
    std::cout << "Calibrating (keep clear)..." << std::flush;
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


void M3StandbyPublishState::entryCode(void) {
    //robot->initTorqueControl();
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), false);
    sm->Command = 2;
    sm->Contribution = .0;
    sm->MvtProgress = .0;

    std::cout << "Press S to decrease mass (-100g), W to increase (+100g)." << std::endl;
}
void M3StandbyPublishState::duringCode(void) {
    
    //Bound mass to +-5kg
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
        mass -=0.5;robot->printStatus();
        robot->printJointStatus();
        std::cout << "Mass: " << mass << std::endl;
    }
    if(robot->keyboard->getW()) {
        mass +=0.5;robot->printStatus();
        robot->printJointStatus();
        std::cout << "Mass: " << mass << std::endl;
    }

    // if(iterations()%50==1) {
    //     if(spdlog::get_level()<=spdlog::level::debug) {
    //         double v = robot->getEndEffVelocity().norm();
    //         printProgress(v, "Speed\t", "vFilt="+to_string(v)+"\n");
    //     }
    //     if (sm->UIserver->isCmd("LACC", accel))
    //     {
    //         std::cout << "Acceleration x: " << to_string(accel[0]) << std::endl;
    //         std::cout << "Acceleration y: " << to_string(accel[1]) << std::endl;
    //         std::cout << "Acceleration z: " << to_string(accel[2]) << std::endl;
    //     }
    // }

}
void M3StandbyPublishState::exitCode(void) {
    robot->setEndEffForceWithCompensation(VM3(0,0,sm->MassComp*9.8), false);
}


void M3LockState::entryCode(void) {
    //robot->initTorqueControl();
    robot->setEndEffForceWithCompensation(VM3::Zero(), false);
    //robot->initVelocityControl();
    //robot->setJointVelocity(VM3::Zero());

    sm->Command = 5;
    sm->Contribution = .0;
    sm->MvtProgress = .0;

    X0 = robot->getEndEffPosition();
}
void M3LockState::duringCode(void) {
    //robot->setEndEffVelocity(VM3::Zero());

    //Impedance on point
    Eigen::Matrix3d K = k*Eigen::Matrix3d::Identity();
    Eigen::Matrix3d D = d*Eigen::Matrix3d::Identity();
    VM3 Fd = impedance(K, D, X0, robot->getEndEffPosition(), robot->getEndEffVelocity());
    robot->setEndEffForceWithCompensation(Fd, false);


    /*
    //K tuning
    if(robot->keyboard->getS()) {
        k -= 5;
        std::cout << "K=" << k << " D=" << d<< std::endl;
    }
    if(robot->keyboard->getW()) {
        k += 5;
        std::cout << "K=" << k << " D=" << d<< std::endl;
    }

    //D tuning
    if(robot->keyboard->getD()) {
        d -= 1;
        std::cout << "K=" << k << " D=" << d<< std::endl;
    }
    if(robot->keyboard->getA()) {
        d += 1;
        std::cout << "K=" << k << " D=" << d<< std::endl;
    }*/
}
void M3LockState::exitCode(void) {
    //robot->setEndEffVelocity(VM3::Zero());
    robot->setEndEffForceWithCompensation(VM3::Zero(), false);
}


void M3ReproduceState::entryCode(void) {
    //robot->initTorqueControl();
    robot->setEndEffForceWithCompensation(VM3::Zero(), false);

    sm->Command = 3;
    sm->Contribution = .0;
    sm->MvtProgress = .0;

    trajPtIdx=0;
    Xi=robot->getEndEffPosition();
    if(trajPts.size()>0) {
        Xf=trajPts[trajPtIdx].X;
    }
    else {
        Xf=robot->getEndEffPosition();
    }

    veryFirstPt=true;
    startTime=running();
    paused=false; //not used
    status=.0;
}
void M3ReproduceState::duringCode(void) {
    VM3 X=robot->getEndEffPosition();
    VM3 dX=robot->getEndEffVelocity();
    VM3 Xd=Xi;
    VM3 Fd(0,0,0);
    double status = 0;

    //Normal operation: apply path guidance w/ assistance
    if(!veryFirstPt) {
        //Find closest point on path
        VM3 PathUnitV=(Xf-Xi)/(Xf-Xi).norm();
        Xd = Xi + ( (X-Xi).dot(PathUnitV)*PathUnitV);

        //Progress along path
        status = sign((X-Xi).dot(PathUnitV))*(Xd-Xi).norm()/(Xf-Xi).norm();
        double b=0;
        //Assistance progressive up and down in first and last 10% of movement
        if(status>=0.1 && status<0.9) {
            b = 1.0;
        }
        if(status<0.1) {
            b = status*10.;
        }
        if(status>0.9) {
            b = (1-status)*10.;
        }
        if(status<=0) {
            //"Before" Xi
            Xd = Xi;
            b = 0.;
        }
        if(status>=1) {
            //After "Xf"
            Xd = Xf;
            b = 0.;
        }

        //Velocity vector projection
        VM3 dX_path = dX.dot(PathUnitV)*PathUnitV;
        VM3 dX_ortho = dX - dX_path;

        //Impedance towards path  point with only ortho velocity component
        Eigen::Matrix3d K = k*Eigen::Matrix3d::Identity();
        Eigen::Matrix3d D = d*Eigen::Matrix3d::Identity();
        Fd = impedance(K, D, Xd, X, dX_ortho);

        //Assitive viscosity in path direction
        Fd+=viscous_assistance*b*dX_path;
    }
    //If this is the very first point: bring arm using passive mobilisation
    else {
        VM3 dXd;
        //Compute current desired interpolated point. Arbitrary 2s.
        status = JerkIt(Xi, Xf, 2.0, running()-startTime, Xd, dXd);

        //Impedance on current point
        Eigen::Matrix3d K = k*Eigen::Matrix3d::Identity();
        Eigen::Matrix3d D = d*Eigen::Matrix3d::Identity();
        Fd = impedance(K, D, Xd, X, dX, dXd);
    }

    //Add mass compensation feed-forward
    Fd += VM3(0,0,sm->MassComp*9.8);

    //Apply force with gravity compensation but w/o friction compensation
    robot->setEndEffForceWithCompensation(Fd, false);


    //Have we reached a point? And not currently feeding the pts list
    if(status>=1. && !stop) {
        //Go to next point
        trajPtIdx++;
        if(trajPtIdx>=trajPts.size()){
            trajPtIdx=0;
        }
        /*From where we are
        Xi=robot->getEndEffPosition();*/
        //From last desired point on last path
        Xi=Xd;
        //to next pt (if it exists)
        if(trajPts.size()>0) {
            Xf=trajPts[trajPtIdx].X;
        }
        else {
            Xf=robot->getEndEffPosition();
        }
        veryFirstPt=false;
    }
     sm->MvtProgress = trajPtIdx+status;

    //Display progression
    if(spdlog::get_level()<=spdlog::level::debug) {
        if(iterations()%100==1) {
            printProgress(status, "Path progress: (Point " + to_string(trajPtIdx) + ")", "k="+to_string(k)+"\td="+to_string(d)+"\n");
            robot->printStatus();
        }
    }

    /*For gains tuning only
    if(robot->keyboard->getS()) {
        k -=20;
        std::cout << "k: " << k << std::endl;
    }
    if(robot->keyboard->getW()) {
        k +=20;
        std::cout << "k: " << k << std::endl;
    }
    if(robot->keyboard->getKeyUC()=='D') {
        d -=0.5;
        std::cout << "d: " << d << std::endl;
    }
    if(robot->keyboard->getKeyUC()=='E') {
        d +=0.5;
        std::cout << "d: " << d << std::endl;
    }*/
}
void M3ReproduceState::exitCode(void) {
    robot->setEndEffForceWithCompensation(VM3::Zero());
}



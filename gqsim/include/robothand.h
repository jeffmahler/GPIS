//================================================================================
//         RobotHand: inherited from GSystem
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _SYSTEM_
#define _SYSTEM_

#include <vector>
#include <list>
#include <string>
#include "gear.h"
#include "motor.h"
#include "tinyxml.h"
#include "utilfunc.h" // for mean filter

class RigidBody;

//=============================================================
//                 RobotHand
//=============================================================
class RobotHand: public GSystem
{
public:
	// elements created by new operator (they must be deleted in the destructor)
	std::vector<RigidBody*> _pbodies_new;
	std::vector<GJoint*> _pjoints_new;

	// end-effector
	RigidBody *_pbody_ee;	// the body to which the endeffector is attached
	SE3 _T_ee;				// relative transformation (i.e., position and orientation) of the endeffector frame w.r.t. the endeffector body frame

	// joint coordinates (root joint will not be included here)
	std::vector<GCoordinate*> _pjointcoords;

	// motors driving the joints
	std::vector<Motor> _motors;

	// rendering
	bool _bRendering;

public:
	RobotHand() : _pbody_ee(NULL), _bRendering(true) {}
	~RobotHand() { _delete_elements(); }

	virtual bool getReady();

	virtual void render();	

	//virtual void stepSimulation(double h, double cur_time); // proceed the simulation (h = step size, cur_time = current simulation time)

	virtual void updateKinematics();

	void initSimulation();					// initialize motors
	bool isClosed();						// return true if all fingers have been closed (return true if all motors have been stopped)
	void lockMotors();						// lock motors so that the fingers cannot close further
	void Close(double h, double cur_time);	// close the fingers (h = step size, cur_time = current simulation time)
	void Open(double h);					// open the fingers
	double Move(double h, double t, Vec3 dir, double t0, double d, double vm, double a);	
											// move along the global direction (dir) for distance (d)
											// by applying the velocity obtained from a trepezoidal profile at time t
											// and updating the position using integration with step size h.
											//  - trepezoidal profile is defined by the distance (d), maximum velocity (vm) and acceleration (a)
											//  - t0 = the time at which the profile starts (e.g., if t0=0.5, then the hand moves after 0.5sec pause.)
											// return: tf (the end-time of the profile)
	
	void updateStaticJointTorques();		// compuate joint torques equivalent to the contact forces (updateKinematics() must be called before this)

	void placeHandWithEndEffectorTransform(SE3 Tee); // place the hand so that the end-effector transform (w.r.t. {global}) becomes Tee
	bool setJointValues(const std::vector<double> &q);	// set joint angles (unit: rad) with q, and velocities, accelerations and torques to zeros
	void getJointValues(std::vector<double> &q); // get joint angles
	void getJointValues(std::vector<double> &q, std::vector<double> &dq); // get joint angles and velocities

	SE3 getEndEffectorTransform();			// get end-effector transform w.r.t. {global}

	RigidBody* getBase() { if ( _pbodies_new.size() == 0 ) { return NULL; } return (RigidBody*)_pbodies_new[0]; }

	RigidBody* getBody(std::string name) { for (std::list<GBody*>::iterator iter_pbody = pBodies.begin(); iter_pbody != pBodies.end(); iter_pbody++) { if ( (*iter_pbody)->getName() == name ) return (RigidBody*)(*iter_pbody); } return NULL; }
	std::vector<RigidBody*> getBodies() { std::vector<RigidBody*> pbodies; for (std::list<GBody*>::iterator iter_pbody = pBodies.begin(); iter_pbody != pBodies.end(); iter_pbody++) { pbodies.push_back((RigidBody*)(*iter_pbody)); } return pbodies; }

	void showMotorMessage(bool b) { for (size_t i=0; i<_motors.size(); i++) { _motors[i]._b_show_message = b; } }

	// ------- sub-functions -----------------------------------------------------------------

	void _delete_elements();
};

#endif


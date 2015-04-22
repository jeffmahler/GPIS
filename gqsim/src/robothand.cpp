#include <list>
#include <fstream>
#include <algorithm>
#include "robothand.h"
#include "rigidbody.h"
#include "tinyxml.h"
#include "xmlparser.h"

using namespace std;

static double s_timer = 0.0;

//=============================================================
//                 RobotHand
//=============================================================
bool RobotHand::getReady()
{
	if ( !_pbody_ee ) {
		cerr << "error:: end-effector body not defined in the robot hand!" << endl;
		return false;
	}
	if ( _motors.size() == 0 ) {
		cout << "warning:: no motor defined in the robot hand!" << endl;
	}
	return true;
}

void RobotHand::render()
{
	if ( !_bRendering ) return;
	for (list<GBody*>::iterator iter_pbody = pBodies.begin(); iter_pbody != pBodies.end(); iter_pbody++) {
		(*iter_pbody)->render();
	}
}

void RobotHand::updateKinematics()
{
	GSystem::updateKinematics();
}

void RobotHand::initSimulation()
{
	for (size_t i=0; i<_motors.size(); i++) {
		_motors[i].init();
	}
}

void RobotHand::Close(double h, double cur_time)
{
	for (size_t i=0; i<_motors.size(); i++) {
		_motors[i].close(h, cur_time);
	}
	updateKinematics();
}

void RobotHand::Open(double h)
{
	for (size_t i=0; i<_motors.size(); i++) {
		_motors[i].open(h);
	}
	updateKinematics();
}

bool RobotHand::isClosed()
{
	for (size_t i=0; i<_motors.size(); i++) {
		if ( _motors[i]._closingdirection == 0 )
			continue;
		if ( !_motors[i].isClosingDone() )
			return false;
	}
	return true;
}

void RobotHand::lockMotors()
{
	for (size_t i=0; i<_motors.size(); i++) {
		_motors[i]._is_locked = true;
	}
}

double RobotHand::Move(double h, double t, Vec3 dir, double t0, double d, double vm, double a)
{
	// compute velocity (v) at time t
	double v = 0, tf;
	if ( d > vm*vm/a ) { // trepezoidal velocity profile
		double t1 = t0 + vm/a;
		double t2 = t1 + (d-vm*vm/a)/vm;
		tf = t1 + t2 - t0;
		if ( t >= t0 && t < t1 ) {
			v = a*(t-t0);
		} else if ( t >= t1 && t < t2 ) {
			v = vm;
		} else if ( t >= t2 && t < tf ) {
			v = a*(tf-t);
		}
	} else { // triangular velocity profile
		double t1 = t0 + sqrt(d/a);
		tf = 2*t1 - t0;
		if ( t >= t0 && t < t1 ) {
			v = a*(t-t0);
		} else if ( t >= t1 && t < tf ) {
			v = a*(tf-t);
		}
	}

	// compute the moving direction w.r.t. local base coordinate
	SO3 R = ((GJointFree*)pJoints.front())->T_global.GetRotation();
	Vec3 ldir = ~R * dir;

	// apply the velocity along the direction and update position by integrating it with step size h
	GCoordinate *pcoords[3];
	for (int i=0; i<3; i++) {
		pcoords[i] = &(((GJointFree*)pJoints.front())->translational_joint.coordinates[i]);
		pcoords[i]->dq = v * ldir[i];
		pcoords[i]->q += h * pcoords[i]->dq;
	}

	updateKinematics();
	return tf;
}

void RobotHand::updateStaticJointTorques()
{
	// backward recursion
	for (list<GBody*>::reverse_iterator iter_pbody = pBodies.rbegin(); iter_pbody != pBodies.rend(); iter_pbody++) {
		// update F
		(*iter_pbody)->F.SetZero();
		(*iter_pbody)->F -= (*iter_pbody)->Fe;
		for (list<GBody*>::iterator iter_pbody_child = (*iter_pbody)->pChildBodies.begin(); iter_pbody_child != (*iter_pbody)->pChildBodies.end(); iter_pbody_child++) {
			(*iter_pbody)->F += (*iter_pbody_child)->getTransformed_F();
		}
		// update tau
		(*iter_pbody)->update_tau();
	}
}

void RobotHand::placeHandWithEndEffectorTransform(SE3 Tee)
{
	getBase()->pBaseJoint->set_q(0.0);
	updateKinematics();
	
	SE3 Tb2e = Inv(getBase()->pBaseJoint->T_left) * getEndEffectorTransform();
	getBase()->pBaseJoint->T_left = Tee * Inv(Tb2e);

	// project the rotation matrix onto SO(3)
	SE3 &T = getBase()->pBaseJoint->T_left;
	gReal quat[4];
	iQuat(T.GetRotation(), quat); // rotation matrix --> quaternion (25 multiplications, 1 division)
	T.SetRotation(Quat(quat)); // quaternion --> rotation matrix, this includes projection in the quaternion space (4 multiplications, 3 divisions, 1 sqrt)

	updateKinematics();
}

bool RobotHand::setJointValues(const std::vector<double> &q)
{
	if ( q.size() != _pjointcoords.size() ) {
		cerr << "size mismatch:: q.size() = " << q.size() << " != _pjointcoords.size() = " << _pjointcoords.size() << endl;
		return false;
	}
	for (size_t i=0; i<_pjointcoords.size(); i++) {
		_pjointcoords[i]->q = q[i];
		_pjointcoords[i]->dq = 0;
		_pjointcoords[i]->ddq = 0;
		_pjointcoords[i]->tau = 0;
	}
	updateKinematics();
	return true;
}

void RobotHand::getJointValues(std::vector<double> &q)
{
	q.resize(_pjointcoords.size());
	for (size_t i=0; i<_pjointcoords.size(); i++) {
		q[i] = _pjointcoords[i]->q;
	}
}

void RobotHand::getJointValues(std::vector<double> &q, std::vector<double> &dq)
{
	q.resize(_pjointcoords.size());
	dq.resize(_pjointcoords.size());
	for (size_t i=0; i<_pjointcoords.size(); i++) {
		q[i] = _pjointcoords[i]->q;
		dq[i] = _pjointcoords[i]->dq;
	}
}

SE3 RobotHand::getEndEffectorTransform() 
{ 
	return _pbody_ee->getPoseGlobal(_T_ee); 
}

void RobotHand::_delete_elements()
{
	// main purpose of this is disconnecting joints from the ground
	for (size_t i=0; i<_pjoints_new.size(); i++) {
		_pjoints_new[i]->disconnectBodies();
	}

	// delete the elements created by new operator
	for (size_t i=0; i<_pbodies_new.size(); i++) {
		delete _pbodies_new[i];
	}
	for (size_t i=0; i<_pjoints_new.size(); i++) {
		delete _pjoints_new[i];
	}
	_pbodies_new.clear();
	_pjoints_new.clear();

	// clear the pointer lists
	pCoordinates.clear();
	pBodies.clear();
	pJoints.clear();
}


//================================================================================
//         RIGID OBJECT
// 
//                                                               junggon@gmail.com
//================================================================================

#include <math.h>
#include <assert.h>
#include "rigidobject.h"
#include "glsub.h"

using namespace std;

void RigidObject::setMass(const gReal &mass, const gReal &ixx, const gReal &iyy, const gReal &izz, const gReal &ixy, const gReal &ixz, const gReal &iyz, const SE3 &T_ref)
{
	// I = generalized inertia w.r.t. {ref}
	Inertia I;	
	I.SetMass(mass);
	I.SetInertia(ixx,iyy,izz,ixy,ixz,iyz);

	// _I = generalized inertia w.r.t. {body}
	_I = I.Transform(Inv(T_ref));

	// precompute the inverse of the inertia
	_I.InvToArray(_invI);

	// save center of mass position
	_com.SetZero();
	if ( getMass() > 1E-8 ) {
		_com = ((gReal)1./getMass()) * _I.GetOffDiag();
	}
}

void RigidObject::addMass(const gReal &mass, const gReal &ixx, const gReal &iyy, const gReal &izz, const gReal &ixy, const gReal &ixz, const gReal &iyz, const SE3 &T_ref)
{
	// I = generalized inertia w.r.t. {ref}
	Inertia I;								
	I.SetMass(mass);
	I.SetInertia(ixx,iyy,izz,ixy,ixz,iyz);
	
	// I_b = generalized inertia w.r.t. {body}
	Inertia I_b = I.Transform(Inv(T_ref));	

	// I += I_b
	for (int i=0; i<6; i++) { _I._I[i] += I_b._I[i]; }
	for (int i=0; i<3; i++) { _I._r[i] += I_b._r[i]; }
	_I._m += I_b._m;

	// precompute the inverse of the inertia
	_I.InvToArray(_invI);

	// save center of mass position
	_com.SetZero();
	if ( getMass() > 1E-8 ) {
		_com = ((gReal)1./getMass()) * _I.GetOffDiag();
	}
}

bool RigidObject::addSurfaceTriData(const std::vector<double> &x, const std::vector<int> &f, Vec3 scale, SE3 T_ref, bool bcol, bool brender)
{
	// create a surface attached to the body coordinate frame
	RigidSurface *psurf = new RigidSurface(&_T, &_V, &_F);
	psurf->_parentbodyname = this->getName();

	// load surface mesh data
	if ( !psurf->loadFromDataTri(x, f, scale, T_ref) ) {
		return false;
	}

	// collision, rendering
	psurf->enableCollision(bcol);
	psurf->enableRendering(brender);

	// add the surface pointer to _pSurfs
	_pSurfs.push_back(psurf);

	return true;
}

bool RigidObject::getReady()
{
	// if mass and inertia have not been set, set default mass and inertia
	if ( isDynamic() && getMass() < 1E-12 ) {
		setMass(1.0, 0.01/6., 0.01/6., 0.01/6., 0, 0, 0); // cubic box with mass = 1kg and width = depth = height = 0.1m
		cout << "warning:: mass and inertia have not been set for the dynamic object (" << getName() << "):: default values applied." << endl;
	}

	// get surfaces ready
	for (size_t i=0; i<_pSurfs.size(); i++) {
		if ( !_pSurfs[i]->getReady() ) {
			cerr << "error:: failed in getting surface ready" << endl;
			return false;
		}
	}

	return true;
}

void RigidObject::render()
{
	if ( !_bRendering ) 
		return;
	
	for (int i=0; i<(int)_pSurfs.size(); i++) {
		_pSurfs[i]->render();
	}
}

void RigidObject::setGlobalForceAtGlobalPosition(const Vec3 &fg, const Vec3 &pg)
{
	_F = dAd(_T, dse3(Cross(pg, fg), fg));
}

void RigidObject::setGlobalForceAtLocalPosition(const Vec3 &fg, const Vec3 &pl)
{
	Vec3 fl = ~(_T.GetRotation()) * fg; // force w.r.t. {body}
	_F = dse3(Cross(pl, fl), fl);
}

void RigidObject::setLocalForceAtLocalPosition(const Vec3 &fl, const Vec3 &pl)
{
	_F = dse3(Cross(pl, fl), fl);
}

void RigidObject::addGlobalForceAtGlobalPosition(const Vec3 &fg, const Vec3 &pg)
{
	_F += dAd(_T, dse3(Cross(pg, fg), fg));
}

void RigidObject::addGlobalForceAtLocalPosition(const Vec3 &fg, const Vec3 &pl)
{
	Vec3 fl = ~(_T.GetRotation()) * fg; // force w.r.t. {body}
	_F += dse3(Cross(pl, fl), fl);
}

void RigidObject::addLocalForceAtLocalPosition(const Vec3 &fl, const Vec3 &pl)
{
	_F += dse3(Cross(pl, fl), fl);
}

void RigidObject::stepSimulation(gReal h)
{
	if ( _bStatic ) return;

	// dVdt = _invI * ( _F + dad(_V, _I * _V) )
	se3 dVdt;
	dse3 F = _F;
	F += dad(_V, _I * _V);
	matSet_multAB(dVdt.GetArray(), _invI, F.GetArray(), 6, 6, 6, 1);

	// update _V
	_V += h * dVdt;

	// update _T
	//SO3 R = Exp(h*_V.GetW()); 
	//SE3 dT(R, h*_V.GetV()); // SE3(eye, h*v) * SE3(exp(h*w), 0) where (w,v) = _V
	//SE3 dT(R, h*(R*_V.GetV())); // SE3(exp(h*w), 0) * SE3(eye, h*v) where (w,v) = _V
	SE3 dT = Exp(h*_V); 
	_T *= dT;

	// compensate roundoff error in the orientation by projecting in quaternion space
	gReal quat[4];
	iQuat(_T.GetRotation(), quat); // rotation matrix --> quaternion (25 multiplications, 1 division)
	_T.SetRotation(Quat(quat)); // quaternion --> rotation matrix, this includes projection in the quaternion space (4 multiplications, 3 divisions, 1 sqrt)

	// update bounding boxes of the surfaces for collision check
	updateBoundingBox();

	// filter the velocity
	for (int i=0; i<6; i++) {
		_filter_V[i].pushValue(_V[i]);
	}
}

void RigidObject::initSimulation() 
{ 
	_V.SetZero();
	_F.SetZero();
	for (int i=0; i<6; i++) { 
		_filter_V[i].clearBuffer(); 
	} 
}

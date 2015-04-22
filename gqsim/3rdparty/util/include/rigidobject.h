//================================================================================
//         RIGID OBJECT
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _RIGID_OBJECT_
#define _RIGID_OBJECT_

#include <vector>
#include "gear.h"
#include "rigidsurface.h"
#include "utilfunc.h"

class RigidObject : public GElement
{
public:
	RigidObject() : _V(0,0,0,0,0,0), _F(0,0,0,0,0,0), _bCollision(true), _bRendering(true), _com(0,0,0) { matSet_zero(_invI, 36); _filter_V.resize(6); for (int i=0; i<6; i++) { _filter_V[i].setBufferSize(100); } }
	~RigidObject() { for (int i=0; i<(int)_pSurfs.size(); i++) { delete _pSurfs[i]; } }

	// set/add mass and inertia
	void setMass(const gReal &mass_, const gReal &ixx_, const gReal &iyy_, const gReal &izz_, const gReal &ixy_, const gReal &ixz_, const gReal &iyz_, const SE3 &T_ref_ = SE3());
	void addMass(const gReal &mass_, const gReal &ixx_, const gReal &iyy_, const gReal &izz_, const gReal &ixy_, const gReal &ixz_, const gReal &iyz_, const SE3 &T_ref_ = SE3());

	// adds a triangular surface from serialized data
	bool addSurfaceTriData(const std::vector<double> &x, const std::vector<int> &f, Vec3 scale = Vec3(1,1,1), SE3 T_ref = SE3(), bool bcol = true, bool brender = true);

	// access to surfaces
	RigidSurface* getSurface(int idx) { if ( idx >= 0 && idx < (int)_pSurfs.size() ) { return _pSurfs[idx]; } else { return NULL; } }
	RigidSurface* getSurfaceLast() { if ( _pSurfs.size() > 0 ) { return _pSurfs[_pSurfs.size()-1]; } else { return NULL; } }
	std::vector<RigidSurface*> &getSurfaces() { return _pSurfs; }

	// collision
	void enableCollision(bool b_) { _bCollision = b_; }
	bool isEnabledCollision() { return _bCollision; }

	// rendering
	void enableRendering(bool b) { _bRendering = b; }
	bool isEnabledRendering() { return _bRendering; }
	
	void render();

	// init
	bool getReady();

	// get state
	SE3 &getPose() { return _T; }
	Vec3 getPosition() { return _T.GetPosition(); }
	SO3 getOrientation() { return _T.GetRotation(); }
	se3 getVelocity() { return Ad(_T, _V); }
	se3 getFilteredVelocity() { se3 Vf; for (int i=0; i<6; i++) { Vf[i] = _filter_V[i].getValue(); } return Ad(_T, Vf); }

	// set state
	void setPose(const SE3 &T) { _T = T; updateBoundingBox(); }
	void setPosition(const Vec3 &p) { _T.SetPosition(p); updateBoundingBox(); }
	void setOrientation(const SO3 &R) { _T.SetRotation(R); updateBoundingBox(); }
	void updateBoundingBox() { for (size_t i=0; i<_pSurfs.size(); i++) { _pSurfs[i]->updateBoundingBox(); } }

	// get mass property
	gReal getMass() { return _I.GetMass(); }
	Vec3 &getCOM() { return _com; }

	// set forces
	void initForce() { _F.SetZero(); }
	void setGlobalForceAtGlobalPosition(const Vec3 &fg, const Vec3 &pg); // fg = force w.r.t. {global}, pg = position w.r.t. {global}
	void setGlobalForceAtLocalPosition(const Vec3 &fg, const Vec3 &pl);  // fg = force w.r.t. {global}, pl = position w.r.t. {body}
	void setLocalForceAtLocalPosition(const Vec3 &fl, const Vec3 &pl);   // fl = force w.r.t. {body}, pl = position w.r.t. {body}
	void addGlobalForceAtGlobalPosition(const Vec3 &fg, const Vec3 &pg);
	void addGlobalForceAtLocalPosition(const Vec3 &fg, const Vec3 &pl);
	void addLocalForceAtLocalPosition(const Vec3 &fl, const Vec3 &pl);
	void addGlobalMoment(const Vec3 &mg) { _F += dse3(~(_T.GetRotation())*mg, Vec3(0,0,0)); }	// mg = moment w.r.t. {global}
	void addLocalMoment(const Vec3 &ml) { _F += dse3(ml, Vec3(0,0,0)); }						// ml = moment w.r.t. {body}
	void addGravityForce(const Vec3 &g) { addGlobalForceAtLocalPosition(getMass()*g, getCOM()); }  // g = direction and magnitude of the gravity force (w.r.t. {global})

	// set dynamic or static
	void setDynamic(bool b) { _bStatic = !b; }
	void setStatic(bool b) { _bStatic = b; }
	bool isDynamic() { return !_bStatic; }
	bool isStatic() { return _bStatic; }

	// dynamics simulation
	void stepSimulation(gReal h);	// update state (_T and _V) at the next time step (h = step size in second) 
									// with current body force and inertia (_F and _I).

	// init simulation
	void initSimulation();

public:
	// dynamics
	SE3 _T;				// position and orientation of the body coordinate frame w.r.t. {ground}
	se3 _V;				// generalized body velocity w.r.t. {body}, _V = Inv(_T) * dT/dt
	dse3 _F;			// generalized body force w.r.t. {body} (This includes the forces by the gravity and contacts.)
	Inertia _I;			// generalized inertia w.r.t. {body}
	gReal _invI[36];	// inverse of the generalized inertia (This will be precomputed in setMass() or addMass().)
	Vec3 _com;			// center of mass position w.r.t. {body} (This will be precomputed in setMass() or addMass().)

	// static or dynamic
	bool _bStatic;		// set true if the object is static (Static body will never move.)

	// surfaces
	std::vector<RigidSurface*> _pSurfs;	// array of pointers to the surfaces (for both rendering and collision)

	// collision
	bool _bCollision;					// indicates if the rigid body is collidable or not
	
	// rendering
	bool _bRendering;					// set true to render the body

	// filter
	std::vector<MovingAverageFilter> _filter_V;	// the buffers are cleared in initSimulation()
};

#endif


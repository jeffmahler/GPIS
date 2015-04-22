//================================================================================
//         RIGID BODY INHERITED FROM GBODY
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _RIGIDBODY_
#define _RIGIDBODY_

#include <vector>
#include "gear.h"
#include "rigidsurface.h"

class RigidBody: public GBody
{
public:
	RigidBody() : bCollision(true), bRendering(true) {}
	~RigidBody() { for (int i=0; i<(int)pSurfs.size(); i++) { delete pSurfs[i]; } }

	// adds a triangular surface from serialized data
	bool addSurfaceTriData(const std::vector<double> &x, const std::vector<int> &f, Vec3 scale = Vec3(1,1,1), SE3 T_ref = SE3(), bool bcol = true, bool brender = true);

	// adds a set of surface vertices from serialized data
	bool addSurfaceVtxData(const std::vector<double> &x, const std::vector<double> &n, Vec3 scale = Vec3(1,1,1), SE3 T_ref = SE3(), bool bcol = true, bool brender = true);

	// surface
	RigidSurface* getSurface(int idx) { if ( idx >= 0 && idx < (int)pSurfs.size() ) { return pSurfs[idx]; } else { return NULL; } }
	RigidSurface* getSurfaceLast() { if ( pSurfs.size() > 0 ) { return pSurfs[pSurfs.size()-1]; } else { return NULL; } }
	std::vector<RigidSurface*> &getSurfaces() { return pSurfs; }

	// collision
	void enableCollision(bool b_) { bCollision = b_; }
	bool isEnabledCollision() { return bCollision; }

	// rendering
	void enableRendering(bool b) { bRendering = b; }
	bool isEnabledRendering() { return bRendering; }

public: // virtual functions
	bool getReady();
	void update_T();
	void render();

public:
	std::vector<RigidSurface*> pSurfs;	// array of pointers to the surfaces (for both rendering and collision)
	bool bCollision;					// indicates if the rigid body is collidable or not
	bool bRendering;					// set true to render the body
};

#endif


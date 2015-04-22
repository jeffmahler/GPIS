//================================================================================
//         RIGID SURFACE : SURFACE ATTACHED TO A RIGID BODY
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _RIGID_SURFACE_
#define _RIGID_SURFACE_

#include <vector>
#include "liegroup.h"
#include "surfmesh.h"
#include "boundingbox.h"

class RigidBody;

class RigidSurface: public SurfaceMesh
{
public:
	class SurfacePatch
	{
	public:
		std::vector<int> _idxFaces;	// indices of the faces
		std::vector< std::pair<int, int> > _idxEdgeVertices; // indices of the edge vertex pair
		std::vector< std::pair<int, int> > _idxEdgeFaces; // indices of the edge face pair
	};

	enum ContactFrictionType { FRICTION_STATIC, FRICTION_DYNAMIC, FRICTION_NONE };

	// reference coordinate frame to which the surface is attached (usually, this represents a rigid body)
	SE3 *_pT;					// pointer to a reference coordinate frame (*_pT) = SE3: {ground} --> {ref} to compute the global positions of the vertices.
	se3 *_pV;					// pointer to the frame velocity (generalized body velocity w.r.t. the frame) to consider the collision velocity. (Set NULL to ignore the collision velocity.)
	dse3 *_pF;					// pointer to the frame force (generalized body force w.r.t. the frame) to which contact forces will be applied. (Set NULL to ignore the contact force transmission.)
	std::string _parentbodyname; // the name of the body that the surface is attached to

	// for collision checking
	bool _bCollision;			// set true if the surface is collidable
	BoundingBox _obb;			// bounding box
	Vec3 _pos_obb_local;		// position of the bounding box w.r.t. {ref}
	double _col_depth_limit;	// collision depth(>=0) under the surface (this will be used to ignore too deep penetration in checking collision.)
	std::vector<SurfacePatch> _surfPatch; // surface patches of the vertices
	std::vector<int> _bColSurfPatch;
	std::vector<int> _bColSeedVertex;

	// for contact force computation
	std::vector<int> _bContact;			// contact or not at current step : 1(contact), 0(no contact)
	std::vector<int> _bContactPrev;		// contact or not at previous step : 1(contact), 0(no contact)
	std::vector<Vec3> _xf_ref;			// reference contact positions w.r.t. the contact face's local coordinate frame (for computing static friction force)
	std::vector<Vec3> _fc;				// contact forces (for rendering only)
	std::vector<int> _bStaticFriction;	// static friction or not at current step : 1(static), 0(dynamic) (for rendering only)

	// list of contacts
	std::vector<int> _collidingVertexIndices;		// list of colliding vertices (if there is no contact, _collidingVertexIndices.size() == 0 )
	std::vector<int> _collidingSurfPatchIndices;	// list of colliding surface patches (if there is no contact, _collidingSurfPatchIndices.size() == 0)
	std::vector<int> _seedVertexIndices;			// list of seed vertex indices for collision

	// rendering option
	bool _bRenderingBoundingBox;				// set true to render bounding box
	bool _bRenderingContactPoints;				// set true to render contact points
	bool _bRenderingContactForces;				// set true to render contact forces acting on the contact points
	bool _bRenderingContactForcesReversed;		// set true to render contact forces in opposite direction 
	bool _bRenderingCollidingSurfacePatches;	// set true to render colliding surface patches
	double _force_scale;						// scale for rendering force

public:
	RigidSurface() : _pT(NULL), _pV(NULL), _pF(NULL), _bCollision(true), _col_depth_limit(0.01), _bRenderingBoundingBox(false), _bRenderingContactPoints(false), _bRenderingContactForces(false), _bRenderingContactForcesReversed(false), _bRenderingCollidingSurfacePatches(false), _force_scale(0.002) {}
	RigidSurface(SE3 *pT, se3 *pV, dse3 *pF) : _pT(pT), _pV(pV), _pF(pF), _bCollision(true), _col_depth_limit(0.01), _bRenderingBoundingBox(false), _bRenderingContactPoints(false), _bRenderingContactForces(false), _bRenderingContactForcesReversed(false), _bRenderingCollidingSurfacePatches(false), _force_scale(0.002) {}
	~RigidSurface() {}

	// virtual functions
	bool getReady();
	void render();

	// attach the surface to a reference coordinate frame (or a rigid body)
	void attachToFrame(SE3 *pT, se3 *pV = NULL, dse3 *pF = NULL) { _pT = pT; _pV = pV; _pF = pF; }

	// set the surface collidable
	void enableCollision(bool b) { _bCollision = b; }
	bool isEnabledCollision() { return _bCollision; }

	// set collision depth
	void setCollisionDepthLimit(double d) { _col_depth_limit = d; }
	void setCollisionDepthLimitRatio(double r); // _col_depth_limit = r * min(_obb.extents[0],_obb.extents[1],_obb.extents[2])
	double getCollisionDepthLimit() { return _col_depth_limit; }

	// reset contact info
	void resetContactInfo();

	// copy _bContact into _bContactPrev
	void savePrevContactInfo() { _bContactPrev = _bContact; } 

	// bounding box
	void buildBoundingBox(double margin_ = 0); // set the bounding box automatically using the vertex positions (margin_ will be added to obb.extents[])
	void updateBoundingBox();
	BoundingBox &getBoundingBox() { return _obb; }

	// rendering options
	void enableRenderingContactPoints(bool b) { _bRenderingContactPoints = b; }
	void enableRenderingContactForces(bool b) { _bRenderingContactForces = b; }
	void enableRenderingBoundingBox(bool b) { _bRenderingBoundingBox = b; }
	void enableRenderingCollidingSurfacePatches(bool b) { _bRenderingCollidingSurfacePatches = b; }
	bool isEnabledRenderingContactPoints() { return _bRenderingContactPoints; }
	bool isEnabledRenderingContactForces() { return _bRenderingContactForces; }
	bool isEnabledRenderingBoundingBox() { return _bRenderingBoundingBox; }

	// sub-functions for collision checking
	void _scanSurfacePatches();
	bool _checkCollisionWithSurfacePatch(double &pdepth, Vec3 &contactnormal, int &idxface, int idxsurfpatch, const Vec3 &p, const Vec3 &np = Vec3(0,0,0));	
																									// check if a vertex is colliding with a surface patch
																									// input: idxsurfpatch = index of the surface patch (same as the index of the center vertex of the surface patch)
																									//        p = vertex position w.r.t. {surface}
																									//        np = vertex normal w.r.t. {surface} (set zero vector not to consider vertex normal)
																									// output: pdepth = penetration depth
																									//         contactnormal = contact normal vector w.r.t. {surface}
																									//         idxface = index of the surface face colliding with the vertex p
	
	void _highlightSurfacePatch(int vidx, Vec3 c = Vec3(1,0,0), DrawType dt=DT_SOLID);

	bool _checkCollisionFully(double &pdepth, Vec3 &normal, const Vec3 &p);							// check if p is colliding with the surface (all faces will be tested)
																									// input: p (w.r.t. {surface})
																									// output: pdepth = penetration depth, normal = contact normal vector

};


#endif


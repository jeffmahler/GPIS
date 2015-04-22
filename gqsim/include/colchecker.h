#ifndef _COLLISION_CHECKER_
#define _COLLISION_CHECKER_

#include <vector>
#include "liegroup.h"

class RigidSurface;

class CollisionChecker
{
public:

	// contact parameters for computing frictional contact forces ("stable penalty-based model of frictional contacts" by Katsu Yamane, ICRA, 2006)
	class ContactParam
	{
	public:
		double _Kp, _Kd;		// parameters for contact normal force computation (elasticity and viscosity)
		double _Kfp, _Kfd;		// parameters for friction force computation
		double _mu_s, _mu_d;	// static and dynamic Coulomb friction coefficients
		double _k;				// coefficient of the weight function

		ContactParam() : _Kp(1E5), _Kd(1E4), _Kfp(2000), _Kfd(1400), _mu_s(0.5), _mu_d(0.4), _k(0.1/0.001) {}
		ContactParam(double Kp, double Kd, double Kfp, double Kfd, double mu_s, double mu_d, double k) : _Kp(Kp), _Kd(Kd), _Kfp(Kfp), _Kfd(Kfd), _mu_s(mu_s), _mu_d(mu_d), _k(k) {}
	};

	class CollisionInfo 
	{
	public:
		// collision information
		RigidSurface *_psurfA, *_psurfB;	// a vertex in surface A is colliding with surface B
		int _idxVertexA;					// index of the colliding vertex in surface A vertices
		int _idxNearestVertexB;				// index of the nearest vertex in surface B
		Vec3 _pos;							// global position of the collision
		Vec3 _normal;						// global normal vector (contact force along the normal will apply to the vertex)
		gReal _penetration_depth;			// penetration depth
		int _idxFaceB;						// index of the face in surface B colliding with the vertex

		// constructor/destructor
		CollisionInfo(RigidSurface *psa, RigidSurface *psb, int idxva, int idxvb, const Vec3 &p, const Vec3 &n, gReal d, int idxfb) : _psurfA(psa), _psurfB(psb), _idxVertexA(idxva), _idxNearestVertexB(idxvb), _pos(p), _normal(n), _penetration_depth(d), _idxFaceB(idxfb) {}
		~CollisionInfo() {}
	};

	class CollidableSurfacePair
	{
	public:

		// collidable surface pair
		RigidSurface *_psurfA, *_psurfB;

		// type of collision check
		bool _bBidirectional;	// bBidirectional = true  : vertices of psurfA <--> faces of psurfB, vertices of psurfB <--> faces of psurfA
								// bBidirectional = false : vertices of psurfA <--> faces of psurfB

		bool _bReusePrevInfo;	// set true to use previous collision information for fast collision check when there is little change in the relative pose between the two surfaces

		bool _bAveragedNormal;	// set true to use averaged normal (face normal of psurfB and negative vertex normal of psurfA)
		
		// contact parameter
		ContactParam _cp;

		// internal variables for collision check
		std::vector<int> _idxActiveVertexA, _idxActiveVertexB;	// indices of the active vertices in surface A/B which are possibly colliding with the surface B/A
		std::vector<int> _idxNearestVertexB, _idxNearestVertexA; // indices of the nearest vertices of the active vertices in surface B/A
		SE3 _Tab_prev; // position and orientation of surface B w.r.t. surface A at a previous time step

		// collision check final output
		std::vector<CollisionInfo> _cols;

		// constructor/destructor
		CollidableSurfacePair(RigidSurface *psa, RigidSurface *psb, ContactParam cp = ContactParam(), bool bbidir = true, bool breuseprevinfo = true, bool baveragednormal = false);
		~CollidableSurfacePair() {}

		// check collision and save the output to _cols
		void checkCollision();

		// apply frictional contact forces to the surfaces
		void applyContactForces();

		// subfunctions
		void _checkCollisionFast();
	};


	CollisionChecker() {}
	~CollisionChecker() {}

	size_t getNumCollisionSurfacePairs() { return _colsurfpairs.size(); }
	std::vector<CollidableSurfacePair> & getCollisionSurfacePairs() { return _colsurfpairs; }

	int getNumSurfaces() { return (int)_psurfs.size(); }
	std::vector<RigidSurface*> & getSurfaces() { return _psurfs; }

	void clearCollisionSurfacePairs() { _colsurfpairs.clear(); _psurfs.clear(); }
	void addCollisionSurfacePair(RigidSurface *psa, RigidSurface *psb, ContactParam cp, bool bidir, bool breuseprevinfo, bool baveragednormal);

	void checkCollision(); // check collision in the surface pairs (_colsurfpairs)
	void applyContactForces();

protected:
	std::vector<CollidableSurfacePair> _colsurfpairs;
	std::vector<RigidSurface*> _psurfs; // unrepeated surface list in the collision surface pairs
};

#endif


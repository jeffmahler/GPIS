//================================================================================
//         BOUNDING BOX
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _BOUNDING_BOX_
#define _BOUNDING_BOX_

#include "liegroup.h"

class BoundingBox
{
public:
	double extents[3];	// extents of the box (half-length)
	Vec3 position;		// global position of the box center
	SO3 orientation;	// global orientation of the box

	BoundingBox() : position(0,0,0) { extents[0] = extents[1] = extents[2] = 0; }
	BoundingBox(double e[], const Vec3 &p, const SO3 &R) : position(p), orientation(R) { extents[0] = e[0]; extents[1] = e[1]; extents[2] = e[2]; }
	~BoundingBox() {}
};

#endif


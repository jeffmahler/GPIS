//================================================================================
//         FUNCTIONS FOR COLLISION DETECTION
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _FUNCTIONS_FOR_COLLISION_DETECTION_
#define _FUNCTIONS_FOR_COLLISION_DETECTION_

#include "liegroup.h"


bool dcPointTriangle(gReal &d, Vec3 &n, const Vec3 &x, const Vec3 &t0, const Vec3 &t1, const Vec3 &t2);
// ======================================================================================
// bool dcPointTriangle(d, n, x, t0, t1, t2)
// --------------------------------------------------------------------------------------
// Detects whether a point x is located inside of a triangular prism defined by {t0, t1, t2}.
// outputs: 
//		return = true if the test point can be projected on the triangle.
//		d = signed distance between the point and the triangle plane. (d > 0 if the point is located above the plane.)
//		n = normal vector to the triangle plane. The right hand rule for {t0, t1, t2} will be applied to determine n.
// inputs:
//		x = The location of the test point
//		{t0, t1, t2} = Triplet defining the triangle plane
// ======================================================================================

bool dcPointTriangle2(gReal &d, const Vec3 &x, const Vec3 &t0, const Vec3 &t1, const Vec3 &t2, const Vec3 &n);
// ======================================================================================
// bool dcPointTriangle2(d, x, t0, t1, t2, n)
// --------------------------------------------------------------------------------------
// Detects whether a point x can be projected on a triangular prism defined by {t0, t1, t2}.
// outputs: 
//		return = true if the test point can be projected on the triangle.
//		d = signed distance between the point and the triangle plane. (d > 0 if the point is located above the plane.)
// inputs:
//		x = The location of the test point
//		{t0, t1, t2} = Triplet defining the triangle plane
//		n = precomputed normal vector of the triangle. n = cross(t1-t0, t2-t0)
// ======================================================================================

bool dcPointTriangle(gReal &d, Vec3 &n, const Vec3 &x, const Vec3 &t0, const Vec3 &t1, const Vec3 &t2, gReal d_lower, gReal d_upper = 0.0);
// ======================================================================================
// bool dcPointTriangle(d, n, x, t0, t1, t2, d_lower, d_upper)
// --------------------------------------------------------------------------------------
// Detects whether a point x is located within a prism defined by a triangle (t0, t1, t2) and height range (d_lower, d_upper).
// outputs: 
//		return = true if the test point is located within the prism.
//		d = signed distance between the point and the triangle plane. (d > 0 if the point is located above the plane.)
//		n = normal vector to the triangle plane. The right hand rule for {t0, t1, t2} will be applied to determine n.
// inputs:
//		x = The location of the test point
//		{t0, t1, t2} = Triplet defining the triangle plane
//		[d_lower, d_upper] = The range of d for contact. ex) [-0.1, 0.0]
// ======================================================================================

bool detectCollision(gReal &pd, gReal &s, gReal &alpha, gReal &beta, Vec3 &x_collision, Vec3 &n, const Vec3 &x_present, const Vec3 &x_previous, const Vec3 &t0, const Vec3 &t1, const Vec3 &t2);
// ======================================================================================
// bool detectCollision(pd, s, x_collision, n, x_present, x_previous, t0, t1, t2)
// --------------------------------------------------------------------------------------
// Detects whether the line trajectory(from x_previous to x_present) collides to the inside of the triangle plane defined by {t0, t1, t2}.
// outputs: 
//		return = true if collision happens
//		pd = penetration depth
//		s = distance ratio: (x_collision, x_previous) / (x_present, x_previous)
//		(alpha, beta) = local 2-dimensional coordinates of x_collision, alpha >= 0, beta >= 0, alpha+beta <= 1
//		x_collision = the intersectional point of the line trajectory and the plane
//		n = normal vector to the triangle plane. The right hand rule for {t0, t1, t2} will be applied to determine n.
// inputs:
//		x_present = The present position of the test point
//		x_previous = The previous position of the test point
//		{t0, t1, t2} = Triplet defining the triangle plane
// ======================================================================================

bool dcBox2d(gReal *x, gReal *y, gReal *X, gReal *Y);
// ======================================================================================
// bool dcBox2d(gReal *x, gReal *y, gReal *X, gReal *Y)
// --------------------------------------------------------------------------------------
// Detects whether the two plane boxes(rectangles or parallelograms) are colliding with each other. 
// output:
//		return = true if collision happens.
// inputs:
//		(x[i], y[i]), (X[i], Y[i]) are the corner points of the two boxes in 2D. (i=0,1,2,3)
//		The index ordering of the points should be clock or counter-clock wise.
//			
//			  (x[0],y[0])  ---------------- (x[3],y[3])         (x[0],y[0])	----------------- (x[1],y[1])
//						  /				 /               or     		   / 	           /
//						 /			    /	                     		  /		          /
//			(x[1],y[1])	---------------- (x[2],y[2])         (x[3],y[3]) ----------------- (x[2],y[2])

bool dcOBBOverlap(gReal *Ea, gReal *Pa, gReal *Ra, gReal *Eb, gReal *Pb, gReal *Rb);
// ======================================================================================
// bool dcOBBOverlap(gReal *Ea, gReal *Pa, gReal *Ra, gReal *Eb, gReal *Pb, gReal *Ra)
// --------------------------------------------------------------------------------------
// Detects whether two Oriented Bounding Boxes (OBBs) are colliding with each other. 
// output:
//		return = true if collision happens.
// inputs:
//		OBB A = (Ea, Pa, Ra), OBB B = (Eb, Pb, Rb) where
//		(Ea[0],Ea[1],Ea[2]) = extents
//		(Pa[0],Pa[1],Pa[2]) = position
//		[Ra[0] Ra[3] Ra[6]]
//		[Ra[1] Ra[4] Ra[7]] = orientation
//		[Ra[2] Ra[5] Ra[8]]

bool dcTriOBBOverlap(gReal *v0, gReal *v1, gReal *v2, gReal *Ea, gReal *Pa, gReal *Ra);
// ======================================================================================
// bool dcTriOBBOverlap(gReal *v0, gReal *v1, gReal *v2, gReal *Ea, gReal *Pa, gReal *Ra)
// --------------------------------------------------------------------------------------
// Detects whether a triangle and an Oriented Bounding Box (OBB) are colliding with each other. 
// output:
//		return = true if collision happens.
// inputs:
//		triangle = (v0, v1, v2) where vi = gReal[3] = i-th vertex position of the triangle
//		OBB A = (Ea, Pa, Ra) where
//		(Ea[0],Ea[1],Ea[2]) = extents
//		(Pa[0],Pa[1],Pa[2]) = position
//		[Ra[0] Ra[3] Ra[6]]
//		[Ra[1] Ra[4] Ra[7]] = orientation
//		[Ra[2] Ra[5] Ra[8]]

bool dcPointOBBOverlap(gReal *v, gReal *Ea, gReal *Pa, gReal *Ra);
// ======================================================================================
// bool dcPointOBBOverlap(gReal *v, gReal *Ea, gReal *Pa, gReal *Ra)
// --------------------------------------------------------------------------------------
// Detects whether a point and an Oriented Bounding Box (OBB) are colliding with each other. 
// output:
//		return = true if collision happens.
// inputs:
//		v = position of the point
//		OBB A = (Ea, Pa, Ra) where
//		(Ea[0],Ea[1],Ea[2]) = extents
//		(Pa[0],Pa[1],Pa[2]) = position
//		[Ra[0] Ra[3] Ra[6]]
//		[Ra[1] Ra[4] Ra[7]] = orientation
//		[Ra[2] Ra[5] Ra[8]]


#endif


//================================================================================
//         FUNCTIONS FOR COLLISION DETECTION
// 
//                                                               junggon@gmail.com
//================================================================================

#include "cdfunc.h"

bool triBoxOverlap(gReal boxcenter[3],gReal boxhalfsize[3],gReal trivert0[3],gReal trivert1[3],gReal trivert2[3]);

bool dcPointTriangle(gReal &d, Vec3 &n, const Vec3 &x, const Vec3 &t0, const Vec3 &t1, const Vec3 &t2)
{
	Vec3 e1, e2, e3, x_t0;

	x_t0 = x-t0;

	e1 = t1-t0;
	e2 = t2-t0;
	
	// n = normal vector to the triangle plane with the right hand rule
	n = Cross(e1, e2);
	n.Normalize();

	// d = signed distance
	d = Inner(n, x_t0);

	// checks if x is inside of the trigonal prism
	if ( Inner(x_t0, Cross(n, e1)) < 0 ) return false;
	if ( Inner(x_t0, Cross(e2, n)) < 0 ) return false;
	if ( Inner(x-t1, Cross(n, t2-t1)) < 0 ) return false;

    return true;
}

bool dcPointTriangle2(gReal &d, const Vec3 &x, const Vec3 &t0, const Vec3 &t1, const Vec3 &t2, const Vec3 &n)
{
	Vec3 x_t0 = x-t0;

	// d = signed distance
	d = Inner(n, x-t0);

	// checks if x is inside of the trigonal prism
	if ( Inner(x_t0, Cross(n, t1-t0)) < 0 ) return false;
	if ( Inner(x_t0, Cross(t2-t0, n)) < 0 ) return false;
	if ( Inner(x-t1, Cross(n, t2-t1)) < 0 ) return false;

    return true;
}

bool dcPointTriangle(gReal &d, Vec3 &n, const Vec3 &x, const Vec3 &t0, const Vec3 &t1, const Vec3 &t2, gReal d_lower, gReal d_upper)
{
	Vec3 e1, e2, e3, x_t0;

	x_t0 = x-t0;

	e1 = t1-t0;
	e2 = t2-t0;
	
	// n = normal vector to the triangle plane with the right hand rule
	n = Cross(e1, e2);
	n.Normalize();

	// d = signed distance
	d = Inner(n, x_t0);

	// if d is not in the range of [d_lower, d_upper], x is not in contact.
	if ( d > d_upper || d < d_lower ) return false;

	// checks if x is inside of the trigonal prism
	if ( Inner(x_t0, Cross(n, e1)) < 0 ) return false;
	if ( Inner(x_t0, Cross(e2, n)) < 0 ) return false;
	if ( Inner(x-t1, Cross(n, t2-t1)) < 0 ) return false;

    return true;
}

bool detectCollision(gReal &pd, gReal &s, gReal &alpha, gReal &beta, Vec3 &x_collision, Vec3 &n, const Vec3 &x_present, const Vec3 &x_previous, const Vec3 &t0, const Vec3 &t1, const Vec3 &t2)
{
	gReal a, b, e1e2, sd;
	Vec3 e1, e2, xc0;
	gReal eps = 1E-8;

	// initialize variables to be returned
	pd = -1.0;
	s = -1.0;
	alpha = beta = -1.0;
	x_collision[0] = x_collision[1] = x_collision[2] = 0.0;

	e1 = t1-t0;
	e2 = t2-t0;
	e1.Normalize();
	e2.Normalize();

	// n = normal vector to the triangle plane with the right hand rule
	n = Cross(e1, e2);
	n.Normalize();

	// If the present position of the test point is still above the plane, return false.
	// (This function cannot detect the reverse-sided collision.)
	sd = Inner(n, x_present - t0);
	if ( sd > 0.0 ) return false;
	
	pd = -sd;	// penetration depth

	// ---------------- calculate x_collision ----------------------------------------------------------
	// x_collision = the intersectional point of a line trajectory(from x_previous to x_present) and the plane
	//      1. x_collision = x_previous + s * (x_present - x_previous) where 0 <= s <= 1.
	//      2. Inner(n, x_collision - t0) = 0.
	//      ---> a * s = b where a = Inner(n, x_present - x_previous), b = -Inner(n, x_previous - t0).
	// -------------------------------------------------------------------------------------------------

	a = Inner(n, x_present - x_previous);
	b = -Inner(n, x_previous - t0);
	
	// If the trajectory of x from x_previous to x_present is parallel to the plane so that it doesn's make a new collision, then return false.
	if ( fabs(a) < eps ) return false;	

	s = b/a;

	// If the trajectory is apart from the plane, return false.
	if ( s < 0.0 || s > 1.0 ) return false;

	// colliding position
	x_collision = x_previous + s * (x_present - x_previous);

	// ---------------- check if the colliding point is inside of the triangle ---------------------
	// alpha * e1 + beta * e2 = x_collision - t0 where alpha >= 0, beta >= 0, alpha+beta <= 1
	// Let xc0 = x_collision - t0, <a,b> = Inner(a,b).
	// Then alpha = (<xc0,e1> - <xc0,e2> * <e1,e2>) / (1 - <e1,e2>^2),
	//      beta  = (<xc0,e2> - <xc0,e1> * <e1,e2>) / (1 - <e1,e2>^2) = <xc0,e2> - alpha * <e1,e2>
	// -------------------------------------------------------------------------------------------------

	e1e2 = Inner(e1, e2);
	xc0 = x_collision - t0;
	alpha = (Inner(xc0, e1) - Inner(xc0, e2) * e1e2) / (1.0 - e1e2 * e1e2);
	beta = Inner(xc0, e2) - alpha * e1e2;

	// If the colliding point is outside of the triangle plane, return false.
	if ( alpha < 0.0 || beta < 0.0 || alpha + beta > 1.0 ) return false;

	return true;
}

bool dcBox2d(gReal *x, gReal *y, gReal *X, gReal *Y)
{
	gReal a1, a2, b1, b2;
	for (int i=0; i<4; i++) {
		a1 = (x[1]-x[0]) * (Y[i]-y[0]) - (y[1]-y[0]) * (X[i]-x[0]);
		a2 = (x[1]-x[0]) * (Y[i]-y[3]) - (y[1]-y[0]) * (X[i]-x[3]);
		b1 = (x[3]-x[0]) * (Y[i]-y[0]) - (y[3]-y[0]) * (X[i]-x[0]);
		b2 = (x[3]-x[0]) * (Y[i]-y[1]) - (y[3]-y[0]) * (X[i]-x[1]);
		if ( a1*a2 <= 0 && b1*b2 <= 0 ) return true;
	}
	for (int i=0; i<4; i++) {
		a1 = (X[1]-X[0]) * (y[i]-Y[0]) - (Y[1]-Y[0]) * (x[i]-X[0]);
		a2 = (X[1]-X[0]) * (y[i]-Y[3]) - (Y[1]-Y[0]) * (x[i]-X[3]);
		b1 = (X[3]-X[0]) * (y[i]-Y[0]) - (Y[3]-Y[0]) * (x[i]-X[0]);
		b2 = (X[3]-X[0]) * (y[i]-Y[1]) - (Y[3]-Y[0]) * (x[i]-X[1]);
		if ( a1*a2 <= 0 && b1*b2 <= 0 ) return true;
	}
	return false;
}

bool dcOBBOverlap(gReal *a, gReal *Pa, gReal *Ra, gReal *b, gReal *Pb, gReal *Rb)
{
	// "Simple Intersection Tests For Games" by Miguel Gomez
	// http://www.gamasutra.com/view/feature/3383/simple_intersection_tests_for_games.php?page=5

	//translation, in global frame
	gReal v[3];
	v[0] = Pb[0] - Pa[0];
	v[1] = Pb[1] - Pa[1];
	v[2] = Pb[2] - Pa[2];

	//translation, in Ra's frame
	// T = transpose(Ra)*v
	gReal T[3];
	T[0] = Ra[0]*v[0] + Ra[1]*v[1] + Ra[2]*v[2];
	T[1] = Ra[3]*v[0] + Ra[4]*v[1] + Ra[5]*v[2];
	T[2] = Ra[6]*v[0] + Ra[7]*v[1] + Ra[8]*v[2];

	//calculate rotation matrix R = transpose(Ra)*Rb
	gReal R[3][3];
	R[0][0] = Ra[0]*Rb[0] + Ra[1]*Rb[1] + Ra[2]*Rb[2];	R[0][1] = Ra[0]*Rb[3] + Ra[1]*Rb[4] + Ra[2]*Rb[5];	R[0][2] = Ra[0]*Rb[6] + Ra[1]*Rb[7] + Ra[2]*Rb[8];
	R[1][0] = Ra[3]*Rb[0] + Ra[4]*Rb[1] + Ra[5]*Rb[2];	R[1][1] = Ra[3]*Rb[3] + Ra[4]*Rb[4] + Ra[5]*Rb[5];	R[1][2] = Ra[3]*Rb[6] + Ra[4]*Rb[7] + Ra[5]*Rb[8];
	R[2][0] = Ra[6]*Rb[0] + Ra[7]*Rb[1] + Ra[8]*Rb[2];	R[2][1] = Ra[6]*Rb[3] + Ra[7]*Rb[4] + Ra[8]*Rb[5];	R[2][2] = Ra[6]*Rb[6] + Ra[7]*Rb[7] + Ra[8]*Rb[8];
	 

	/*ALGORITHM: Use the separating axis test for all 15 potential 
	separating axes. If a separating axis could not be found, the two 
	boxes overlap. */

	gReal ra, rb, t;
	int i, k;

	//Ra's basis vectors
	for( i=0 ; i<3 ; i++ )
	{
		ra = a[i];
		rb = b[0]*fabs(R[i][0]) + b[1]*fabs(R[i][1]) + b[2]*fabs(R[i][2]);
		t = fabs( T[i] );
		if( t > ra + rb ) return false;
	}

	//Rb's basis vectors
	for( k=0 ; k<3 ; k++ )
	{
		ra = a[0]*fabs(R[0][k]) + a[1]*fabs(R[1][k]) + a[2]*fabs(R[2][k]);
		rb = b[k];
		t = fabs( T[0]*R[0][k] + T[1]*R[1][k] + T[2]*R[2][k] );
		if( t > ra + rb ) return false;
	}

	//9 cross products

	//L = A0 x B0
	ra = a[1]*fabs(R[2][0]) + a[2]*fabs(R[1][0]);
	rb = b[1]*fabs(R[0][2]) + b[2]*fabs(R[0][1]);
	t = fabs( T[2]*R[1][0] - T[1]*R[2][0] );
	if( t > ra + rb ) return false;

	//L = A0 x B1
	ra = a[1]*fabs(R[2][1]) + a[2]*fabs(R[1][1]);
	rb = b[0]*fabs(R[0][2]) + b[2]*fabs(R[0][0]);
	t = fabs( T[2]*R[1][1] - T[1]*R[2][1] );
	if( t > ra + rb ) return false;

	//L = A0 x B2
	ra = a[1]*fabs(R[2][2]) + a[2]*fabs(R[1][2]);
	rb = b[0]*fabs(R[0][1]) + b[1]*fabs(R[0][0]);
	t = fabs( T[2]*R[1][2] - T[1]*R[2][2] );
	if( t > ra + rb ) return false;

	//L = A1 x B0
	ra = a[0]*fabs(R[2][0]) + a[2]*fabs(R[0][0]);
	rb = b[1]*fabs(R[1][2]) + b[2]*fabs(R[1][1]);
	t = fabs( T[0]*R[2][0] - T[2]*R[0][0] );
	if( t > ra + rb ) return false;

	//L = A1 x B1
	ra = a[0]*fabs(R[2][1]) + a[2]*fabs(R[0][1]);
	rb = b[0]*fabs(R[1][2]) + b[2]*fabs(R[1][0]);
	t = fabs( T[0]*R[2][1] - T[2]*R[0][1] );
	if( t > ra + rb ) return false;

	//L = A1 x B2
	ra = a[0]*fabs(R[2][2]) + a[2]*fabs(R[0][2]);
	rb = b[0]*fabs(R[1][1]) + b[1]*fabs(R[1][0]);
	t = fabs( T[0]*R[2][2] - T[2]*R[0][2] );
	if( t > ra + rb ) return false;

	//L = A2 x B0
	ra = a[0]*fabs(R[1][0]) + a[1]*fabs(R[0][0]);
	rb = b[1]*fabs(R[2][2]) + b[2]*fabs(R[2][1]);
	t = fabs( T[1]*R[0][0] - T[0]*R[1][0] );
	if( t > ra + rb ) return false;

	//L = A2 x B1
	ra = a[0]*fabs(R[1][1]) + a[1]*fabs(R[0][1]);
	rb = b[0] *fabs(R[2][2]) + b[2]*fabs(R[2][0]);
	t = fabs( T[1]*R[0][1] - T[0]*R[1][1] );
	if( t > ra + rb ) return false;

	//L = A2 x B2
	ra = a[0]*fabs(R[1][2]) + a[1]*fabs(R[0][2]);
	rb = b[0]*fabs(R[2][1]) + b[1]*fabs(R[2][0]);
	t = fabs( T[1]*R[0][2] - T[0]*R[1][2] );
	if( t > ra + rb ) return false;

	/*no separating axis found,
	the two boxes overlap */

	return true;
}

bool dcTriOBBOverlap(gReal *v0, gReal *v1, gReal *v2, gReal *a, gReal *Pa, gReal *Ra)
{
	gReal P[12], Pnew[12];
	P[0] = Pa[0];	P[3] = v0[0];	P[6] = v1[0];	P[9] = v2[0];
	P[1] = Pa[1];	P[4] = v0[1];	P[7] = v1[1];	P[10] = v2[1];
	P[2] = Pa[2];	P[5] = v0[2];	P[8] = v1[2];	P[11] = v2[2];

	// convert OBB:Tri into AABB:Tri
	matSet_multAtB(Pnew, Ra, P, 3, 3, 3, 4); // Pnew = transpose(Ra)*P

	return triBoxOverlap(Pnew, a, &Pnew[3], &Pnew[6], &Pnew[9]);
}

bool dcPointOBBOverlap(gReal *v, gReal *Ea, gReal *Pa, gReal *Ra)
{
	gReal vnew[3], v_p[3];
	v_p[0] = v[0]-Pa[0];
	v_p[1] = v[1]-Pa[1];
	v_p[2] = v[2]-Pa[2];
	matSet_multAtB(vnew, Ra, v_p, 3, 3, 3, 1); // vnew = transpose(Ra)*(v-Pa)
	if ( vnew[0] < -Ea[0] || vnew[0] > Ea[0] || vnew[1] < -Ea[1] || vnew[1] > Ea[1] || vnew[2] < -Ea[2] || vnew[2] > Ea[2] )
		return false;
	return true;
}



//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//
// http://jgt.akpeters.com/papers/AkenineMoller01/tribox.html
//
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

/********************************************************/
/* AABB-triangle overlap test code                      */
/* by Tomas Akenine-Möller                              */
/* Function: int triBoxOverlap(gReal *boxcenter,       */
/*          gReal *boxhalfsize, gReal *trivert0       */
/*          gReal *trivert1, gReal *trivert2);        */
/* History:                                             */
/*   2001-03-05: released the code in its first version */
/*   2001-06-18: changed the order of the tests, faster */
/*                                                      */
/* Acknowledgement: Many thanks to Pierre Terdiman for  */
/* suggestions and discussions on how to optimize code. */
/* Thanks to David Hunt for finding a ">="-bug!         */
/********************************************************/
#include <math.h>
#include <stdio.h>

#define X 0
#define Y 1
#define Z 2

#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0]; 

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2]; 

#define FINDMINMAX(x0,x1,x2,min,max) \
  min = max = x0;   \
  if(x1<min) min=x1;\
  if(x1>max) max=x1;\
  if(x2<min) min=x2;\
  if(x2>max) max=x2;

int planeBoxOverlap(gReal normal[3],gReal d, gReal maxbox[3])
{
  int q;
  gReal vmin[3],vmax[3];
  for(q=X;q<=Z;q++)
  {
    if(normal[q]>0.0f)
    {
      vmin[q]=-maxbox[q];
      vmax[q]=maxbox[q];
    }
    else
    {
      vmin[q]=maxbox[q];
      vmax[q]=-maxbox[q];
    }
  }
  if(DOT(normal,vmin)+d>0.0f) return 0;
  if(DOT(normal,vmax)+d>=0.0f) return 1;
  
  return 0;
}


/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)			   \
	p0 = a*v0[Y] - b*v0[Z];			       	   \
	p2 = a*v2[Y] - b*v2[Z];			       	   \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_X2(a, b, fa, fb)			   \
	p0 = a*v0[Y] - b*v0[Z];			           \
	p1 = a*v1[Y] - b*v1[Z];			       	   \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)			   \
	p0 = -a*v0[X] + b*v0[Z];		      	   \
	p2 = -a*v2[X] + b*v2[Z];	       	       	   \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)			   \
	p0 = -a*v0[X] + b*v0[Z];		      	   \
	p1 = -a*v1[X] + b*v1[Z];	     	       	   \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)			   \
	p1 = a*v1[X] - b*v1[Y];			           \
	p2 = a*v2[X] - b*v2[Y];			       	   \
        if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)			   \
	p0 = a*v0[X] - b*v0[Y];				   \
	p1 = a*v1[X] - b*v1[Y];			           \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(min>rad || max<-rad) return 0;

bool triBoxOverlap(gReal *boxcenter, gReal *boxhalfsize, gReal *trivert0, gReal *trivert1, gReal *trivert2)
{

  /*    use separating axis theorem to test overlap between triangle and box */
  /*    need to test for overlap in these directions: */
  /*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
  /*       we do not even need to test these) */
  /*    2) normal of the triangle */
  /*    3) crossproduct(edge from tri, {x,y,z}-directin) */
  /*       this gives 3x3=9 more tests */
   gReal v0[3],v1[3],v2[3];
   gReal min,max,d,p0,p1,p2,rad,fex,fey,fez;  
   gReal normal[3],e0[3],e1[3],e2[3];

   /* This is the fastest branch on Sun */
   /* move everything so that the boxcenter is in (0,0,0) */
   SUB(v0,trivert0,boxcenter);
   SUB(v1,trivert1,boxcenter);
   SUB(v2,trivert2,boxcenter);

   /* compute triangle edges */
   SUB(e0,v1,v0);      /* tri edge 0 */
   SUB(e1,v2,v1);      /* tri edge 1 */
   SUB(e2,v0,v2);      /* tri edge 2 */

   /* Bullet 3:  */
   /*  test the 9 tests first (this was faster) */
   fex = fabs(e0[X]);
   fey = fabs(e0[Y]);
   fez = fabs(e0[Z]);
   AXISTEST_X01(e0[Z], e0[Y], fez, fey);
   AXISTEST_Y02(e0[Z], e0[X], fez, fex);
   AXISTEST_Z12(e0[Y], e0[X], fey, fex);

   fex = fabs(e1[X]);
   fey = fabs(e1[Y]);
   fez = fabs(e1[Z]);
   AXISTEST_X01(e1[Z], e1[Y], fez, fey);
   AXISTEST_Y02(e1[Z], e1[X], fez, fex);
   AXISTEST_Z0(e1[Y], e1[X], fey, fex);

   fex = fabs(e2[X]);
   fey = fabs(e2[Y]);
   fez = fabs(e2[Z]);
   AXISTEST_X2(e2[Z], e2[Y], fez, fey);
   AXISTEST_Y1(e2[Z], e2[X], fez, fex);
   AXISTEST_Z12(e2[Y], e2[X], fey, fex);

   /* Bullet 1: */
   /*  first test overlap in the {x,y,z}-directions */
   /*  find min, max of the triangle each direction, and test for overlap in */
   /*  that direction -- this is equivalent to testing a minimal AABB around */
   /*  the triangle against the AABB */

   /* test in X-direction */
   FINDMINMAX(v0[X],v1[X],v2[X],min,max);
   if(min>boxhalfsize[X] || max<-boxhalfsize[X]) return false;

   /* test in Y-direction */
   FINDMINMAX(v0[Y],v1[Y],v2[Y],min,max);
   if(min>boxhalfsize[Y] || max<-boxhalfsize[Y]) return false;

   /* test in Z-direction */
   FINDMINMAX(v0[Z],v1[Z],v2[Z],min,max);
   if(min>boxhalfsize[Z] || max<-boxhalfsize[Z]) return false;

   /* Bullet 2: */
   /*  test if the box intersects the plane of the triangle */
   /*  compute plane equation of triangle: normal*x+d=0 */
   CROSS(normal,e0,e1);
   d=-DOT(normal,v0);  /* plane eq: normal.x+d=0 */
   if(!planeBoxOverlap(normal,d,boxhalfsize)) return false;

   return true;   /* box and triangle overlaps */
}
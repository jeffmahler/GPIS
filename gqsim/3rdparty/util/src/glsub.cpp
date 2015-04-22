//================================================================================
//         SUB-FUNCTIONS FOR OPENGL RENDERING
// 
//                                                               junggon@gmail.com
//================================================================================

#include "glsub.h"

static const gReal PI = (gReal)3.14159265358979;

void glsub_set_z_axis(Vec3 axis)
{
	SE3 T;
	Vec3 z(0,0,1);
	axis.Normalize();

	if ( Norm(axis-z) < 1E-6 )		// if axis = z
		T.SetIdentity();	// identity
	else if ( Norm(axis+z) < 1E-6 )	// else if axis = -z
		T = SE3(SO3(1,0,0,0,-1,0,0,0,-1),Vec3(0,0,0));
	else
	{
		Vec3 m = Cross(z,axis);
		gReal theta = asin(Norm(m));
		if ( Inner(z,axis) < 0 ) theta = (gReal)3.14159265358979 - theta;

		m.Normalize();
		T = SE3(Exp(m*theta), Vec3(0,0,0));
	}

#if GEAR_DOUBLE_PRECISION
	glMultMatrixd(T.GetArray());
#else
	glMultMatrixf(T.GetArray());
#endif
}

void glsub_draw_arrow(GLUquadricObj *qobj, Vec3 n, double length1, double length2, double radius1, double radius2, int slice, int stack, int ring)
{
	if ( Norm(n) < 1E-6 ) return;
	n.Normalize();
	glPushMatrix();
	glsub_set_z_axis(n);
	gluCylinder(qobj, radius1, radius1, length1, slice, stack);
	glTranslated(0, 0, length1);
	//gluDisk(qobj, 0, radius1, slice, ring);
	gluCylinder(qobj, radius2, 0, length2, slice, stack);
	glsub_set_z_axis(Vec3(0,0,-1));
	gluDisk(qobj, 0, radius2, slice, ring);
	glTranslated(0, 0, length1);
	gluDisk(qobj, 0, radius1, slice, ring);
	glPopMatrix();
}

void glsub_draw_arrow_with_double_heads(GLUquadricObj *qobj, Vec3 n, double length1, double length2, double radius1, double radius2, int slice, int stack, int ring)
{
	if ( Norm(n) < 1E-6 ) return;
	n.Normalize();
	glPushMatrix();
	glsub_set_z_axis(n);
	gluCylinder(qobj, radius1, radius1, length1, slice, stack);
	glTranslated(0, 0, length1);
	gluCylinder(qobj, radius2, 0, 0.6*length2, slice, stack);
	glTranslated(0, 0, 0.4*length2);
	gluCylinder(qobj, radius2, 0, 0.6*length2, slice, stack);
	glsub_set_z_axis(Vec3(0,0,-1));
	gluDisk(qobj, 0, radius2, slice, ring);
	glTranslated(0, 0, 0.4*length2);
	gluDisk(qobj, 0, radius2, slice, ring);
	glTranslated(0, 0, length1);
	gluDisk(qobj, 0, radius1, slice, ring);
	glPopMatrix();
}

void glsub_draw_ground(GLfloat L, GLfloat h, GLfloat color1[], GLfloat color2[], GLfloat position[], GLfloat normal[])
{
	GLint previous_polygonmode[2];
	GLfloat previous_color[4];
	glGetFloatv(GL_CURRENT_COLOR, previous_color);
	glGetIntegerv(GL_POLYGON_MODE, previous_polygonmode);
	glPolygonMode(GL_FRONT,GL_FILL);

	glPushMatrix();
	glTranslatef(position[0],position[1],position[2]);
	glsub_set_z_axis(Vec3((gReal)normal[0],(gReal)normal[1],(gReal)normal[2]));
	glNormal3f(0,0,1);
	bool bcolor1 = true, bcolor1_ = true;
	for (GLfloat x = -L; x < L; x += h) {
		bcolor1 = bcolor1_;
		for (GLfloat y = -L; y < L; y += h) {
			if ( bcolor1 ) { glColor4fv(color1); } else { glColor4fv(color2); }
			glRectd( x, y, x+h, y+h );
			bcolor1 = !bcolor1;
		}
		bcolor1_ = !bcolor1_;
	}
	glPopMatrix();

	glColor4fv(previous_color);
	glPolygonMode(previous_polygonmode[0], previous_polygonmode[1]);
}

void glsub_draw_coord_sys(GLUquadricObj *qobj, double L)
{
	glColor3f(1,0,0);
	glsub_draw_arrow(qobj, Vec3(1,0,0), L*0.7, L*0.3, L/20*1, L/20*2, 10, 1, 2);
	glColor3f(0,1,0);
	glsub_draw_arrow(qobj, Vec3(0,1,0), L*0.7, L*0.3, L/20*1, L/20*2, 10, 1, 2);
	glColor3f(0,0,1);
	glsub_draw_arrow(qobj, Vec3(0,0,1), L*0.7, L*0.3, L/20*1, L/20*2, 10, 1, 2);
}

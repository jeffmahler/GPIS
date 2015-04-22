//================================================================================
//         SUB-FUNCTIONS FOR OPENGL RENDERING
// 
//                                                               junggon@gmail.com
//================================================================================

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "liegroup.h"

void glsub_draw_arrow(GLUquadricObj *qobj, Vec3 n, double length1, double length2, double radius1, double radius2, int slice, int stack, int ring);
void glsub_draw_arrow_with_double_heads(GLUquadricObj *qobj, Vec3 n, double length1, double length2, double radius1, double radius2, int slice, int stack, int ring);
void glsub_set_z_axis(Vec3 axis);
void glsub_draw_ground(GLfloat L, GLfloat h, GLfloat color1[], GLfloat color2[], GLfloat position[], GLfloat normal[]);
void glsub_draw_coord_sys(GLUquadricObj *qobj, double L);


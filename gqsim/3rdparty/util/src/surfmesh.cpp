//================================================================================
//         SURFACE MESH
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

#include <algorithm>
#include <utility>
#include <fstream>
#include "surfmesh.h"
#include "glsub.h"
#include "rmatrix3j.h"

using namespace std;

bool SurfaceMesh::loadFromDataTri(const std::vector<double> &x, const std::vector<int> &f, Vec3 scale, SE3 T0)
{
	int n, m;
	Vec3 tmp;

	// check data size (x.size() and f.size() must be multiples of 3)
	if ( x.size() % 3 != 0 || f.size() % 3 != 0 ) {
		cerr << "error:: size must be multiples of 3" << endl;
		return false;
	}

	n = x.size()/3; m = f.size()/3;

	// get vertex positions
	vertices.resize(n);
	for (int i=0; i<n; i++) {
		tmp[0] = x[3*i];
		tmp[1] = x[3*i+1];
		tmp[2] = x[3*i+2];

		// apply scale factors
		tmp[0] *= scale[0];
		tmp[1] *= scale[1];
		tmp[2] *= scale[2];

		// position of nodes in {body}
		vertices[i] = T0 * tmp;
	}

	// get vertex indices of the faces
	faces.resize(m);
	for (int i=0; i<m; i++) {
		faces[i].elem_type = MT_TRIANGLE;
		faces[i].vertex_indices[0] = f[3*i];
		faces[i].vertex_indices[1] = f[3*i+1];
		faces[i].vertex_indices[2] = f[3*i+2];
		int idx0 = faces[i].vertex_indices[0], idx1 = faces[i].vertex_indices[1], idx2 = faces[i].vertex_indices[2];
		if ( idx0 == idx1 || idx0 == idx1 || idx1 == idx2 ) {
			cerr << "error:: duplicated vertices in a triangle face " << i << ": (" << idx0 << ", " << idx1 << ", " << idx2 << ")" << endl;
			return false;
		}
		if ( idx0 < 0 || idx0 >= n || idx1 < 0 || idx1 >= n || idx2 < 0 || idx2 >= n ) {
			cout << "error:: invalid vertex index in a triangle face " << i << ": (" << idx0 << ", " << idx1 << ", " << idx2 << ")" << endl;
			return false;
		}
	}

	return true;
}

bool SurfaceMesh::loadFromDataVtx(const std::vector<double> &x, const std::vector<double> &vn, Vec3 scale, SE3 T0)
{
	int n;
	Vec3 tmp;

	// check data size
	if ( x.size() % 3 != 0 ) {
		cerr << "error:: size of vertices must be multiples of 3" << endl;
		return false;
	}
	if ( vn.size() > 0 && vn.size() != x.size() ) {
		cerr << "error:: size mismatch (vn.size() != x.size())" << endl;
		return false;
	}

	n = x.size()/3; 

	// get vertex positions
	vertices.resize(n);
	for (int i=0; i<n; i++) {
		tmp[0] = x[3*i];
		tmp[1] = x[3*i+1];
		tmp[2] = x[3*i+2];

		// apply scale factors
		tmp[0] *= scale[0];
		tmp[1] *= scale[1];
		tmp[2] *= scale[2];

		// position of nodes in {body}
		vertices[i] = T0 * tmp;
	}

	// get vertex indices normals
	if ( vn.size() > 0 && vn.size() == x.size()) {
		normals.resize(n);
		vertexnormals.resize(n);
		for (int i=0; i<n; i++) {
			normals[i][0] = vn[3*i];
			normals[i][1] = vn[3*i+1];
			normals[i][2] = vn[3*i+2];
			vertexnormals[i] = normals[i];
		}
	}

	return true;
}

void SurfaceMesh::saveToFileSTL(const char *filepath, double scale)
{
	bool btri=true;
	int cnt=0;
	for (size_t i=0; i<faces.size(); i++) {
		if ( faces[i].elem_type != MT_TRIANGLE ) { btri = false; cnt++; }
	}
	if ( !btri ) {
		cout << "error:: failed in saving to STL:: " << cnt << " faces are not triangular" << endl;
		return;
	}

	ofstream fout(filepath);
	fout << "solid " << name << endl;
	for (size_t i=0; i<faces.size(); i++) {
		if ( faces[i].elem_type != MT_TRIANGLE ) { cout << "warning:: mesh face is not triangular" << endl; }
		Vec3 v0, v1, v2, n;
		v0 = scale * vertices[faces[i].vertex_indices[0]];
		v1 = scale * vertices[faces[i].vertex_indices[1]];
		v2 = scale * vertices[faces[i].vertex_indices[2]];
		n = Cross(v1-v0, v2-v0);
		n.Normalize();
		fout << "  facet normal " << n[0] << " " << n[1] << " " << n[2] << endl;
		fout << "    outer loop" << endl;
		fout << "      vertex " << v0[0] << " " << v0[1] << " " << v0[2] << endl;
		fout << "      vertex " << v1[0] << " " << v1[1] << " " << v1[2] << endl;
		fout << "      vertex " << v2[0] << " " << v2[1] << " " << v2[2] << endl;
		fout << "    endloop" << endl;
		fout << "  endfacet" << endl;
	}
	fout << "endsolid " << name << endl;
	fout.close();
}

void SurfaceMesh::saveToFileTRI(const char *filepath, double scale)
{
	bool btri=true;
	int cnt=0;
	for (size_t i=0; i<faces.size(); i++) {
		if ( faces[i].elem_type != MT_TRIANGLE ) { btri = false; cnt++; }
	}
	if ( !btri ) {
		cout << "error:: failed in saving to TRI:: " << cnt << " faces are not triangular" << endl;
		return;
	}

	ofstream fout(filepath);
	fout << vertices.size() << endl;
	fout << faces.size() << endl;
	fout << endl;
	for (size_t i=0; i<vertices.size(); i++) {
		fout << scale*vertices[i][0] << " " << scale*vertices[i][1] << " " << scale*vertices[i][2] << endl;
	}
	fout << endl;
	for (size_t i=0; i<faces.size(); i++) {
		fout << faces[i].vertex_indices[0] << " " << faces[i].vertex_indices[1] << " " << faces[i].vertex_indices[2] << endl;
	}
	fout << endl;
	fout.close();
}

void SurfaceMesh::saveToFileOBJ(const char *filepath, double scale)
{
	bool btri=true;
	int cnt=0;
	for (size_t i=0; i<faces.size(); i++) {
		if ( faces[i].elem_type != MT_TRIANGLE ) { btri = false; cnt++; }
	}
	if ( !btri ) {
		cout << "error:: failed in saving to OBJ:: " << cnt << " faces are not triangular" << endl;
		return;
	}

	ofstream fout(filepath);
	fout << "# " << name << endl;
	fout << endl;
	for (size_t i=0; i<vertices.size(); i++) {
		fout << "v " << scale*vertices[i][0] << " " << scale*vertices[i][1] << " " << scale*vertices[i][2] << endl;
	}
	fout << endl;
	for (size_t i=0; i<faces.size(); i++) {
		fout << "f " << faces[i].vertex_indices[0]+1 << "// " << faces[i].vertex_indices[1]+1 << "// " << faces[i].vertex_indices[2]+1 << "//" << endl;
	}
	fout << endl;
	fout.close();
}

bool SurfaceMesh::getReady()
{
	_removeVertexRedundancy();
	_scanAdjacentFacesAndVertices();
	updateFaceNormalsWithVertexPositions();
	updateVertexNormalsByAveragingAdjacentFaceNormals();
	if ( faces.size() == 0 && vertices.size() > 0 ) {
		bRenderingVertices = true;
	}
	return true;
}

void SurfaceMesh::updateFaceNormalsWithVertexPositions()
{
	if ( faces.size() == 0 ) return;

	// vertex order is assumed to be counter-clockwise around the normal
	normals.resize(faces.size());
	for (int i=0; i<(int)faces.size(); i++) {
		normals[i] = _calcFaceNormal(i);
		faces[i].normal_indices[0] = i;
		faces[i].normal_type = NT_FACE;
	}
}

void SurfaceMesh::updateVertexNormalsByAveragingAdjacentFaceNormals()
{
	if ( faces.size() == 0 ) return;

	vertexnormals.resize(vertices.size());
	for (size_t i=0; i<vertices.size(); i++) {
		Vec3 n(0,0,0);
		if ( _idxAdjFaces[i].size() > 0 ) {
			for (size_t j=0; j<_idxAdjFaces[i].size(); j++) {
				n += normals[faces[_idxAdjFaces[i][j]].normal_indices[0]];
			}
			n *= (1./(double)(_idxAdjFaces[i].size()));
			n.Normalize();
		}
		vertexnormals[i] = n;
	}
}

void SurfaceMesh::render()
{
	if ( !bRendering )
		return;

	GLint previous_polygonmode[2];
	GLfloat previous_color[4];
	glGetFloatv(GL_CURRENT_COLOR, previous_color);
	glGetIntegerv(GL_POLYGON_MODE, previous_polygonmode);

	//glColor3f((float)color[0],(float)color[1],(float)color[2]);
	float specular[4] = {0.75f, 0.75f, 0.75f, 1.0f};
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess); // range 0 ~ 128
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color_specular);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glColor4f(color[0],color[1],color[2],alpha);

	switch (draw_type) {
	case DT_SOLID:
		glPolygonMode(GL_FRONT,GL_FILL);
		break;
	case DT_WIRE:
		glPolygonMode(GL_FRONT,GL_LINE);
		break;
	}

	for (size_t i=0; i<faces.size(); i++) {
		switch ( faces[i].elem_type ) {
		case MT_TRIANGLE:
			glBegin(GL_TRIANGLES);
			if ( faces[i].normal_type == NT_FACE ) {
				glNormal3dv(normals[faces[i].normal_indices[0]].GetArray());
			}
			for (int j=0; j<3; j++) {
				if ( faces[i].normal_type == NT_VERTEX ) {
					glNormal3dv(normals[faces[i].normal_indices[j]].GetArray());
				}
				glVertex3dv(vertices[faces[i].vertex_indices[j]].GetArray());
			}
			glEnd();
			break;
		case MT_QUAD:
			glBegin(GL_QUADS);
			if ( faces[i].normal_type == NT_FACE ) {
				glNormal3dv(normals[faces[i].normal_indices[0]].GetArray());
			}
			for (int j=0; j<4; j++) {
				if ( faces[i].normal_type == NT_VERTEX ) {
					glNormal3dv(normals[faces[i].normal_indices[j]].GetArray());
				}
				glVertex3dv(vertices[faces[i].vertex_indices[j]].GetArray());
			}
			glEnd();
			break;
		}
	}

	if ( bRenderingVertices ) {
		glColor3f(0,0,0);
		if ( vertex_radius == 0 ) {
			glBegin(GL_POINTS);
			for (size_t i=0; i<vertices.size(); i++) {
				glVertex3dv(vertices[i].GetArray());
			}
			glEnd();
		} else {
			GLUquadricObj *qobj = gluNewQuadric();
			for (size_t i=0; i<vertices.size(); i++) {
				glPushMatrix();
				glTranslated(vertices[i][0], vertices[i][1], vertices[i][2]);
				gluSphere(qobj, vertex_radius, 10, 10);
				glPopMatrix();
			}
			gluDeleteQuadric(qobj);
		}
	}

	if ( bRenderingFaceNormals ) {
		GLUquadricObj *qobj = gluNewQuadric();
		glColor3f(0,0,1);
		for (size_t i=0; i<faces.size(); i++) {
			Vec3 c(0,0,0), n(0,0,0);
			switch ( faces[i].elem_type ) {
			case MT_TRIANGLE:
				c = vertices[faces[i].vertex_indices[0]] + vertices[faces[i].vertex_indices[1]] + vertices[faces[i].vertex_indices[2]];
				c *= (1./3.);
				break;
			case MT_QUAD:
				c = vertices[faces[i].vertex_indices[0]] + vertices[faces[i].vertex_indices[1]] + vertices[faces[i].vertex_indices[2]] + vertices[faces[i].vertex_indices[3]];
				c *= 0.25;
				break;
			}
			switch ( faces[i].normal_type ) {
			case NT_FACE:
				n = normals[faces[i].normal_indices[0]];
				break;
			}
			glPushMatrix();
			glTranslated(c[0],c[1],c[2]);
			glsub_draw_arrow(qobj, n, 0.7*normallength, 0.3*normallength, 0.05*normallength, 0.08*normallength, 6, 1, 2);
			glPopMatrix();
		}
		gluDeleteQuadric(qobj);
	}

	if ( bRenderingVertexNormals ) {
		GLUquadricObj *qobj = gluNewQuadric();
		glColor3f(1,0,0);
		for (size_t i=0; i<vertexnormals.size(); i++) {
			glPushMatrix();
			glTranslated(vertices[i][0], vertices[i][1], vertices[i][2]);
			glsub_draw_arrow(qobj, vertexnormals[i], 0.7*normallength, 0.3*normallength, 0.05*normallength, 0.08*normallength, 6, 1, 2);
			glPopMatrix();
		}
		gluDeleteQuadric(qobj);
	}

	if ( idx_selected_vertices.size() > 0 ) {
		glColor3f(color_selected[0],color_selected[1],color_selected[2]);
		GLUquadricObj *qobj = gluNewQuadric();
		for (size_t i=0; i<idx_selected_vertices.size(); i++) {
			int idx = idx_selected_vertices[i];
			glPushMatrix();
			glTranslated(vertices[idx][0], vertices[idx][1], vertices[idx][2]);
			gluSphere(qobj, radius_ratio_selected*vertex_radius, 10, 10);
			glPopMatrix();
		}
		gluDeleteQuadric(qobj);
	}

	if ( idx_selected_faces.size() > 0 ) {
		glColor3f(color_selected[0],color_selected[1],color_selected[2]);
		for (size_t i=0; i<idx_selected_faces.size(); i++) {
			int idx = idx_selected_faces[i];
			switch ( faces[idx].elem_type ) {
			case MT_TRIANGLE:
				glBegin(GL_TRIANGLES);
				if ( faces[idx].normal_type == NT_FACE ) {
					glNormal3dv(normals[faces[idx].normal_indices[0]].GetArray());
				}
				for (int j=0; j<3; j++) {
					if ( faces[idx].normal_type == NT_VERTEX ) {
						glNormal3dv(normals[faces[idx].normal_indices[j]].GetArray());
					}
					glVertex3dv(vertices[faces[idx].vertex_indices[j]].GetArray());
				}
				glEnd();
				break;
			case MT_QUAD:
				glBegin(GL_QUADS);
				if ( faces[idx].normal_type == NT_FACE ) {
					glNormal3dv(normals[faces[idx].normal_indices[0]].GetArray());
				}
				for (int j=0; j<4; j++) {
					if ( faces[idx].normal_type == NT_VERTEX ) {
						glNormal3dv(normals[faces[idx].normal_indices[j]].GetArray());
					}
					glVertex3dv(vertices[faces[idx].vertex_indices[j]].GetArray());
				}
				glEnd();
				break;
			}
		}
	}

	glColor4fv(previous_color);
	glPolygonMode(previous_polygonmode[0], previous_polygonmode[1]);
}

bool SurfaceMesh::computeVolumeCentroidAndMomentOfInertia(double &vol, double &Cx, double &Cy, double &Cz, double &Ixx, double &Iyy, double &Izz, double &Ixy, double &Ixz, double &Iyz)
{
	// algorithm by Michael Kallay (http://jgt.akpeters.com/papers/Kallay06/Moment_of_Inertia.cpp)

	vol = Cx = Cy = Cz = Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0;
	double _m=0,_Cx=0,_Cy=0,_Cz=0,_xx=0,_yy=0,_zz=0,_yx=0,_zx=0,_zy=0;

	for (size_t i=0; i<faces.size(); i++) {
		if ( faces[i].elem_type != MT_TRIANGLE ) {
			cout << "error:: unsupported mesh type" << endl;
			return false;
		}
		
		double x1,y1,z1,x2,y2,z2,x3,y3,z3;
		x1 = vertices[faces[i].vertex_indices[0]][0];
		y1 = vertices[faces[i].vertex_indices[0]][1];
		z1 = vertices[faces[i].vertex_indices[0]][2];
		x2 = vertices[faces[i].vertex_indices[1]][0];
		y2 = vertices[faces[i].vertex_indices[1]][1];
		z2 = vertices[faces[i].vertex_indices[1]][2];
		x3 = vertices[faces[i].vertex_indices[2]][0];
		y3 = vertices[faces[i].vertex_indices[2]][1];
		z3 = vertices[faces[i].vertex_indices[2]][2];

		// Signed volume of this tetrahedron.
		double v = x1*y2*z3 + y1*z2*x3 + x2*y3*z1 - (x3*y2*z1 + x2*y1*z3 + y3*z2*x1);
        
		// Contribution to the mass
		_m += v;

		// Contribution to the centroid
		double x4 = x1 + x2 + x3;           _Cx += (v * x4);
		double y4 = y1 + y2 + y3;           _Cy += (v * y4);
		double z4 = z1 + z2 + z3;           _Cz += (v * z4);

		// Contribution to moment of inertia monomials
		_xx += v * (x1*x1 + x2*x2 + x3*x3 + x4*x4);
		_yy += v * (y1*y1 + y2*y2 + y3*y3 + y4*y4);
		_zz += v * (z1*z1 + z2*z2 + z3*z3 + z4*z4);
		_yx += v * (y1*x1 + y2*x2 + y3*x3 + y4*x4);
		_zx += v * (z1*x1 + z2*x2 + z3*x3 + z4*x4);
		_zy += v * (z1*y1 + z2*y2 + z3*y3 + z4*y4);        
	}

	//if ( _m < 1E-12 ) {
	//	cout << "error:: zero volume" << endl;
	//	return false;
	//}

	// Centroid.  
    double r = 1.0 / (4 * _m);
    Cx = _Cx * r;
    Cy = _Cy * r;
    Cz = _Cz * r;

    // Mass
    vol = _m / 6;

    // Moment of inertia about the centroid.
    r = 1.0 / 120;
    Ixy = _yx * r - vol * Cy*Cx;
    Ixz = _zx * r - vol * Cz*Cx;
    Iyz = _zy * r - vol * Cz*Cy;

    _xx = _xx * r - vol * Cx*Cx;
    _yy = _yy * r - vol * Cy*Cy;
    _zz = _zz * r - vol * Cz*Cz;

    Ixx = _yy + _zz;
    Iyy = _zz + _xx;
    Izz = _xx + _yy;

	return true;
}

bool SurfaceMesh::computeAreaCentroidAndMomentOfInertia(double &area, double &Cx, double &Cy, double &Cz, double &Ixx, double &Iyy, double &Izz, double &Ixy, double &Ixz, double &Iyz)
{
	// ref: http://en.wikipedia.org/wiki/Inertia_tensor_of_triangle

	area = Cx = Cy = Cz = Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0;

	RMatrix S = Ones(3,3); S(0,0) = 2; S(1,1) = 2; S(2,2) = 2; S *= (1./24.); // S = 1/24 * [2 1 1; 1 2 1; 1 1 2]
	RMatrix V(3,3), C(3,3), J = Zeros(3,3);

	for (size_t i=0; i<faces.size(); i++) {
		if ( faces[i].elem_type != MT_TRIANGLE ) {
			cout << "error:: unsupported mesh type" << endl;
			return false;
		}

		Vec3 &v0 = vertices[faces[i].vertex_indices[0]];
		Vec3 &v1 = vertices[faces[i].vertex_indices[1]];
		Vec3 &v2 = vertices[faces[i].vertex_indices[2]];

		// V = [v0; v1; v2]
		V(0,0) = v0[0]; V(0,1) = v0[1]; V(0,2) = v0[2];
		V(1,0) = v1[0]; V(1,1) = v1[1]; V(1,2) = v1[2];
		V(2,0) = v2[0]; V(2,1) = v2[1]; V(2,2) = v2[2];

		double a = Norm(Cross(v1-v0,v2-v0)); // twice the area of the triangle
		C = a * (~V * S * V); // covariance of the triangle
		double trC = a / 24. * ( SquareSum(v0) + SquareSum(v1) + SquareSum(v2) + SquareSum(v0+v1+v2) ); // trance of C
		Vec3 c = 1./3. * (v0 + v1 + v2); // centroid of the triangle

		area += 0.5 * a;
		J += trC * Eye(3,3) - C; // inertia w.r.t. the refence coordinate frame of the mesh
		Cx += 0.5 * a * c[0];
		Cy += 0.5 * a * c[1];
		Cz += 0.5 * a * c[2];
	}

	//if ( area < 1E-12 ) {
	//	cout << "error:: zero area" << endl;
	//	return false;
	//}

	Cx /= area;
	Cy /= area;
	Cz /= area;

	double CtC = Cx*Cx + Cy*Cy + Cz*Cz;

	RMatrix CCt(3,3);
	CCt(0,0) = Cx*Cx; CCt(0,1) = Cx*Cy; CCt(0,2) = Cx*Cz;
	CCt(1,0) = Cx*Cy; CCt(1,1) = Cy*Cy; CCt(1,2) = Cy*Cz;
	CCt(2,0) = Cx*Cz; CCt(2,1) = Cy*Cz; CCt(2,2) = Cz*Cz;

	RMatrix Jc = J - area * (CtC*Eye(3,3)-CCt); // inertia w.r.t. the centeroid

	Ixx = Jc(0,0); Iyy = Jc(1,1); Izz = Jc(2,2); Ixy = Jc(0,1); Ixz = Jc(0,2); Iyz = Jc(0,1);

	return true;
}

int SurfaceMesh::searchNearestVertex(const Vec3 &x, int seedidx)
{
	if ( vertices.size() == 0 ) return -1;
	if ( seedidx < 0 || seedidx >= vertices.size() ) seedidx = 0;
	int idx = seedidx, idxloc = idx;
	gReal dist = SquareSum(vertices[idx]-x), distloc = dist;
	while ( 1 ) {
		for (size_t i=0; i<_idxAdjVertices[idx].size(); i++) {
			distloc = SquareSum(vertices[_idxAdjVertices[idx][i]]-x);
			if ( distloc < dist ) {
				dist = distloc;
				idxloc = _idxAdjVertices[idx][i];
			}
		}
		if ( idxloc == idx ) break;
		idx = idxloc;
	}
	return idx;
}

int SurfaceMesh::searchNearestVertex(const Vec3 &x, int seedidx, const Vec3 &N)
{
	if ( vertices.size() == 0 ) return -1;
	if ( seedidx < 0 || seedidx >= vertices.size() ) seedidx = 0;
	int idx = seedidx, idxloc = idx;
	gReal dist = SquareSum(vertices[idx]-x), distloc = dist, anglimit = 135./180.*3.14159, ang, inn;
	while ( 1 ) {
		bool ballbadang = true;
		for (size_t i=0; i<_idxAdjVertices[idx].size(); i++) {
			distloc = SquareSum(vertices[_idxAdjVertices[idx][i]]-x);
			inn = Inner(N, vertexnormals[_idxAdjVertices[idx][i]]);
			if ( inn > 1 ) inn = 1; 
			if ( inn < -1 ) inn = -1;
			ang = acos(inn);
			if ( distloc < dist && ang > anglimit ) {
				dist = distloc;
				idxloc = _idxAdjVertices[idx][i];
			}
			if ( ang > anglimit ) {
				ballbadang = false;
			}
		}
		if ( idxloc == idx ) break;
		//if ( idxloc == idx && !ballbadang ) break;
		//if ( ballbadang ) {
		//	idxloc = int(prand(vertices.size()-1)); // if trapped by bad normals, seed again with random index
		//}
		idx = idxloc;
	}
	return idx;
}

void SurfaceMesh::_removeVertexRedundancy(double eps)
{
	//  0         2  5           0         2  
	//   \--------/ /\            \--------/\ 
	//    \      / /  \            \      /  \ 
	//     \    / /    \    ==>     \    /    \ 
	//      \  / /      \            \  /      \ 
	//       \/ /--------\            \/--------\
	//       1  3        4 		      1         3 
	//
	//  (5 vertices, 2 faces)  ==>  (3 vertices, 2 faces)

	size_t cnt=0;
	vector<bool> bremoved(vertices.size(), false);

	for (size_t i=0; i<vertices.size(); i++) {
		// if vertex i has already been disabled, pass
		if ( bremoved[i] ) continue;
		for (size_t j=i+1; j<vertices.size(); j++) {
			// if vertex j has already been disabled, pass
			if ( bremoved[j] ) continue;
			// if vertex i and j are identical, disable the vertex j and change the index i into j in all faces
			//if ( vertices[i][0] == vertices[j][0] && vertices[i][1] == vertices[j][1] && vertices[i][2] == vertices[j][2] ) {
			if ( SquareSum(vertices[i]-vertices[j]) <= eps*eps ) {
				bremoved[j] = true;
				cnt++;
				// change j into i
				for (size_t k=0; k<faces.size(); k++) {
					for (size_t l=0; l<4; l++) {
						if ( faces[k].vertex_indices[l] == j ) {
							faces[k].vertex_indices[l] = i;
						}
					}
				}
			}
		}
	}

	vector<Vec3> vertices_new(vertices.size()-cnt, Vec3(0,0,0));
	int idx=0;
	for (size_t i=0; i<vertices.size(); i++) {
		if ( !bremoved[i] ) {
			vertices_new[idx] = vertices[i];
			if ( idx != i ) {
				// change i into idx
				for (size_t j=0; j<faces.size(); j++) {
					for (size_t k=0; k<4; k++) {
						if ( faces[j].vertex_indices[k] == i ) {
							faces[j].vertex_indices[k] = idx;
						}
					}
				}
			}
			idx++;
		}
	}

	if ( vertices.size() != vertices_new.size() ) {
		cout << "vertex redundancy removed: number of vertices: " << vertices.size() << " --> " << vertices_new.size() << endl;
	}

	vertices = vertices_new;
}

void SurfaceMesh::_scanAdjacentFacesAndVertices()
{
	// scan adjacent faces of the vertices
	_idxAdjFaces.resize(vertices.size());
	for (size_t i=0; i<vertices.size(); i++) {
		_idxAdjFaces[i].clear();
		for (size_t j=0; j<faces.size(); j++) {
			if ( faces[j].vertex_indices[0] == i || faces[j].vertex_indices[1] == i || faces[j].vertex_indices[2] == i || faces[j].vertex_indices[3] == i ) {
				_idxAdjFaces[i].push_back(j);
			}
		}
	}

	// scan adjacent vertices of the vertices
	_idxAdjVertices.resize(vertices.size());
	for (size_t i=0; i<vertices.size(); i++) {
		_idxAdjVertices[i].clear();
		for (size_t j=0; j<_idxAdjFaces[i].size(); j++) {
			for (size_t k=0; k<4; k++) {
				int idx = faces[_idxAdjFaces[i][j]].vertex_indices[k];
				if ( idx != i && idx >= 0 && idx < vertices.size() ) {
					_idxAdjVertices[i].push_back(idx);
				}
			}
		}
		// remove duplication
		std::sort(_idxAdjVertices[i].begin(), _idxAdjVertices[i].end());
		_idxAdjVertices[i].erase(std::unique(_idxAdjVertices[i].begin(), _idxAdjVertices[i].end()), _idxAdjVertices[i].end());
	}

	// scan adjacent faces of the faces
	_idxAdjFacesOfFaces.resize(faces.size());
	for (size_t i=0; i<faces.size(); i++) {
		_idxAdjFacesOfFaces[i].clear();
		for (size_t j=0; j<4; j++) {
			int vidx = faces[i].vertex_indices[j];
			if ( vidx < 0 ) continue;
			for (size_t k=0; k<_idxAdjFaces[vidx].size(); k++) {
				if ( _checkAdjacencyOfFaces(i, _idxAdjFaces[vidx][k]) ) {
					_idxAdjFacesOfFaces[i].push_back(_idxAdjFaces[vidx][k]);
				}
			}
		}
		// remove duplication
		std::sort(_idxAdjFacesOfFaces[i].begin(), _idxAdjFacesOfFaces[i].end());
		_idxAdjFacesOfFaces[i].erase(std::unique(_idxAdjFacesOfFaces[i].begin(), _idxAdjFacesOfFaces[i].end()), _idxAdjFacesOfFaces[i].end());
		if ( _idxAdjFacesOfFaces[i].size() == 0 && faces.size() > 1 ) {
			cout << "warning:: no adjacent face for the " << i << "-th face" << endl;
		}
	}

}

bool SurfaceMesh::_checkAdjacencyOfFaces(int fidx1, int fidx2)
{
	if ( fidx1 < 0 || fidx1 >= faces.size() ) return false;
	if ( fidx2 < 0 || fidx2 >= faces.size() ) return false;
	if ( fidx1 == fidx2 ) return false;

	// find edges of the faces
	vector< std::pair<int,int> > vpair1, vpair2;
	switch ( faces[fidx1].elem_type ) {
	case MT_TRIANGLE:
		vpair1.push_back(std::make_pair(faces[fidx1].vertex_indices[0], faces[fidx1].vertex_indices[1]));
		vpair1.push_back(std::make_pair(faces[fidx1].vertex_indices[1], faces[fidx1].vertex_indices[2]));
		vpair1.push_back(std::make_pair(faces[fidx1].vertex_indices[2], faces[fidx1].vertex_indices[0]));
		break;
	case MT_QUAD:
		vpair1.push_back(std::make_pair(faces[fidx1].vertex_indices[0], faces[fidx1].vertex_indices[1]));
		vpair1.push_back(std::make_pair(faces[fidx1].vertex_indices[1], faces[fidx1].vertex_indices[2]));
		vpair1.push_back(std::make_pair(faces[fidx1].vertex_indices[2], faces[fidx1].vertex_indices[3]));
		vpair1.push_back(std::make_pair(faces[fidx1].vertex_indices[3], faces[fidx1].vertex_indices[0]));
		break;
	}
	switch ( faces[fidx2].elem_type ) {
	case MT_TRIANGLE:
		vpair2.push_back(std::make_pair(faces[fidx2].vertex_indices[0], faces[fidx2].vertex_indices[1]));
		vpair2.push_back(std::make_pair(faces[fidx2].vertex_indices[1], faces[fidx2].vertex_indices[2]));
		vpair2.push_back(std::make_pair(faces[fidx2].vertex_indices[2], faces[fidx2].vertex_indices[0]));
		break;
	case MT_QUAD:
		vpair2.push_back(std::make_pair(faces[fidx2].vertex_indices[0], faces[fidx2].vertex_indices[1]));
		vpair2.push_back(std::make_pair(faces[fidx2].vertex_indices[1], faces[fidx2].vertex_indices[2]));
		vpair2.push_back(std::make_pair(faces[fidx2].vertex_indices[2], faces[fidx2].vertex_indices[3]));
		vpair2.push_back(std::make_pair(faces[fidx2].vertex_indices[3], faces[fidx2].vertex_indices[0]));
		break;
	}

	// check if there is a common edge
	for (size_t i=0; i<vpair1.size(); i++) {
		for (size_t j=0; j<vpair2.size(); j++) {
			if ( vpair1[i] == std::make_pair(vpair2[j].second, vpair2[j].first) ) {
				return true;
			}
		}
	}

	return false;
}

Vec3 SurfaceMesh::_calcFaceNormal(int fidx)
{
	if ( fidx < 0 || fidx >= faces.size() ) return Vec3(0,0,0);

	Vec3 n(0,0,0);
	if ( faces[fidx].elem_type == MT_TRIANGLE ) { 
		Vec3 &x0 = vertices[faces[fidx].vertex_indices[0]];
		Vec3 &x1 = vertices[faces[fidx].vertex_indices[1]];
		Vec3 &x2 = vertices[faces[fidx].vertex_indices[2]];
		n = Cross(x1-x0,x2-x0); n.Normalize();
	} else if ( faces[fidx].elem_type == MT_QUAD ) { 
		Vec3 &x0 = vertices[faces[fidx].vertex_indices[0]];
		Vec3 &x1 = vertices[faces[fidx].vertex_indices[1]];
		Vec3 &x2 = vertices[faces[fidx].vertex_indices[2]];
		Vec3 &x3 = vertices[faces[fidx].vertex_indices[3]];
		Vec3 n1 = Cross(x1-x0,x2-x0); n1.Normalize();
		Vec3 n2 = Cross(x2-x0,x3-x0); n2.Normalize();
		n = n1 + n2; n.Normalize();
	}

	return n;
}

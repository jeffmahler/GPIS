//================================================================================
//         RIGID SURFACE : SURFACE ATTACHED TO A RIGID BODY
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

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <algorithm>
#include "rigidsurface.h"
#include "glsub.h"
#include "cdfunc.h"

using namespace std;

static GLUquadricObj *qobj = gluNewQuadric();

bool RigidSurface::getReady()
{
	if ( !SurfaceMesh::getReady() ) return false;

	buildBoundingBox(); 

	size_t n = vertices.size(); 
	_bContact.resize(n); 
	_bContactPrev.resize(n);
	_xf_ref.resize(n); 
	_fc.resize(n); 
	_bStaticFriction.resize(n);
	for (size_t i=0; i<n; i++) {
		_bContact[i] = 0;
		_bContactPrev[i] = 0;
		_xf_ref[i].SetZero();
		_fc[i].SetZero();
		_bStaticFriction[i] = 1;
	}

	_scanSurfacePatches();

	return true;
}

void RigidSurface::resetContactInfo()
{
	for (vector<int>::iterator iter_idx = _collidingVertexIndices.begin(); iter_idx != _collidingVertexIndices.end(); iter_idx++) {
		_bContact[*iter_idx] = 0;
		_fc[*iter_idx].SetZero();
		_bStaticFriction[*iter_idx] = 1;
	}
	_collidingVertexIndices.clear();

	for (vector<int>::iterator iter_idx = _collidingSurfPatchIndices.begin(); iter_idx != _collidingSurfPatchIndices.end(); iter_idx++) {
		_bColSurfPatch[*iter_idx] = 0;
	}
	_collidingSurfPatchIndices.clear();

	for (vector<int>::iterator iter_idx = _seedVertexIndices.begin(); iter_idx != _seedVertexIndices.end(); iter_idx++) {
		_bColSeedVertex[*iter_idx] = 0;
	}
	_seedVertexIndices.clear();
}

void RigidSurface::render() 
{ 
	if ( _pT == NULL ) return;
	glPushMatrix(); 
#if GEAR_DOUBLE_PRECISION
	glMultMatrixd(_pT->GetArray()); 
#else
	glMultMatrixf(_pT->GetArray()); 
#endif
	SurfaceMesh::render(); 
	glPopMatrix(); 

	if ( _bRenderingBoundingBox ) {
		GLfloat previous_color[4];
		glGetFloatv(GL_CURRENT_COLOR, previous_color);
		glColor3f(1,0,0);
		glPushMatrix();
#if GEAR_DOUBLE_PRECISION
		glMultMatrixd(SE3(_obb.orientation, _obb.position).GetArray());
#else
		glMultMatrixf(SE3(_obb.orientation, _obb.position).GetArray());
#endif
		glScalef((GLfloat)_obb.extents[0], (GLfloat)_obb.extents[1], (GLfloat)_obb.extents[2]);
		glutWireCube(2.0);
		glPopMatrix();
		glColor4fv(previous_color);
	}

	if ( _bRenderingContactPoints ) {
		GLfloat previous_color[4];
		glGetFloatv(GL_CURRENT_COLOR, previous_color);
		Vec3 p;
		for (size_t i=0; i<vertices.size(); i++) {
			if ( _bContact[i] ) {
				if ( _bStaticFriction[i] ) {
					glColor3f(1,0.5,0);		// orange for static friction force
				} else {
					glColor3f(1,0,1);		// magenta for dynamic friction force
				}
				p = (*_pT) * vertices[i];
				glPushMatrix();
				glTranslated(p[0], p[1], p[2]);
				gluSphere(qobj, 0.001, 10, 10);
				glPopMatrix();
			}
		}
		glColor4fv(previous_color);
	}

	if ( _bRenderingContactForces ) {
		GLfloat previous_color[4];
		GLfloat previous_linewidth[1];
		glGetFloatv(GL_CURRENT_COLOR, previous_color);
		glGetFloatv(GL_LINE_WIDTH, previous_linewidth);
		double L; 
		Vec3 p;
		for (size_t i=0; i<vertices.size(); i++) {
			if ( _bContact[i] ) {
				glColor3f(1,0.5,0);
				p = (*_pT) * vertices[i];
				L = _force_scale * Norm(_fc[i]); 
				glPushMatrix();
				glTranslated(p[0], p[1], p[2]);
				if ( _bRenderingContactForcesReversed ) {
					glsub_draw_arrow(qobj, -_fc[i], L*0.7, L*0.3, 0.0005, 0.001, 10, 1, 2);
				} else {
					glsub_draw_arrow(qobj, _fc[i], L*0.7, L*0.3, 0.0005, 0.001, 10, 1, 2);
				}
				glPopMatrix();
			}
		}
		glLineWidth(previous_linewidth[0]);
		glColor4fv(previous_color);
	}

	if ( _bRenderingCollidingSurfacePatches ) {
		glPushMatrix(); 
#if GEAR_DOUBLE_PRECISION
		glMultMatrixd(_pT->GetArray()); 
#else
		glMultMatrixf(_pT->GetArray()); 
#endif
		GLUquadricObj *qobj = gluNewQuadric();
		GLfloat previous_color[4];
		GLfloat previous_linewidth[1];
		glGetFloatv(GL_CURRENT_COLOR, previous_color);
		glGetFloatv(GL_LINE_WIDTH, previous_linewidth);
		glLineWidth(2*previous_linewidth[0]);
		for (size_t i=0; i<vertices.size(); i++) {
			if ( _bColSurfPatch[i] ) {
				glPushMatrix();
				glTranslated(vertices[i][0], vertices[i][1], vertices[i][2]);
				glColor3f(0,0,1);
				gluSphere(qobj, 0.001, 10, 10);
				glPopMatrix();
				glColor3f(1,0,0);
				for (size_t j=0; j<_surfPatch[i]._idxFaces.size(); j++) {
					int fidx = _surfPatch[i]._idxFaces[j];
					switch ( faces[fidx].elem_type ) {
					case MT_TRIANGLE:
						glBegin(GL_TRIANGLES);
						if ( faces[fidx].normal_type == NT_FACE ) {
							glNormal3dv(normals[faces[fidx].normal_indices[0]].GetArray());
						}
						for (int j=0; j<3; j++) {
							if ( faces[fidx].normal_type == NT_VERTEX ) {
								glNormal3dv(normals[faces[fidx].normal_indices[j]].GetArray());
							}
							glVertex3dv(vertices[faces[fidx].vertex_indices[j]].GetArray());
						}
						glEnd();
						break;
					case MT_QUAD:
						glBegin(GL_QUADS);
						if ( faces[fidx].normal_type == NT_FACE ) {
							glNormal3dv(normals[faces[fidx].normal_indices[0]].GetArray());
						}
						for (int j=0; j<4; j++) {
							if ( faces[fidx].normal_type == NT_VERTEX ) {
								glNormal3dv(normals[faces[fidx].normal_indices[j]].GetArray());
							}
							glVertex3dv(vertices[faces[fidx].vertex_indices[j]].GetArray());
						}
						glEnd();
						break;
					}
				}
			}
			if ( _bColSeedVertex[i] ) {
				glPushMatrix();
				glTranslated(vertices[i][0], vertices[i][1], vertices[i][2]);
				glColor3f(0,1,0);
				gluSphere(qobj, 0.0008, 10, 10);
				glPopMatrix();
			}
		}
		glLineWidth(previous_linewidth[0]);
		glColor4fv(previous_color);
		glPopMatrix(); 
	}
}

void RigidSurface::buildBoundingBox(gReal margin_)
{
	const gReal inf = 1E10;
	gReal maxX = -inf, minX = inf, maxY = -inf, minY = inf, maxZ = -inf, minZ = inf;
	gReal *p;

	if ( getNumVertices() == 0 ) {
		_obb.extents[0] = _obb.extents[1] = _obb.extents[2] = 0.0;
		_pos_obb_local.SetZero();
		return;
	}

	for (size_t i=0; i<getNumVertices(); i++) {
		p = vertices[i].GetArray();
		if ( p[0] > maxX ) maxX = p[0];
		if ( p[0] < minX ) minX = p[0];
		if ( p[1] > maxY ) maxY = p[1];
		if ( p[1] < minY ) minY = p[1];
		if ( p[2] > maxZ ) maxZ = p[2];
		if ( p[2] < minZ ) minZ = p[2];
	}

	_obb.extents[0] = 0.5*(maxX-minX) + margin_;
	_obb.extents[1] = 0.5*(maxY-minY) + margin_;
	_obb.extents[2] = 0.5*(maxZ-minZ) + margin_;
	_pos_obb_local = Vec3(0.5*(minX+maxX),0.5*(minY+maxY),0.5*(minZ+maxZ));

	// this handles too thin bounding box
	for (size_t i=0; i<3; i++) {
		if ( fabs(_obb.extents[i]) < fabs(_col_depth_limit) ) {
			bool bpositive=false, bnegative=false;
			Vec3 n(0,0,0); n[i] = 1;
			if ( vertexnormals.size() == vertices.size() ) {
				for (size_t j=0; j<vertices.size(); j++) {
					if ( !bnegative && Inner(vertexnormals[j], n) > 0 ) {
						bnegative = true;
					} 
					if ( !bpositive && Inner(vertexnormals[j], n) < 0 ) {
						bpositive = true;
					}
				}
				if ( bpositive && bnegative ) {
					_obb.extents[i] = fabs(_col_depth_limit);
				} else if ( bpositive && !bnegative ) {
					_pos_obb_local[i] += fabs(_col_depth_limit) - fabs(_obb.extents[i]);
					_obb.extents[i] = fabs(_col_depth_limit);
				} else if ( !bpositive && bnegative ) {
					_pos_obb_local[i] -= fabs(_col_depth_limit) - fabs(_obb.extents[i]);
					_obb.extents[i] = fabs(_col_depth_limit);
				}
			} else {
				_obb.extents[i] = fabs(_col_depth_limit);
			}
		}
	}

	updateBoundingBox();
}

void RigidSurface::updateBoundingBox() 
{ 
	if ( _pT == NULL ) return;
	_obb.position = (*_pT) * _pos_obb_local; 
	_obb.orientation = _pT->GetRotation(); 
}


void RigidSurface::setCollisionDepthLimitRatio(double r)
{ 
	double extent_min = _obb.extents[0] > _obb.extents[1] ? (_obb.extents[1] > _obb.extents[2] ? _obb.extents[2] : _obb.extents[1]) : (_obb.extents[0] > _obb.extents[2] ? _obb.extents[2] : _obb.extents[0]);
	_col_depth_limit = r * extent_min;
}

void RigidSurface::_scanSurfacePatches()
{
	_surfPatch.resize(vertices.size());

	// add faces to the patches
	for (size_t i=0; i<_surfPatch.size(); i++) {
		_surfPatch[i]._idxFaces = _idxAdjFaces[i]; // add the adjacent faces of the vertex

		// an alternative to using voronoi cell computation
		for (size_t j=0; j<_idxAdjFaces[i].size(); j++) {
			// get far side edge vertices
			vector<int> idxev;
			for (size_t k=0; k<3; k++) {
				if ( faces[_idxAdjFaces[i][j]].vertex_indices[k] != i ) {
					idxev.push_back(faces[_idxAdjFaces[i][j]].vertex_indices[k]);
				}
			}
			if ( idxev.size() != 2 ) {
				cout << "error in finding far-side edge!!!" << endl;
				cout << "  center vertex index = " << i << endl;
				cout << "  adjacent face index = " << _idxAdjFaces[i][j] << endl;
				cout << "  vertex indices of the face = (" << faces[_idxAdjFaces[i][j]].vertex_indices[0] << ", " << faces[_idxAdjFaces[i][j]].vertex_indices[1] << ", " << faces[_idxAdjFaces[i][j]].vertex_indices[2] << ")" << endl;
				idx_selected_vertices.push_back(i);
				idx_selected_faces.push_back(_idxAdjFaces[i][j]);
				return;
			}
			// compuate angle
			Vec3 p0 = vertices[idxev[0]]-vertices[i], p1 = vertices[idxev[1]]-vertices[i];
			p0.Normalize(); p1.Normalize();
			gReal ang = acos(Inner(p0, p1));
			// if the angle is larger than 90 deg, add the adjacent faces of the two edge vertices
			if ( ang > 0.5 * 3.14159 ) {
				// find the adjacent face
				for (size_t k=0; k<faces.size(); k++) {
					if ( k == _idxAdjFaces[i][j] )
						continue;
					if ( ( faces[k].vertex_indices[0]==idxev[0] || faces[k].vertex_indices[1]==idxev[0] || faces[k].vertex_indices[2]==idxev[0] )
						|| ( faces[k].vertex_indices[0]==idxev[1] || faces[k].vertex_indices[1]==idxev[1] || faces[k].vertex_indices[2]==idxev[1] ) ) 
					{
						if ( find(_surfPatch[i]._idxFaces.begin(), _surfPatch[i]._idxFaces.end(), k) == _surfPatch[i]._idxFaces.end() ) {
							_surfPatch[i]._idxFaces.push_back(k);
						}
					}
				}
			}
		}

		// add adjacent face, if Dihedral angle is larger than ang_tol
		gReal ang_tol = 45. / 180. * 3.14159;
		vector<int> idxfaces = _surfPatch[i]._idxFaces;
		for (size_t j=0; j<idxfaces.size(); j++) {
			Vec3 n1 = _calcFaceNormal(idxfaces[j]);
			for (size_t k=0; k<_idxAdjFacesOfFaces[idxfaces[j]].size(); k++) {
				int fidx = _idxAdjFacesOfFaces[idxfaces[j]][k];
				if ( find(idxfaces.begin(), idxfaces.end(), fidx) == idxfaces.end() ) {
					Vec3 n2 = _calcFaceNormal(fidx);
					if ( Norm(n1) > 1E-6 && acos(Inner(n1,n2)) > ang_tol ) {
						_surfPatch[i]._idxFaces.push_back(fidx);
					}
				}
			}
		}
		//if ( _surfPatch[i]._idxFaces.size() - idxfaces.size() > 0 ) {
		//	cout << _surfPatch[i]._idxFaces.size() - idxfaces.size() << " faces added" << endl;
		//}
	}

	// find edges in each patch
	for (size_t i=0; i<_surfPatch.size(); i++) {
		for (size_t j=0; j<_surfPatch[i]._idxFaces.size(); j++) {
			vector<int> vidxfj(3), vidxfk(3);
			vidxfj[0] = faces[_surfPatch[i]._idxFaces[j]].vertex_indices[0];
			vidxfj[1] = faces[_surfPatch[i]._idxFaces[j]].vertex_indices[1];
			vidxfj[2] = faces[_surfPatch[i]._idxFaces[j]].vertex_indices[2];
			for (size_t k=j+1; k<_surfPatch[i]._idxFaces.size(); k++) {
				int cnt=0;
				vector<int> vidx;
				vidxfk[0] = faces[_surfPatch[i]._idxFaces[k]].vertex_indices[0];
				vidxfk[1] = faces[_surfPatch[i]._idxFaces[k]].vertex_indices[1];
				vidxfk[2] = faces[_surfPatch[i]._idxFaces[k]].vertex_indices[2];
				if ( find(vidxfj.begin(), vidxfj.end(), vidxfk[0]) != vidxfj.end() ) { vidx.push_back(vidxfk[0]); cnt++; }
				if ( find(vidxfj.begin(), vidxfj.end(), vidxfk[1]) != vidxfj.end() ) { vidx.push_back(vidxfk[1]); cnt++; }
				if ( find(vidxfj.begin(), vidxfj.end(), vidxfk[2]) != vidxfj.end() ) { vidx.push_back(vidxfk[2]); cnt++; }
				if ( cnt == 2 ) {
					_surfPatch[i]._idxEdgeFaces.push_back(make_pair(_surfPatch[i]._idxFaces[j], _surfPatch[i]._idxFaces[k]));
					_surfPatch[i]._idxEdgeVertices.push_back(make_pair(vidx[0], vidx[1]));
				}
			}
		}
	}

	_bColSurfPatch.resize(vertices.size());
	_bColSeedVertex.resize(vertices.size());
	for (size_t i=0; i<_bColSurfPatch.size(); i++) {
		_bColSurfPatch[i] = 0;
		_bColSeedVertex[i] = 0;
	}
}

bool RigidSurface::_checkCollisionFully(double &pdepth, Vec3 &normal, const Vec3 &p)
{
	bool binside, binside_tmp;
	bool bcol, bcol_tmp;
	gReal depthlimit = getCollisionDepthLimit(); // penetration depth must be between 0 to depthlimit

	// check if the point is located inside any of the triangle prisms of the faces
	binside = false;
	bcol = false;
	gReal sd=-1E10, sd_tmp;
	for (size_t i=0; i<faces.size(); i++) {
		int *pidx = faces[i].vertex_indices;
		Vec3 &N = normals[i];
		binside_tmp = dcPointTriangle2(sd_tmp, p, vertices[pidx[0]], vertices[pidx[1]], vertices[pidx[2]], N); 
		if ( binside_tmp && sd_tmp >= 0 ) {
			return false;
		}
		bcol_tmp = (binside_tmp && sd_tmp < 0 && sd_tmp >= -depthlimit);
		if ( binside_tmp ) {
			binside = true;
		}
		if ( bcol_tmp && sd_tmp > sd ) {
			bcol = true;
			normal = N;
			sd = sd_tmp;
		}
	}
	if ( binside ) {
		if ( bcol ) {
			pdepth = -sd;
			return true;
		} else {
			return false;
		}
	}

	return false;
}

bool RigidSurface::_checkCollisionWithSurfacePatch(double &pdepth, Vec3 &contactnormal, int &idxface, int idxsurfpatch, const Vec3 &p, const Vec3 &np)
{
	bool binside, binside_tmp;
	bool bcol, bcol_tmp;
	int idxface_tmp=-1;
	gReal depthlimit = getCollisionDepthLimit(); // penetration depth must be between 0 to depthlimit

	// check if the point is located inside any of the triangle prisms of the faces
	binside = false;
	bcol = false;
	gReal sd=-1E10, sd_tmp;
	for (size_t i=0; i<_surfPatch[idxsurfpatch]._idxFaces.size(); i++) {
		int *pidx = faces[_surfPatch[idxsurfpatch]._idxFaces[i]].vertex_indices;
		Vec3 &nf = normals[_surfPatch[idxsurfpatch]._idxFaces[i]]; // face normal
		binside_tmp = dcPointTriangle2(sd_tmp, p, vertices[pidx[0]], vertices[pidx[1]], vertices[pidx[2]], nf); 
		if ( binside_tmp && sd_tmp >= 0 ) {
			return false;
		}
		bcol_tmp = ( binside_tmp && sd_tmp < 0 && sd_tmp >= -depthlimit );
		if ( Norm(np) > 1E-6 ) {
			bcol_tmp = bcol_tmp && Inner(nf,np) < 0;
		}
		if ( binside_tmp ) {
			binside = true;
		}
		if ( bcol_tmp && sd_tmp > sd ) {
			bcol = true;
			sd = sd_tmp;
			contactnormal = 0.5*(nf-np); contactnormal.Normalize();
			idxface_tmp = _surfPatch[idxsurfpatch]._idxFaces[i];
		}
	}
	if ( binside ) {
		if ( bcol ) {
			pdepth = -sd;
			_bColSurfPatch[idxsurfpatch] = 1;
			_collidingSurfPatchIndices.push_back(idxsurfpatch);
			idxface = idxface_tmp;
			return true;
		} else {
			return false;
		}
	}

	//// check if the point can be projected onto the edges
	//binside = false;
	//bcol = false;
	//gReal dist=1E10, dist_tmp;
	//for (size_t i=0; i<_surfPatch[idxsurfpatch]._idxEdgeVertices.size(); i++) {
	//	Vec3 &x0 = vertices[_surfPatch[idxsurfpatch]._idxEdgeVertices[i].first];
	//	Vec3 &x1 = vertices[_surfPatch[idxsurfpatch]._idxEdgeVertices[i].second];
	//	Vec3 &n0 = normals[_surfPatch[idxsurfpatch]._idxEdgeFaces[i].first];
	//	Vec3 &n1 = normals[_surfPatch[idxsurfpatch]._idxEdgeFaces[i].second];
	//	Vec3 x1_x0 = x1-x0, p_x0 = p-x0;
	//	gReal a = Inner(x1_x0, p_x0);
	//	gReal b = SquareSum(x1_x0);
	//	gReal sd0 = Inner(n0, p_x0);
	//	gReal sd1 = Inner(n1, p_x0);
	//	binside_tmp = (a > 0 && a < b); // true if the point can be projected onto the edge
	//	if ( binside_tmp && sd0 >= 0 && sd1 >= 0 ) {
	//		return false;
	//	}
	//	bcol_tmp = (binside_tmp && sd0 < 0 && sd1 < 0 && sd0 >= -depthlimit && sd1 >= -depthlimit); // true if the point is beneath the two edge faces
	//	if ( binside_tmp ) {
	//		binside = true;
	//	}
	//	if ( bcol_tmp ) {
	//		Vec3 xp_p = x0 + (a/b) * (x1_x0) - p; // vector from p to the projected point
	//		dist_tmp = Norm(xp_p);
	//		if ( dist_tmp < dist ) {
	//			bcol = true;
	//			dist = dist_tmp;
	//			contactnormal = xp_p;
	//			contactnormal.Normalize();
	//		}
	//	}
	//}
	//if ( binside ) {
	//	if ( bcol ) {
	//		pdepth = dist;
	//		return true;
	//	} else {
	//		return false;
	//	}
	//}

	//// check if the vertex is beneath the adjacent faces
	//binside = true;
	//Vec3 p_x = p - vertices[idxsurfpatch];
	//for (size_t i=0; i<_idxAdjFaces[idxsurfpatch].size(); i++) {
	//	Vec3 &n = normals[_idxAdjFaces[idxsurfpatch][i]];
	//	if ( Inner(n, p_x) > 0 ) {
	//		binside = false;
	//		break;
	//	}
	//}
	//if ( binside ) {
	//	gReal d = Norm(p_x);
	//	if ( d < depthlimit ) {
	//		pdepth = d;
	//		contactnormal = -p_x;
	//		contactnormal.Normalize();
	//	}
	//}

	return false;
}

void RigidSurface::_highlightSurfacePatch(int vidx, Vec3 c, DrawType dt)
{
	GLint previous_polygonmode[2];
	GLfloat previous_color[4];
	GLfloat previous_linewidth[1];
	glGetFloatv(GL_CURRENT_COLOR, previous_color);
	glGetIntegerv(GL_POLYGON_MODE, previous_polygonmode);
	glGetFloatv(GL_LINE_WIDTH, previous_linewidth);

	glColor3f((float)c[0],(float)c[1],(float)c[2]);

	switch (dt) {
	case DT_SOLID:
		glPolygonMode(GL_FRONT,GL_FILL);
		break;
	case DT_WIRE:
		glPolygonMode(GL_FRONT,GL_LINE);
		break;
	}

	// faces
	for (size_t i=0; i<_surfPatch[vidx]._idxFaces.size(); i++) {
		int fidx = _surfPatch[vidx]._idxFaces[i];
		switch ( faces[fidx].elem_type ) {
		case MT_TRIANGLE:
			glBegin(GL_TRIANGLES);
			if ( faces[fidx].normal_type == NT_FACE ) {
				glNormal3dv(normals[faces[fidx].normal_indices[0]].GetArray());
			}
			for (int j=0; j<3; j++) {
				if ( faces[fidx].normal_type == NT_VERTEX ) {
					glNormal3dv(normals[faces[fidx].normal_indices[j]].GetArray());
				}
				glVertex3dv(vertices[faces[fidx].vertex_indices[j]].GetArray());
			}
			glEnd();
			break;
		case MT_QUAD:
			glBegin(GL_QUADS);
			if ( faces[fidx].normal_type == NT_FACE ) {
				glNormal3dv(normals[faces[fidx].normal_indices[0]].GetArray());
			}
			for (int j=0; j<4; j++) {
				if ( faces[fidx].normal_type == NT_VERTEX ) {
					glNormal3dv(normals[faces[fidx].normal_indices[j]].GetArray());
				}
				glVertex3dv(vertices[faces[fidx].vertex_indices[j]].GetArray());
			}
			glEnd();
			break;
		}
	}

	// edges
	glLineWidth(5*previous_linewidth[0]);
	glColor3f(0,1,0);
	glBegin(GL_LINES);
	for (size_t i=0; i<_surfPatch[vidx]._idxEdgeVertices.size(); i++) {
		glVertex3dv(vertices[_surfPatch[vidx]._idxEdgeVertices[i].first].GetArray());
		glVertex3dv(vertices[_surfPatch[vidx]._idxEdgeVertices[i].second].GetArray());
	}
	glEnd();

	glColor4fv(previous_color);
	glPolygonMode(previous_polygonmode[0], previous_polygonmode[1]);
	glLineWidth(previous_linewidth[0]);
}

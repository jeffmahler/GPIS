//================================================================================
//         UTILITY TOOLS FOR XML PARSING
// 
//                                                               junggon@gmail.com
//================================================================================

#include <string>
#include <algorithm>
#include <string>
#include <fstream>
#include <math.h>
#include "xmlparser.h"
#include "gear.h"
#include "rigidbody.h"
#include "rigidobject.h"

using namespace std;

static vector<string> _s_names;
static vector< vector<string> > _s_bookmarks;


bool xmlParseObject(TiXmlElement *pelement, RigidObject* pobject, std::string prefix)
{
	if ( !pelement || !pobject ) return false;

	vector<xmlElementProperty> properties;
	xmlScanElementProperties(pelement, properties);

	double mass=0, Ixx=0, Iyy=0, Izz=0, Ixy=0, Ixz=0, Iyz=0;
	bool bmass=false, binertia=false;

	for (size_t i=0; i<properties.size(); i++) {
		string key = properties[i].key;
		string val = properties[i].value;
		int row = properties[i].row;
		if ( key == "name" ) {
			string name = prefix;
			name += string(val);
			pobject->setName(name);
		}
		else if ( key == "mass" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				mass = narr.x[0];
				bmass = true;
			} else {
				cerr << "error:: mass = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "inertia" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 6 ) {
				Ixx = narr.x[0]; Iyy = narr.x[1]; Izz = narr.x[2]; Ixy = narr.x[3]; Ixz = narr.x[4]; Iyz = narr.x[5];
				binertia = true;
			} else {
				cerr << "error:: inertia = [" << val << "]:: numeric value size mismatch:: size must be 6 (Ixx Iyy Izz Ixy Ixz Iyz)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "geom" || key == "geometry" ) {
			if ( !xmlParseObjectGeometry(properties[i].pxmlelement, pobject) ) {
				cerr << "error:: failed in parsing body geometry" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else {
			cout << "warning:: unrecognized body property \"" << key << "\"" << " (line " << row << ")" << endl;
		}
	}

	if ( bmass || binertia ) {
		pobject->setMass(mass, Ixx, Iyy, Izz, Ixy, Ixz, Iyz);
	}

	return true;
}

RigidObject* xmlParseObject(TiXmlElement *pelement, std::string prefix)
{
	RigidObject *pobject = new RigidObject;
	if ( !xmlParseObject(pelement, pobject, prefix) ) {
		delete pobject;
		return NULL;
	}
	return pobject;
}

bool xmlParseObjectGeometry(TiXmlElement *pelement, RigidObject* pobject)
{
	if ( !pelement || !pobject ) return false;

	vector<xmlElementProperty> properties;
	xmlScanElementProperties(pelement, properties);

	enum geomtype { GT_NONE, GT_MESH };
	enum meshformat { MF_NONE, MF_DATA };
	enum drawtype { DT_WIRE, DT_SOLID };

	geomtype gt = GT_NONE;
	meshformat mf = MF_NONE;
	drawtype dt = DT_WIRE;
	string meshfilepath, texturefilepath;
	Vec3 translation(0,0,0);
	SO3 rotation;
	double color[4] = {0,0,1,1}, scale[3] = {1,1,1}, extents[3] = {0,0,0};
	double radius_bottom=0, radius_top=0, height=0, density=0, mass=0, collisiondepthlimit=0;
	int slice;
	bool bcollision=true, brender=true, bhollow=false;
	vector<double> vertices;
	vector<int> faces;

	for (size_t i=0; i<properties.size(); i++) {
		string key = properties[i].key;
		string val = properties[i].value;
		int row = properties[i].row;
		if ( key == "type" ) {
			if ( val == "mesh" ) {
				gt = GT_MESH;
			} else {
				cerr << "error:: unsupported geometry type \"" << val << "\" (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "format" ) {
			if ( val == "data" ) {
				mf = MF_DATA;
			} else {
				cerr << "error:: unsupported mesh format \"" << val << "\" (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "draw" ) {
			if ( val == "wire" ) {
				dt = DT_WIRE;
			} else if ( val == "solid" ) {
				dt = DT_SOLID;
			} else {
				cerr << "error:: unsupported draw type \"" << val << "\" (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "file" ) {
			meshfilepath = val;
		}
		else if ( key == "vertices" ) { // vertices = [x0 y0 z0 x1 y1 z1 ... ]
			xmlNumberArray narr(val);
			vertices = narr.x;
		}
		else if ( key == "faces" ) { // triangular faces  = [f00 f01 f02 f10 f11 f12 f20 f21 f22 f30 f31 f32 f40 f41 f42 ...]
			xmlNumberArrayInt narr(val);
			faces = narr.x;
		}
		else if ( key == "texture" ) {
			texturefilepath = val;
		}
		else if ( key == "scale" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				scale[0] = scale[1] = scale[2] = narr.x[0];
			} else if ( narr.size() == 3 ) {
				scale[0] = narr.x[0]; scale[1] = narr.x[1]; scale[2] = narr.x[2];
			} else {
				cerr << "error:: scale = [" << val << "]:: numeric value size mismatch:: size must be 1 or 3 (sx [sy sz])" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "translation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				translation = narr.get_Vec3();
			} else {
				cerr << "error:: translation = [" << val << "]:: numeric value size mismatch:: size must be 3" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "rotation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 9 ) {
				rotation = narr.get_SO3();
			} else {
				cerr << "error:: rotation = [" << val << "]:: numeric value size mismatch:: size must be 9 (r11 r12 r13 ... r33)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "axisrotation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 4 ) {
				Vec3 n(narr.x[0], narr.x[1], narr.x[2]); n.Normalize();
				gReal theta = narr.x[3] * 3.14159265 / 180.;
				rotation *= Exp(theta*n); // accumulative rotation
			} else {
				cerr << "error:: axisrotation = [" << val << "]:: numeric value size mismatch:: size must be 4::" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "quaternion" || key == "quat" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 4 ) {
				gReal quat[4]; quat[0] = narr.x[0]; quat[1] = narr.x[1]; quat[2] = narr.x[2]; quat[3] = narr.x[3];
				rotation = Quat(quat);
			} else {
				cerr << "error:: quaternion = [" << val << "]:: numeric value size mismatch:: size must be 4::" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "color" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				color[0] = narr.x[0]; color[1] = narr.x[1]; color[2] = narr.x[2];
			} else if ( narr.size() == 4 ) {
				color[0] = narr.x[0]; color[1] = narr.x[1]; color[2] = narr.x[2]; color[3] = narr.x[3];
			} else {
				cerr << "error:: color = [" << val << "]:: numeric value size mismatch:: size must be 3 or 4 (red green blue [alpha])" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "extents" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				extents[0] = narr.x[0]; extents[1] = narr.x[1]; extents[2] = narr.x[2];
			} else {
				cerr << "error:: extents = [" << val << "]:: numeric value size mismatch:: size must be 3" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "radius" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				radius_bottom = radius_top = narr.x[0];
			} else if ( narr.size() == 2 ) {
				radius_bottom = narr.x[0]; radius_top = narr.x[1];
			} else {
				cerr << "error:: radius = [" << val << "]:: numeric value size mismatch:: size must be 1 or 2 (r_bottom [r_top])" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "height" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				height = narr.x[0];
			} else {
				cerr << "error:: height = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "slice" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				slice = (int)narr.x[0];
			} else {
				cerr << "error:: slice = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "collision" ) {
			if ( val == "true" ) {
				bcollision = true;
			} else if ( val == "false" ) {
				bcollision = false;
			} else {
				cerr << "error:: unsupported value \"" << val << "\":: value must be (true|false)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "render" ) {
			if ( val == "true" ) {
				brender = true;
			} else if ( val == "false" ) {
				brender = false;
			} else {
				cerr << "error:: unsupported value \"" << val << "\":: value must be (true|false)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "density" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				density = narr.x[0];
				if ( density < 0 ) {
					cerr << "error:: density = " << density << ":: negative density is not allowed" << " (line " << row << ")" << endl;
					return false;
				}
			} else {
				cerr << "error:: density = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "mass" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				mass = narr.x[0];
				if ( mass < 0 ) {
					cerr << "error:: mass = " << mass << ":: negative mass is not allowed" << " (line " << row << ")" << endl;
					return false;
				}
			} else {
				cerr << "error:: mass = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "hollow" ) {
			if ( val == "true" ) {
				bhollow = true;
			} else if ( val == "false" ) {
				bhollow = false;
			} else {
				cerr << "error:: unsupported value \"" << val << "\":: value must be (true|false)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "collisiondepthlimit" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				collisiondepthlimit = narr.x[0];
				if ( collisiondepthlimit < 0 ) {
					cerr << "error:: collisiondepthlimit = " << collisiondepthlimit << ":: negative collisiondepthlimit is not allowed" << " (line " << row << ")" << endl;
					return false;
				}
			} else {
				cerr << "error:: collisiondepthlimit = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else {
			cout << "warning:: unrecognized body geometry property \"" << key << "\"" << " (line " << row << ")" << endl;
		}
	}

	switch (gt) {
	case GT_MESH:
		switch (mf) {
		case MF_DATA:
			if ( !pobject->addSurfaceTriData(vertices, faces, Vec3(scale[0],scale[1],scale[2]), SE3(rotation, translation), bcollision, brender) ) {
				cerr << "error:: failed in adding surface" << endl;
				return false;
			}
			break;
		default:
			cerr << "error:: unsupported mesh format" << endl;
			return false;
		}
		if ( density > 0 && mass > 0 ) {
			cerr << "error:: mass(=" << mass << ") and density(=" << density << ") cannot be given at the same time!" << endl;
			return false;
		}
		if ( density > 0 ) {
			double m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz;
			if ( bhollow ) {
				cerr << "error:: use \"mass\" (instead of \"density\") to define mass property of a hollow object!" << endl;
				return false;
			} else {
				if ( !pobject->getSurfaceLast()->computeVolumeCentroidAndMomentOfInertia(m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz) ) {
					cerr << "error:: failed in computing volume, centroid and moment of inertia of the mesh" << " (line " << pelement->Row() << ")" << endl;
					return false;
				}
			}
			m *= density; ixx *= density; iyy *= density; izz *= density; ixy *= density; ixz *= density; iyz *= density;
			pobject->addMass(m, ixx, iyy, izz, ixy, ixz, iyz, SE3(rotation, translation) * SE3(Vec3(cx,cy,cz)));
		}
		if ( mass > 0 ) {
			double m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz;
			if ( bhollow ) {
				if ( !pobject->getSurfaceLast()->computeAreaCentroidAndMomentOfInertia(m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz) ) {
					cerr << "error:: failed in computing volume, centroid and moment of inertia of the mesh" << " (line " << pelement->Row() << ")" << endl;
					return false;
				}
			} else {
				if ( !pobject->getSurfaceLast()->computeVolumeCentroidAndMomentOfInertia(m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz) ) {
					cerr << "error:: failed in computing volume, centroid and moment of inertia of the mesh" << " (line " << pelement->Row() << ")" << endl;
					return false;
				}
			}
			density = mass/m;
			m *= density; ixx *= density; iyy *= density; izz *= density; ixy *= density; ixz *= density; iyz *= density;
			pobject->addMass(m, ixx, iyy, izz, ixy, ixz, iyz, SE3(rotation, translation) * SE3(Vec3(cx,cy,cz)));
		}
		break;
	default:
		cerr << "error:: unsupported geometry type" << endl;
		return false;
	}

	RigidSurface *psurf = pobject->getSurfaceLast();
	if ( psurf == NULL ) {
		cerr << "error:: failed in accessing the created geometry" << " (line " << pelement->Row() << ")" << endl;
		return false;
	}

	switch (dt) {
	case DT_WIRE:
		psurf->setDrawType(SurfaceMesh::DT_WIRE);
		break;
	case DT_SOLID:
		psurf->setDrawType(SurfaceMesh::DT_SOLID);
		break;
	default:
		cerr << "error:: unsupported draw type" << endl;
		return false;
	}

	psurf->setColor(color[0],color[1],color[2]);

	if ( collisiondepthlimit > 0 ) {
		psurf->setCollisionDepthLimit(collisiondepthlimit);
	}

	return true;
}

bool xmlParseBody(TiXmlElement *pelement, RigidBody* pbody, std::string prefix)
{
	if ( !pelement || !pbody ) return false;

	vector<xmlElementProperty> properties;
	xmlScanElementProperties(pelement, properties);

	double mass=0, Ixx=0, Iyy=0, Izz=0, Ixy=0, Ixz=0, Iyz=0;
	bool bmass=false, binertia=false;

	for (size_t i=0; i<properties.size(); i++) {
		string key = properties[i].key;
		string val = properties[i].value;
		int row = properties[i].row;
		if ( key == "name" ) {
			string name = prefix;
			name += string(val);
			pbody->setName(name);
		}
		else if ( key == "mass" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				mass = narr.x[0];
				bmass = true;
			} else {
				cerr << "error:: mass = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "inertia" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 6 ) {
				Ixx = narr.x[0]; Iyy = narr.x[1]; Izz = narr.x[2]; Ixy = narr.x[3]; Ixz = narr.x[4]; Iyz = narr.x[5];
				binertia = true;
			} else {
				cerr << "error:: inertia = [" << val << "]:: numeric value size mismatch:: size must be 6 (Ixx Iyy Izz Ixy Ixz Iyz)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "geom" || key == "geometry" ) {
			if ( !xmlParseBodyGeometry(properties[i].pxmlelement, pbody) ) {
				cerr << "error:: failed in parsing body geometry" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else {
			cout << "warning:: unrecognized body property \"" << key << "\"" << " (line " << row << ")" << endl;
		}
	}

	if ( bmass || binertia ) {
		pbody->setMass(mass, Ixx, Iyy, Izz, Ixy, Ixz, Iyz);
	}

	return true;
}

RigidBody* xmlParseBody(TiXmlElement *pelement, std::string prefix)
{
	RigidBody *pbody = new RigidBody;
	if ( !xmlParseBody(pelement, pbody, prefix) ) {
		delete pbody;
		return NULL;
	}
	return pbody;
}

bool xmlParseBodyGeometry(TiXmlElement *pelement, RigidBody* pbody)
{
	if ( !pelement || !pbody ) return false;

	vector<xmlElementProperty> properties;
	xmlScanElementProperties(pelement, properties);

	enum geomtype { GT_NONE, GT_MESH, GT_POINTS };
	enum meshformat { MF_NONE, MF_DATA };
	enum drawtype { DT_WIRE, DT_SOLID };

	geomtype gt = GT_NONE;
	meshformat mf = MF_NONE;
	drawtype dt = DT_WIRE;
	string meshfilepath, texturefilepath;
	Vec3 translation(0,0,0);
	SO3 rotation;
	double color[4] = {0,0,1,1}, scale[3] = {1,1,1}, extents[3] = {0,0,0};
	double radius_bottom=0, radius_top=0, height=0, density=0, mass=0, collisiondepthlimit=0;
	int slice;
	bool bcollision=true, brender=true, bhollow=false;
	vector<double> vertices, vnormals;
	vector<int> faces;

	for (size_t i=0; i<properties.size(); i++) {
		string key = properties[i].key;
		string val = properties[i].value;
		int row = properties[i].row;
		if ( key == "type" ) {
			if ( val == "mesh" ) {
				gt = GT_MESH;
			} else if ( val == "points" ) {
				gt = GT_POINTS;
			} else {
				cerr << "error:: unsupported geometry type \"" << val << "\" (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "format" ) {
			if ( val == "data" ) {
				mf = MF_DATA;
			} else {
				cerr << "error:: unsupported geometry input format \"" << val << "\" (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "draw" ) {
			if ( val == "wire" ) {
				dt = DT_WIRE;
			} else if ( val == "solid" ) {
				dt = DT_SOLID;
			} else {
				cerr << "error:: unsupported draw type \"" << val << "\" (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "file" ) {
			meshfilepath = val;
		}
		else if ( key == "vertices" ) { // vertices = [x0 y0 z0 x1 y1 z1 ... ]
			xmlNumberArray narr(val);
			vertices = narr.x;
		}
		else if ( key == "faces" ) { // triangular faces  = [f00 f01 f02 f10 f11 f12 f20 f21 f22 f30 f31 f32 f40 f41 f42 ...]
			xmlNumberArrayInt narr(val);
			faces = narr.x;
		}
		else if ( key == "vertexnormals" ) { // vertex normals = [n00 n01 n02 n10 n11 n12 n20 n21 n22 n30 n31 n32 n40 n41 n42 ...]
			xmlNumberArray narr(val);
			vnormals = narr.x;
		}
		else if ( key == "texture" ) {
			texturefilepath = val;
		}
		else if ( key == "scale" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				scale[0] = scale[1] = scale[2] = narr.x[0];
			} else if ( narr.size() == 3 ) {
				scale[0] = narr.x[0]; scale[1] = narr.x[1]; scale[2] = narr.x[2];
			} else {
				cerr << "error:: scale = [" << val << "]:: numeric value size mismatch:: size must be 1 or 3 (sx [sy sz])" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "translation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				translation = narr.get_Vec3();
			} else {
				cerr << "error:: translation = [" << val << "]:: numeric value size mismatch:: size must be 3" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "rotation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 9 ) {
				rotation = narr.get_SO3();
			} else {
				cerr << "error:: rotation = [" << val << "]:: numeric value size mismatch:: size must be 9 (r11 r12 r13 ... r33)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "axisrotation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 4 ) {
				Vec3 n(narr.x[0], narr.x[1], narr.x[2]); n.Normalize();
				gReal theta = narr.x[3] * 3.14159265 / 180.;
				rotation *= Exp(theta*n); // accumulative rotation
			} else {
				cerr << "error:: axisrotation = [" << val << "]:: numeric value size mismatch:: size must be 4::" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "quaternion" || key == "quat" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 4 ) {
				gReal quat[4]; quat[0] = narr.x[0]; quat[1] = narr.x[1]; quat[2] = narr.x[2]; quat[3] = narr.x[3];
				rotation = Quat(quat);
			} else {
				cerr << "error:: quaternion = [" << val << "]:: numeric value size mismatch:: size must be 4::" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "color" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				color[0] = narr.x[0]; color[1] = narr.x[1]; color[2] = narr.x[2];
			} else if ( narr.size() == 4 ) {
				color[0] = narr.x[0]; color[1] = narr.x[1]; color[2] = narr.x[2]; color[3] = narr.x[3];
			} else {
				cerr << "error:: color = [" << val << "]:: numeric value size mismatch:: size must be 3 or 4 (red green blue [alpha])" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "extents" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				extents[0] = narr.x[0]; extents[1] = narr.x[1]; extents[2] = narr.x[2];
			} else {
				cerr << "error:: extents = [" << val << "]:: numeric value size mismatch:: size must be 3" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "radius" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				radius_bottom = radius_top = narr.x[0];
			} else if ( narr.size() == 2 ) {
				radius_bottom = narr.x[0]; radius_top = narr.x[1];
			} else {
				cerr << "error:: radius = [" << val << "]:: numeric value size mismatch:: size must be 1 or 2 (r_bottom [r_top])" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "height" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				height = narr.x[0];
			} else {
				cerr << "error:: height = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "slice" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				slice = (int)narr.x[0];
			} else {
				cerr << "error:: slice = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "collision" ) {
			if ( val == "true" ) {
				bcollision = true;
			} else if ( val == "false" ) {
				bcollision = false;
			} else {
				cerr << "error:: unsupported value \"" << val << "\":: value must be (true|false)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "render" ) {
			if ( val == "true" ) {
				brender = true;
			} else if ( val == "false" ) {
				brender = false;
			} else {
				cerr << "error:: unsupported value \"" << val << "\":: value must be (true|false)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "density" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				density = narr.x[0];
				if ( density < 0 ) {
					cerr << "error:: density = " << density << ":: negative density is not allowed" << " (line " << row << ")" << endl;
					return false;
				}
			} else {
				cerr << "error:: density = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "mass" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				mass = narr.x[0];
				if ( mass < 0 ) {
					cerr << "error:: mass = " << density << ":: negative mass is not allowed" << " (line " << row << ")" << endl;
					return false;
				}
			} else {
				cerr << "error:: mass = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "hollow" ) {
			if ( val == "true" ) {
				bhollow = true;
			} else if ( val == "false" ) {
				bhollow = false;
			} else {
				cerr << "error:: unsupported value \"" << val << "\":: value must be (true|false)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "collisiondepthlimit" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				collisiondepthlimit = narr.x[0];
				if ( collisiondepthlimit < 0 ) {
					cerr << "error:: collisiondepthlimit = " << collisiondepthlimit << ":: negative collisiondepthlimit is not allowed" << " (line " << row << ")" << endl;
					return false;
				}
			} else {
				cerr << "error:: collisiondepthlimit = [" << val << "]:: numeric value size mismatch:: size must be 1" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else {
			cout << "warning:: unrecognized body geometry property \"" << key << "\"" << " (line " << row << ")" << endl;
		}
	}

	switch (gt) {
	case GT_MESH:
		switch (mf) {
		case MF_DATA:
			if ( vnormals.size() > 0 ) {
				vnormals.clear();
				cout << "warning:: vertexnormals are ignored in type \"mesh\"" << " (line " << pelement->Row() << ")" << endl;
			}
			if ( !pbody->addSurfaceTriData(vertices, faces, Vec3(scale[0],scale[1],scale[2]), SE3(rotation, translation), bcollision, brender) ) {
				cerr << "error:: failed in adding surface" << endl;
				return false;
			}
			break;
		default:
			cerr << "error:: unsupported format" << endl;
			return false;
		}
		if ( density > 0 && mass > 0 ) {
			cerr << "error:: mass(=" << mass << ") and density(=" << density << ") cannot be given at the same time!" << endl;
			return false;
		}
		if ( density > 0 ) {
			double m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz;
			if ( bhollow ) {
				cerr << "error:: use \"mass\" (instead of \"density\") to define mass property of a hollow object!" << endl;
				return false;
			} else {
				if ( !pbody->getSurfaceLast()->computeVolumeCentroidAndMomentOfInertia(m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz) ) {
					cerr << "error:: failed in computing volume, centroid and moment of inertia of the mesh" << " (line " << pelement->Row() << ")" << endl;
					return false;
				}
			}
			m *= density; ixx *= density; iyy *= density; izz *= density; ixy *= density; ixz *= density; iyz *= density;
			pbody->addMass(m, ixx, iyy, izz, ixy, ixz, iyz, SE3(rotation, translation) * SE3(Vec3(cx,cy,cz)));
		}
		if ( mass > 0 ) {
			double m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz;
			if ( bhollow ) {
				if ( !pbody->getSurfaceLast()->computeAreaCentroidAndMomentOfInertia(m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz) ) {
					cerr << "error:: failed in computing area, centroid and moment of inertia of the mesh" << " (line " << pelement->Row() << ")" << endl;
					return false;
				}
			} else {
				if ( !pbody->getSurfaceLast()->computeVolumeCentroidAndMomentOfInertia(m,cx,cy,cz,ixx,iyy,izz,ixy,ixz,iyz) ) {
					cerr << "error:: failed in computing volume, centroid and moment of inertia of the mesh" << " (line " << pelement->Row() << ")" << endl;
					return false;
				}
			}
			density = mass/m;
			m *= density; ixx *= density; iyy *= density; izz *= density; ixy *= density; ixz *= density; iyz *= density;
			pbody->addMass(m, ixx, iyy, izz, ixy, ixz, iyz, SE3(rotation, translation) * SE3(Vec3(cx,cy,cz)));
		}
		break;
	case GT_POINTS:
		switch (mf) {
		case MF_DATA:
			if ( faces.size() > 0 ) {
				faces.clear();
				cout << "warning:: faces are ignored in type \"points\"" << " (line " << pelement->Row() << ")" << endl;
			}
			if ( !pbody->addSurfaceVtxData(vertices, vnormals, Vec3(scale[0],scale[1],scale[2]), SE3(rotation, translation), bcollision, brender) ) {
				cerr << "error:: failed in adding surface" << endl;
				return false;
			}
			break;
		default:
			cerr << "error:: unsupported format" << endl;
			return false;
		}
		if ( density > 0 ) {
			cout << "warning:: density is ignored in type \"points\"" << " (line " << pelement->Row() << ")" << endl;
		}
		break;

	default:
		cerr << "error:: unsupported geometry type" << endl;
		return false;
	}

	RigidSurface *psurf = pbody->getSurfaceLast();
	if ( psurf == NULL ) {
		cerr << "error:: failed in accessing the created geometry" << " (line " << pelement->Row() << ")" << endl;
		return false;
	}

	switch (dt) {
	case DT_WIRE:
		psurf->setDrawType(SurfaceMesh::DT_WIRE);
		break;
	case DT_SOLID:
		psurf->setDrawType(SurfaceMesh::DT_SOLID);
		break;
	default:
		cerr << "error:: unsupported draw type" << endl;
		return false;
	}

	psurf->setColor(color[0],color[1],color[2]);

	if ( collisiondepthlimit > 0 ) {
		psurf->setCollisionDepthLimit(collisiondepthlimit);
	}

	return true;
}

GJoint* xmlParseJoint(TiXmlElement *pelement, std::vector<RigidBody*> pbodies, std::string prefix)
{
	if ( !pelement || pbodies.size() < 2 ) return false;

	vector<xmlElementProperty> properties;
	xmlScanElementProperties(pelement, properties);

	GJoint::JointType jtype = GJoint::GJOINT_REVOLUTE;
	string name;
	double axis[3] = {1,0,0}, lowerjointlimit = -1E6, upperjointlimit = 1E6;
	Vec3 translation1(0,0,0), translation2(0,0,0);
	SO3 rotation1, rotation2;
	GBody *pbody1=NULL, *pbody2=NULL;
	int cnt=0;

	for (size_t i=0; i<properties.size(); i++) {
		string key = properties[i].key;
		string val = properties[i].value;
		int row = properties[i].row;
		if ( key == "name" ) {
			name = prefix;
			name += val;
		}
		else if ( key == "type" ) {
			if ( val == "fixed" || val == "welding" ) {
				jtype = GJoint::GJOINT_FIXED;
			} 
			else if ( val == "revolute" || val == "hinge" ) {
				jtype = GJoint::GJOINT_REVOLUTE;
			} 
			else {
				cerr << "error:: unsupported joint type \"" << val << "\"" << " (line " << row << ")" << endl;
				return NULL;
			}
		}
		else if ( key == "axis" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				axis[0] = narr.x[0]; axis[1] = narr.x[1]; axis[2] = narr.x[2];
			} else {
				cerr << "error:: axis = [" << val << "]:: numeric value size mismatch:: size must be 3 " << " (line " << row << ")" << endl;
				return NULL;
			}
		}
		else if ( key == "limitsdeg" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 2 ) {
				lowerjointlimit = narr.x[0]/180.*3.14159265; upperjointlimit = narr.x[1]/180.*3.14159265;
			} else {
				cerr << "error:: limitsdeg = [" << val << "]:: numeric value size mismatch:: size must be 2 (lowerlimit, upperlimit) " << " (line " << row << ")" << endl;
				return NULL;
			}
		}
		else if ( key == "limitsrad" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 2 ) {
				lowerjointlimit = narr.x[0]; upperjointlimit = narr.x[1];
			} else {
				cerr << "error:: limitsrad = [" << val << "]:: numeric value size mismatch:: size must be 2 (lowerlimit, upperlimit) " << " (line " << row << ")" << endl;
				return NULL;
			}
		}
		else if ( key == "connection" ) {
			vector<xmlElementProperty> connection_properties;
			xmlScanElementProperties(properties[i].pxmlelement, connection_properties);

			RigidBody *pbody=NULL;
			Vec3 translation(0,0,0);
			SO3 rotation;
			for (size_t j=0; j<connection_properties.size(); j++) {
				string ckey = connection_properties[j].key;
				string cval = connection_properties[j].value;
				int crow = connection_properties[j].row;
				if ( ckey == "body" ) {
					string bodyname = prefix;
					bodyname += cval;
					for (size_t k=0; k<pbodies.size(); k++) {
						if ( pbodies[k]->getName() == bodyname ) {
							pbody = pbodies[k];
						}
					}
					if ( pbody == NULL ) {
						cerr << "error:: undefined body = [" << cval << "]" << " (line " << crow << ")" << endl;
						return NULL;
					}
				} else if ( ckey == "translation" ) {
					xmlNumberArray narr(cval);
					if ( narr.size() == 3 ) {
						translation = narr.get_Vec3();
					} else {
						cerr << "error:: translation = [" << cval << "]:: numeric value size mismatch:: size must be 3" << " (line " << crow << ")" << endl;
						return NULL;
					}
				} else if ( ckey == "rotation" ) {
					xmlNumberArray narr(cval);
					if ( narr.size() == 9 ) {
						rotation = narr.get_SO3();
					} else {
						cerr << "error:: rotation = [" << cval << "]:: numeric value size mismatch:: size must be 9 (r11 r12 r13 ... r33)" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( ckey == "axisrotation" ) {
					xmlNumberArray narr(cval);
					if ( narr.size() == 4 ) {
						Vec3 n(narr.x[0], narr.x[1], narr.x[2]); n.Normalize();
						gReal theta = narr.x[3] * 3.14159265 / 180.;
						rotation *= Exp(theta*n); // accumulative rotation
					} else {
						cerr << "error:: axisrotation = [" << cval << "]:: numeric value size mismatch:: size must be 4::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( ckey == "quaternion" || ckey == "quat" ) {
					xmlNumberArray narr(cval);
					if ( narr.size() == 4 ) {
						gReal quat[4]; quat[0] = narr.x[0]; quat[1] = narr.x[1]; quat[2] = narr.x[2]; quat[3] = narr.x[3];
						rotation = Quat(quat);
					} else {
						cerr << "error:: quaternion = [" << cval << "]:: numeric value size mismatch:: size must be 4::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
			}
			if ( cnt==0 ) {
				pbody1 = pbody;
				translation1 = translation;
				rotation1 = rotation;
				cnt++;
			} else if ( cnt==1 ) {
				pbody2 = pbody;
				translation2 = translation;
				rotation2 = rotation;
				cnt++;
			} else {
				cout << "warning:: connection ignored (two connections have already set)" << " (line " << row << ")" << endl;
			}
		}
		else {
			cout << "warning:: unrecognized joint property \"" << key << "\"" << " (line " << row << ")" << endl;
		}
	}

	if ( pbody1 == NULL || pbody2 == NULL ) {
		cerr << "error:: two connections must be defined" << endl;
		return NULL;
	}

	GJoint *pjoint;

	switch ( jtype ) {
	case GJoint::GJOINT_FIXED:
		pjoint = new GJointFixed;
		break;
	case GJoint::GJOINT_REVOLUTE:
		pjoint = new GJointRevolute;
		((GJointRevolute*)pjoint)->setAxis(axis[0], axis[1], axis[2]);
		((GJointRevolute*)pjoint)->coordinate.qLL = lowerjointlimit;
		((GJointRevolute*)pjoint)->coordinate.qUL = upperjointlimit;
		break;
	default:
		cerr << "error:: unsupported joint type" << endl;
		return NULL;
	}

	pjoint->setName(name);
	pjoint->connectBodies(pbody1, pbody2);
	pjoint->setPositionAndOrientation(SE3(rotation1,translation1), SE3(rotation2,translation2));

	return pjoint;
}

void xmlScanElementProperties(TiXmlElement *pelement, std::vector<xmlElementProperty> &properties)
{
	// bookmark index for the element
	int index = -1;
	for (int i=0; i<_s_names.size(); i++) {
		if ( _s_names[i] == pelement->Value() ) {
			index = i;
			break;
		}
	}
	if ( index < 0 ) {
		cerr << "error:: invalid elements \"" << pelement->Value() << "\"" << endl;
		return;
	}

	// scan attributes
	TiXmlAttribute *attrib = pelement->FirstAttribute();
	while ( attrib ) {
		string name = attrib->Name();
		string val = attrib->Value();
		transform(name.begin(), name.end(), name.begin(), ::tolower);
		transform(val.begin(), val.end(), val.begin(), ::tolower);
		properties.push_back(xmlElementProperty(name.c_str(), val.c_str(), attrib->Row(), attrib->Column()));
		attrib = attrib->Next();
	}

	// scan child elements
	TiXmlNode* child = 0;
	while ( child = pelement->IterateChildren( child ) ) {
		if ( child->Type() == TiXmlNode::TINYXML_ELEMENT ) {
			string val = child->Value();
			transform(val.begin(), val.end(), val.begin(), ::tolower);
			if ( find(_s_bookmarks[index].begin(), _s_bookmarks[index].end(), val) != _s_bookmarks[index].end() ) { // if val is one of the head keywords, save its element pointer
				properties.push_back(xmlElementProperty(val.c_str(), (TiXmlElement*)child, child->Row(), child->Column()));
			} else {
				string text;
				for ( TiXmlNode *grandchild=child->FirstChild(); grandchild != 0; grandchild = grandchild->NextSibling()) {
					if ( grandchild->Type() == TiXmlNode::TINYXML_TEXT ) {
						text = grandchild->ToText()->Value();
						break;
					}
				}
				if ( text.length() == 0 ) {
					cerr << "warning:: unrecognized element \"" << val << "\" in line = " << child->Row() << endl;
				} else {
					transform(text.begin(), text.end(), text.begin(), ::tolower);
					properties.push_back(xmlElementProperty(val.c_str(), text.c_str(), child->Row(), child->Column()));
				}
			}
		}
	}
}

void xmlLoadKeywords(const char *filepath)
{
	vector< vector<string> > keywords;
	string str;
	ifstream fin(filepath);
	getline(fin, str);
	while (fin) {
		vector<string> keywords_in_line;
		const char *delimiters = " ,\t\n\r";
		char *str2 = new char [str.length()+1]; strcpy(str2, str.c_str());
		char *tok = strtok(str2, delimiters);
		while (tok != NULL) {
			keywords_in_line.push_back(std::string(tok));
			tok = strtok(NULL, delimiters);
		}
		delete [] str2;
		keywords.push_back(keywords_in_line);
		getline(fin, str);
	}
	fin.close();

	xmlLoadKeywords(keywords);
}

void xmlLoadKeywords(const std::vector< std::vector< std::string > > &keywords)
{
	_s_names.clear();
	_s_bookmarks.clear();

	for (size_t i=0; i<keywords.size(); i++) {
		if ( keywords[i].size() == 0 )
			continue;
		_s_names.push_back(keywords[i][0]);
		vector<string> tmp;
		for (size_t j=1; j<keywords[i].size(); j++) {
			tmp.push_back(keywords[i][j]);
		}
		_s_bookmarks.push_back(tmp);
	}
}

void xmlPrintKeywords()
{
	cout << "keywords:" << endl;
	for (size_t i=0; i<_s_names.size(); i++) {
		cout << _s_names[i] << " : ";
		for (size_t j=0; j<_s_bookmarks[i].size(); j++) {
			cout << _s_bookmarks[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

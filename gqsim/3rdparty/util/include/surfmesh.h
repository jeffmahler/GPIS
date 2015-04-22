//================================================================================
//         SURFACE MESH
// 
//                                                               junggon@gmail.com
//================================================================================

#ifndef _SURFACE_MESH_
#define _SURFACE_MESH_

#include <vector>
#include <string>
#include "liegroup.h"

// SurfaceMesh mesh class

class SurfaceMesh
{
public:
	enum FaceElementType { 
		MT_NONE = 0,
		MT_TRIANGLE = 3, 
		MT_QUAD = 4,
	};

	enum NormalType {
		NT_NONE = 0,	// normal is not given
		NT_VERTEX = 1,	// normal is given as vertex normals
		NT_FACE = 2,	// normal is given as face normals (only Face::normal_indices[0] will be used)
	};

	enum DrawType {
		DT_WIRE = 0,
		DT_SOLID = 1,
	};

	class Face 
	{
	public:
		FaceElementType elem_type;
		NormalType normal_type;

		int vertex_indices[4];
		int normal_indices[4];

		Face() : elem_type(MT_NONE), normal_type(NT_NONE) { 
			vertex_indices[0]=vertex_indices[1]=vertex_indices[2]=vertex_indices[3]=-1; 
			normal_indices[0]=normal_indices[1]=normal_indices[2]=normal_indices[3]=-1; 
		}
		~Face() {}
	};

public:
	std::vector<Vec3> vertices; // array of vertex positions
	std::vector<Vec3> normals;  // array of normal vectors
	std::vector<Vec3> textures; // array of texture coordinates
	std::vector<Face> faces;    // array of faces
	std::vector<Vec3> vertexnormals; // array of vertex normal vectors

	std::vector< std::vector<int> > _idxAdjVertices;	// _idxAdjVertices[i] = adjacent vertex indices of the i-th vertex
	std::vector< std::vector<int> > _idxAdjFaces;		// _idxAdjFaces[i] = adjacent face indices of the i-th vertex
	std::vector< std::vector<int> > _idxAdjFacesOfFaces; // _idxAdjFacesOfFaces[i] = adjacent face indices of the i-th face (faces sharing an edge)

	std::string name; 
	bool bRendering, bRenderingVertices, bRenderingVertexNormals, bRenderingFaceNormals;
	DrawType draw_type;
	float color[3], color_specular[3], shininess, alpha;
	float vertex_radius, normallength;
	std::vector<int> idx_selected_vertices, idx_selected_faces;
	float color_selected[3], radius_ratio_selected;

public:
	SurfaceMesh() : name(""), bRendering(true), bRenderingVertices(false), bRenderingVertexNormals(false), bRenderingFaceNormals(false), draw_type(DT_WIRE), vertex_radius(0.001), normallength(0.005), radius_ratio_selected(1.2) { color[0]=0; color[1]=0; color[2]=1; color_selected[0] = 1; color_selected[1] = 0; color_selected[2] = 0; color_specular[0] = color_specular[1] = color_specular[2] = 0.75; shininess = 32; alpha = 1; }
	~SurfaceMesh() {}

	// --------- basic functions ----------

	// load serialized vertex and face data
	// v = serialized vertex data = [x0 y0 z0 x1 y1 z1 ... ], size(v) must be a multiple of 3
	// f = serialized triangular face data (vertex indices), [f00 f01 f02 f10 f11 f12 f20 f21 f22 f30 f31 f32 f40 f41 f42 ...], size(f) must be a multiple of 3
	bool loadFromDataTri(const std::vector<double> &v, const std::vector<int> &f, Vec3 scale = Vec3(1,1,1), SE3 T0 = SE3());

	// load serialized vertex and normal data
	// v = serialized vertex data = [x0 y0 z0 x1 y1 z1 ... ], size(v) must be a multiple of 3
	// vn = serialized vertex normal data = [n00 n01 n02 n10 n11 n12 n20 n21 n22 n30 n31 n32 n40 n41 n42 ...], size(vn) must be same to size(v)
	bool loadFromDataVtx(const std::vector<double> &v, const std::vector<double> &vn, Vec3 scale = Vec3(1,1,1), SE3 T0 = SE3());

	// save to file
	void saveToFileSTL(const char *filepath, double scale=1);	// STL file format
	void saveToFileTRI(const char *filepath, double scale=1);	// TRI file format
	void saveToFileOBJ(const char *filepath, double scale=1);	// OBJ file format

	// get the surface ready to use
	virtual bool getReady();

	// render the surface
	virtual void render();

	// --------- get/set functions ----------

	size_t getNumVertices() { return vertices.size(); }
	size_t getNumFaces() { return faces.size(); }
	                                                                               
	void setName(std::string str) { name = str; }
	std::string getName() { return name; }

	void enableRendering(bool b) { bRendering = b; }
	void enableRenderingVertices(bool b) { bRenderingVertices = b; }
	bool isEnabledRendering() { return bRendering; }
	bool isEnabledRenderingVertices() { return bRenderingVertices; }

	void setColor(double r, double g, double b) { color[0] = r; color[1] = g; color[2] = b; }
	Vec3 getColor() { return Vec3(color[0],color[1],color[2]); }

	void setDrawType(DrawType d) { draw_type = d; }
	DrawType getDrawType() { return draw_type; }

	// -------- utility functions --------

	// update face normals with vertex positions (use this when vertex positions have been changed.)
	void updateFaceNormalsWithVertexPositions(); 

	// updata vertex normals
	void updateVertexNormalsByAveragingAdjacentFaceNormals();
	
	// compute volume, centroid and the moment of inertia about the centroid (assumption: uniform density across the volume)
	bool computeVolumeCentroidAndMomentOfInertia(double &vol, double &Cx, double &Cy, double &Cz, double &Ixx, double &Iyy, double &Izz, double &Ixy, double &Ixz, double &Iyz);

	// compute surface area, centroid, and the moment of inertia about the centroid (assumption: thin walled surface, uniform density across the surface)
	bool computeAreaCentroidAndMomentOfInertia(double &area, double &Cx, double &Cy, double &Cz, double &Ixx, double &Iyy, double &Izz, double &Ixy, double &Ixz, double &Iyz);

	// search the nearest vertex from x and return its index (finds a local optima from the seed vertex index)
	int searchNearestVertex(const Vec3 &x, int seedidx);
	int searchNearestVertex(const Vec3 &x, int seedidx, const Vec3 &N); // N = normal vector of x

	// sub-functions
	void _scanAdjacentFacesAndVertices(); // scan adjacent faces and vertices of each vertex
	void _removeVertexRedundancy(double eps=0);	// remove vertex redundancy
	bool _checkAdjacencyOfFaces(int fidx1, int fidx2); // check if the two faces are sharing an edge (Note: if same faces are given, return false.)
	Vec3 _calcFaceNormal(int fidx); // calculate the normal vector of the face and return it
};


#endif // end of _SURFACE_MESH_


//================================================================================
//         RIGID BODY INHERITED FROM GBODY
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

#include <math.h>
#include <assert.h>
#include "rigidbody.h"
#include "glsub.h"

using namespace std;


bool RigidBody::addSurfaceTriData(const std::vector<double> &x, const std::vector<int> &f, Vec3 scale, SE3 T_ref, bool bcol, bool brender)
{
	RigidSurface *psurf = new RigidSurface(&T_global, &V, &Fe);
	psurf->_parentbodyname = this->getName();

	if ( !psurf->loadFromDataTri(x, f, scale, T_ref) ) {
		return false;
	}

	// collision, rendering
	psurf->enableCollision(bcol);
	psurf->enableRendering(brender);
	if ( brender && psurf->getNumFaces() == 0 ) {
		psurf->enableRenderingVertices(true);
	}

	// add the surface pointer to pSurfs
	pSurfs.push_back(psurf);

	return true;
}

bool RigidBody::addSurfaceVtxData(const std::vector<double> &x, const std::vector<double> &n, Vec3 scale, SE3 T_ref, bool bcol, bool brender)
{
	RigidSurface *psurf = new RigidSurface(&T_global, &V, &Fe);
	psurf->_parentbodyname = this->getName();

	if ( !psurf->loadFromDataVtx(x, n, scale, T_ref) ) {
		return false;
	}

	// collision, rendering
	psurf->enableCollision(bcol);
	psurf->enableRendering(brender);
	psurf->enableRenderingVertices(true);

	// add the surface pointer to pSurfs
	pSurfs.push_back(psurf);

	return true;
}

bool RigidBody::getReady()
{
	if ( !GBody::getReady() ) {
		cerr << "error:: failed in getting body ready" << endl; 
		return false;
	}

	// get surfaces ready
	for (size_t i=0; i<pSurfs.size(); i++) {
		if ( !pSurfs[i]->getReady() ) {
			cerr << "error:: failed in getting surface ready" << endl;
			return false;
		}
	}

	return true;
}

void RigidBody::update_T()
{
	GBody::update_T();

	// update bounding boxes of the surfaces for collision check
	for (int i=0; i<(int)pSurfs.size(); i++) {
		pSurfs[i]->updateBoundingBox();
	}
}

void RigidBody::render()
{
	if ( !bRendering ) return;
	
	for (int i=0; i<(int)pSurfs.size(); i++) {
		pSurfs[i]->render();
	}
}

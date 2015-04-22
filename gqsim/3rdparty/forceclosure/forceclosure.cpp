#include "forceclosure.h"
extern "C"
{
#include <qhull/qhull.h>
#include <qhull/mem.h>
#include <qhull/qset.h>
#include <qhull/geom.h>
#include <qhull/merge.h>
#include <qhull/poly.h>
#include <qhull/io.h>
#include <qhull/stat.h>
}
#include <stdio.h> // for tmpfile()

static FILE *errfile=NULL;

using namespace std;
using namespace OpenRAVE::geometry;

typedef RaveTransformMatrix<dReal> TransformMatrix;
#define PI ((dReal)3.14159265358979)


GRASPANALYSIS _AnalyzeContacts3D(const vector<CONTACT>& contacts, dReal mu, int Nconepoints)
{
    if( mu == 0 )
        return _AnalyzeContacts3D(contacts);

    dReal fdeltaang = 2*PI/(dReal)Nconepoints;
    dReal fang = 0;
    vector<pair<dReal,dReal> > vsincos(Nconepoints);
    for (vector< pair<dReal,dReal> >::iterator it = vsincos.begin(); it != vsincos.end(); it++) { //FOREACH(it,vsincos) {
        it->first = RaveSin(fang);
        it->second = RaveCos(fang);
        fang += fdeltaang;
    }
    
    vector<CONTACT> newcontacts;
    newcontacts.reserve(contacts.size()*Nconepoints);
    for(vector<CONTACT>::const_iterator itcontact = contacts.begin(); itcontact != contacts.end(); itcontact++) { //FOREACHC(itcontact,contacts) {
        // find a coordinate system where z is the normal
        TransformMatrix torient = matrixFromQuat(quatRotateDirection(Vector(0,0,1),itcontact->norm));
        Vector right(torient.m[0],torient.m[4],torient.m[8]);
        Vector up(torient.m[1],torient.m[5],torient.m[9]);
	    for (vector< pair<dReal,dReal> >::iterator it = vsincos.begin(); it != vsincos.end(); it++) //FOREACH(it,vsincos)
            newcontacts.push_back(CONTACT(itcontact->pos, (itcontact->norm + mu*it->first*right + mu*it->second*up).normalize3(),0));
    }

    return _AnalyzeContacts3D(newcontacts);
}

GRASPANALYSIS _AnalyzeContacts3D(const vector<CONTACT>& contacts)
{
    GRASPANALYSIS analysis;

	if( contacts.size() < 7 ) {
        //throw openrave_exception("need at least 7 contact wrenches to have force closure in 3D");
		//cerr << "need at least 7 contact wrenches to have force closure in 3D" << endl;
		return analysis;
	}

    vector<double> vpoints(6*contacts.size()), vconvexplanes;

    vector<double>::iterator itpoint = vpoints.begin();
    for(vector<CONTACT>::const_iterator itcontact = contacts.begin(); itcontact != contacts.end(); itcontact++) { //FOREACHC(itcontact,contacts) {
        *itpoint++ = itcontact->norm.x;
        *itpoint++ = itcontact->norm.y;
        *itpoint++ = itcontact->norm.z;
        Vector v = itcontact->pos.cross(itcontact->norm);
        *itpoint++ = v.x;
        *itpoint++ = v.y;
        *itpoint++ = v.z;
    }

    if ( !_ComputeConvexHull(analysis.volume, vpoints,vconvexplanes,6) ) {
		return analysis;
	}

    vector<double> vmean(6, 0.0); //boost::array<double,6> vmean={{0}};
    for(size_t i = 0; i < vpoints.size(); i += 6) {
        for(int j = 0; j < 6; ++j)
            vmean[j] += vpoints[i+j];
    }
    double fipoints = 1.0f/(double)contacts.size();
    for(int j = 0; j < 6; ++j)
        vmean[j] *= fipoints;
    
    // go through each of the faces and check if center is inside, and compute its distance
    double mindist = 1e30;
    for(size_t i = 0; i < vconvexplanes.size(); i += 7) {
        double dist = -vconvexplanes.at(i+6);
        double meandist = 0;
        for(int j = 0; j < 6; ++j)
            meandist += vconvexplanes[i+j]*vmean[j];
        
        if( dist < meandist )
            dist = -dist;
        if( dist < 0 || RaveFabs(dist-meandist) < 1e-15 )
            return analysis;
        mindist = min(mindist,dist);
    }
    analysis.mindist = mindist;
    return analysis;
}

/// Computes the convex hull of a set of points
/// \param vpoints a set of points each of dimension dim
/// \param vconvexplaces the places of the convex hull, dimension is dim+1
double _ComputeConvexHull(double &totvol_, const vector<double>& vpoints, vector<double>& vconvexplanes, int dim)
{
	totvol_ = 0;

    vconvexplanes.resize(0);
    vector<coordT> qpoints(vpoints.size());
    std::copy(vpoints.begin(),vpoints.end(),qpoints.begin());
    
    boolT ismalloc = 0;           // True if qhull should free points in qh_freeqhull() or reallocation
    char flags[]= "qhull Tv FA"; // option flags for qhull, see qh_opt.htm, output volume (FA)

    if( !errfile )
        errfile = tmpfile();    // stderr, error messages from qhull code  
    
    int exitcode= qh_new_qhull (dim, qpoints.size()/dim, &qpoints[0], ismalloc, flags, errfile, errfile);
    if (!exitcode) {
        vconvexplanes.reserve(1000);

        facetT *facet;	          // set by FORALLfacets 
        FORALLfacets { // 'qh facet_list' contains the convex hull
//                if( facet->isarea && facet->f.area < 1e-15 ) {
//                    RAVELOG_VERBOSE(str(boost::format("skipping area: %e\n")%facet->f.area));
//                    continue;
//                }
            for(int i = 0; i < dim; ++i)
                vconvexplanes.push_back(facet->normal[i]);
            vconvexplanes.push_back(facet->offset);
        }
    }
    
    double totvol = qh totvol;
    qh_freeqhull(!qh_ALL);
    int curlong, totlong;	  // memory remaining after qh_memfreeshort 
    qh_memfreeshort (&curlong, &totlong);
    if (curlong || totlong)
	{
        //RAVELOG_ERROR("qhull internal warning (main): did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);
		cerr << "qhull internal warning (main): did not free " << totlong << " bytes of long memory (" << curlong << " pieces)" << endl;
		return false;
	}
    if( exitcode ) {
        //throw openrave_exception(str(boost::format("Qhull failed with error %d")%exitcode));
		cerr << "Qhull failed with error " << exitcode << endl;
		return false;
	}

	totvol_ = totvol;
    return true;
}



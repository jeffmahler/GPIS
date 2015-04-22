//================================================================================
//         Force closure analysis (from OpenRAVE r1716)
//================================================================================

#ifndef _FORCE_CLOSURE_ANALYSIS_
#define _FORCE_CLOSURE_ANALYSIS_

#include <vector>
#include "geometry.h"

typedef double dReal;
typedef OpenRAVE::geometry::RaveVector<dReal> Vector;

class CONTACT
{
public:
    CONTACT() : depth(0) {}
    CONTACT(const Vector& p, const Vector& n, dReal d) : pos(p), norm(n) {depth = d;}

    Vector pos;     ///< where the contact occured
    Vector norm;    ///< the normals of the faces
    dReal depth;    ///< the penetration depth, positive means the surfaces are penetrating, negative means the surfaces are not colliding (used for distance queries)
};

struct GRASPANALYSIS
{
GRASPANALYSIS() : mindist(0), volume(0) {}
    dReal mindist;
    dReal volume;
};


GRASPANALYSIS _AnalyzeContacts3D(const std::vector<CONTACT>& contacts, dReal mu, int Nconepoints);
GRASPANALYSIS _AnalyzeContacts3D(const std::vector<CONTACT>& contacts);
double _ComputeConvexHull(double &totvol_, const std::vector<double>& vpoints, std::vector<double>& vconvexplanes, int dim);

#endif


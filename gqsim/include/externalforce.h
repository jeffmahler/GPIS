#ifndef _EXTERNAL_FORCE_
#define _EXTERNAL_FORCE_

#include "liegroup.h"

enum ForceApplyType { FA_CONSTANT, FA_LINEAR };
enum ForceResponseMeasureType { FRM_MAX_DEVIATION, FRM_RESISTANT_TIME, FRM_ALL };

class ExternalForce
{
public:
	ForceResponseMeasureType _gqm_type;
	ForceApplyType _fa_type;
	Vec3 _dir;				// direction in {global}
	double _mag;			// magnitude
	double _time;			// duration time
public:
	ExternalForce() {}
	ExternalForce(ForceResponseMeasureType gqmtype, ForceApplyType ftype, Vec3 d, double m, double t) : _gqm_type(gqmtype), _fa_type(ftype), _dir(d), _mag(m), _time(t) {}
	~ExternalForce() {}
	bool isTimeOut(double t) { if ( t > _time - 1E-10 ) { return true; } return false; }
	Vec3 getForce(double t) { 
		if ( !isTimeOut(t) ) {
			switch ( _fa_type ) {
			case FA_CONSTANT:
				return _mag * _dir;
			case FA_LINEAR:
				return (_mag * t / _time) * _dir;
			}
		}
		return Vec3(0,0,0);
	}
};

class ExternalMoment : public ExternalForce
{
public:
	ExternalMoment(ForceResponseMeasureType gqmtype, ForceApplyType ftype, Vec3 d, double m, double t) : ExternalForce(gqmtype, ftype, d, m, t) {}
	Vec3 getMomemt(double t) { return getForce(t); }
};



#endif


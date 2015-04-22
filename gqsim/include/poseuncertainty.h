#ifndef _POSE_UNCERTAINTY_
#define _POSE_UNCERTAINTY_

#include <vector>
#include <fstream>
#include "liegroup.h"

class PoseUncertainty
{
public:
	int _N; // sampling size from a normal distribution model for the object pose uncertainty
	Vec3 _sigma_pos;		// standard deviation of the normal distribution for position (normal distribution of p)
	double _sigma_angle;	// standard deviation of the normal distribution for rotation angle
	Vec3 _axis;				// axis along which rotation can occur (e.g., [0,0,1] for rotation along z, [1,1,1] for rotation along random axis)
	SO3 _R_principal_axes_pos;	// principal axes of the positional distribution (position = R*p)
	SO3 _R_principal_axes_ori;	// principal axes of the orientational distribution (orientation = exp([R*w]) where w = rotation axis and angle)
		
	std::vector<SE3> _poses; // pose samples from the uncertainty model

public:
	PoseUncertainty() : _N(0), _sigma_pos(Vec3(0,0,0)), _sigma_angle(0), _axis(Vec3(1,1,1)), _R_principal_axes_pos(SO3()), _R_principal_axes_ori(SO3()) {}
	~PoseUncertainty() {}

	int size() { return _N; }

	void set(int n, Vec3 sigma_pos, double sigma_angle, Vec3 axis, SO3 R_principal_axes_pos=SO3(), SO3 R_principal_axes_ori=SO3())
	{
		_N = n;
		_sigma_pos = sigma_pos;
		_sigma_angle = sigma_angle;
		_axis = axis;
		_R_principal_axes_pos = R_principal_axes_pos;
		_R_principal_axes_ori = R_principal_axes_ori;
		sample();
	}

	void sample()
	{
		double sp0=_sigma_pos[0], sp1=_sigma_pos[1], sp2=_sigma_pos[2], sang=_sigma_angle, sa0=_axis[0], sa1=_axis[1], sa2 = _axis[2];
		SO3 Rp = _R_principal_axes_pos, Ro = _R_principal_axes_ori;
		_poses.resize(_N);
		_poses[0] = SE3();
		for (int i=1; i<_N; i++) {
			Vec3 p(randn(0,sp0),randn(0,sp1),randn(0,sp2));
			Vec3 w(randn(0,sa0),randn(0,sa1),randn(0,sa2)); w.Normalize(); // w = rotation axis, random point on a sphere (http://mathworld.wolfram.com/SpherePointPicking.html)
			w *= randn(0,sang); // rotation along the axis
			_poses[i] = SE3(Exp(Ro*w), Rp*p);
		}
	}

	void save(const char *filepath)
	{
		std::ofstream fout(filepath);
		fout << "size = " << _N << std::endl;
		fout << "sigma_pos = " << _sigma_pos;
		fout << "sigma_angle = " << _sigma_angle << std::endl;
		fout << "axis = " << _axis << std::endl;
		fout << "R_principal_axes_pos = " << _R_principal_axes_pos << "R_principal_axes_ori = " << _R_principal_axes_ori << std::endl;
		fout << "p = [" << std::endl;
		for (int i=0; i<_N; i++) {
			Vec3 p = _poses[i].GetPosition();
			fout << p[0] << " " << p[1] << " " << p[2] << std::endl;
		}
		fout << "];" << std::endl;
		fout << "w = [" << std::endl;
		for (int i=0; i<_N; i++) {
			Vec3 w = Log(_poses[i].GetRotation());
			fout << w[0] << " " << w[1] << " " << w[2] << std::endl;
		}
		fout << "];" << std::endl;
		fout << "R_and_p = [" << std::endl;
		for (int i=0; i<_N; i++) {
			SO3 R = _poses[i].GetRotation();
			Vec3 p = _poses[i].GetPosition();
			fout << R[0] << " " << R[1] << " " << R[2] << " " << R[3] << " " << R[4] << " " << R[5] << " " << R[6] << " " << R[7] << " " << R[8] << " ";
			fout << p[0] << " " << p[1] << " " << p[2] << std::endl;
		}
		fout << "];" << std::endl;
		fout.close();
	}
};

#endif


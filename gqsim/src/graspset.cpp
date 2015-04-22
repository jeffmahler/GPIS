#include <algorithm>
#include <fstream>
#include <assert.h>
#include "graspset.h"

bool bettergrasp(const GraspSet::Grasp &a, const GraspSet::Grasp &b )
{
	return a._score > b._score;
}

void GraspSet::sort()
{
	std::sort(_grasps.begin(), _grasps.end(), bettergrasp);
}

void GraspSet::random_shuffle()
{
	std::random_shuffle(_grasps.begin(), _grasps.end());
}

bool GraspSet::load(const char *filepath)
{
	std::ifstream fin(filepath);
	if ( !fin.is_open() ) return false;
	size_t ng, nj;
	SO3 R;
	Vec3 p;
	std::vector<double> preshape;
	double score;
	fin >> ng >> nj;
	_grasps.resize(ng);
	preshape.resize(nj);
	for (size_t i=0; i<ng; i++) {
		fin >> R[0] >> R[1] >> R[2] >> R[3] >> R[4] >> R[5] >> R[6] >> R[7] >> R[8] >> p[0] >> p[1] >> p[2];
		for (size_t j=0; j<nj; j++) { fin >> preshape[j]; }
		fin >> score;
		_grasps[i]._preshape = preshape;
		_grasps[i]._T = SE3(R, p);
		_grasps[i]._score = score;
	}
	fin.close();
	return true;
}

bool GraspSet::save(const char *filepath)
{
	std::ofstream fout(filepath);
	if ( !fout.is_open() ) return false;
	size_t ng = _grasps.size(), nj = 0;
	SO3 R; Vec3 p;
	if ( ng > 0 ) { nj = _grasps[0]._preshape.size(); }
	fout << ng << std::endl;
	fout << nj << std::endl;
	for (size_t i=0; i<_grasps.size(); i++) {
		R = _grasps[i]._T.GetRotation();
		p = _grasps[i]._T.GetPosition();
		for (size_t j=0; j<9; j++) {
			fout << R[j] << " ";
		}
		for (size_t j=0; j<3; j++) {
			fout << p[j] << " ";
		}
		assert(_grasps[i]._preshape.size() == nj);
		for (size_t j=0; j<nj; j++) { fout << _grasps[i]._preshape[j] << " "; }
		fout << _grasps[i]._score << std::endl;
	}
	fout.close();
	return true;
}

std::vector<int> GraspSet::pick(int n)
{
	std::vector<int> picked;

	if ( n <= 0 ) return picked;
	if ( n > (int)_grasps.size() ) { n = (int)_grasps.size(); }

	picked.push_back(0);

	double d, d_max;
	int idx_d_max;
	while ( (int)picked.size() < n ) {
		d_max = -1E10;
		idx_d_max = -1;
		for (size_t i=1; i<_grasps.size(); i++) {
			if ( std::find(picked.begin(), picked.end(), i) != picked.end() )
				continue;
			d = 0;
			for (size_t j=0; j<picked.size(); j++) {
				d *= distance(i,j);
			}
			if ( d > d_max ) {
				d_max = d;
				idx_d_max = i;
			}
		}
		if ( idx_d_max < 0 ) {
			std::cerr << "an error occurred in picking n grasps!" << std::endl;
			break;
		}
		picked.push_back(idx_d_max);
	}

	return picked;
}

double GraspSet::distance(int i, int j)
{
	double alpha = 1./(30./180.*3.14159); // weight for rotation
	double beta = 1./0.05; // weight for translation
	SE3 T = Inv(_grasps[i]._T)*_grasps[j]._T;
	double angdiff = 0;
	for (size_t k=0; k<_grasps[i]._preshape.size(); k++) { 
		angdiff += (_grasps[i]._preshape[k] - _grasps[j]._preshape[k]) * (_grasps[i]._preshape[k] - _grasps[j]._preshape[k]); 
	}
	angdiff = sqrt(angdiff);
	return alpha * Norm(Log(T.GetRotation())) + beta * Norm(T.GetPosition()) + alpha * angdiff;
}

std::vector<int> GraspSet::find_duplicates(double eps)
{
	std::vector<int> dup;
	for (size_t i=0; i<_grasps.size(); i++) {
		for (size_t j=0; j<i; j++) {
			if ( distance(i,j) < eps ) {
				dup.push_back(j);
			}
		}
	}
	return dup;
}

void GraspSet::remove(std::vector<int> indices)
{
	std::vector<Grasp> grasps_new;
	for (size_t i=0; i<_grasps.size(); i++) {
		if ( std::find(indices.begin(), indices.end(), i) == indices.end() ) {
			grasps_new.push_back(_grasps[i]);
		}
	}
	_grasps.clear();
	_grasps = grasps_new;
}

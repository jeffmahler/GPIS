#ifndef _GRASP_SET_
#define _GRASP_SET_

#include <vector>
#include "liegroup.h"

class GraspSet
{
public:
	struct Grasp
	{
		std::vector<double> _preshape;	// preshape (finger joint angles)
		SE3 _T;							// {object} --> {end-effector} of the hand
		double _score;					// grasp quality score (the higher score the better grasp)
	};

	std::vector<Grasp> _grasps;			// grasps

public:
	GraspSet() {}
	~GraspSet() {}

	size_t size() const { return _grasps.size(); }

	void clear() { _grasps.clear(); }

	bool load(const char *filepath);	// load grasp set from file
	bool save(const char *filepath);	// save grasp set to file

										// --- file format ---
										// ng     # number of grasps 
										// nj     # number of finger joints (for preshape)
										// x_0    # x_i = i-th grasp = [rotation matrix, position, preshape, score] (size(x) = 4+3+nj+1)
										// ...
										// x_(ng-1)
										// --------------------
										//  *) rotation matrix = [r00 r10 r20 r01 r11 r21 r02 r12 r22]
										//  *) size(preshape) = nj

	void sort();						// sort grasps in terms of the grasp quality score in descending order

	void random_shuffle();				// rearrange the grasps randomly

	std::vector<int> pick(int n);		// pick n grasps which are farthest from each other and return their indices

	double distance(int i, int j);		// compute the distance between the two grasps

	std::vector<int> find_duplicates(double eps = 1E-6); // find duplicate grasps and return their indices
	void remove(std::vector<int> indices); // remove grasps
};

#endif


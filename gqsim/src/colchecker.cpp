#include <algorithm>
#include "colchecker.h"
#include "cdfunc.h"
#include "rigidsurface.h"

void CollisionChecker::addCollisionSurfacePair(RigidSurface *psa, RigidSurface *psb, CollisionChecker::ContactParam cp, bool bidir, bool breuseprevinfo, bool baveragednormal)
{
	CollidableSurfacePair colsurfpair(psa, psb, cp, bidir, breuseprevinfo, baveragednormal);
	_colsurfpairs.push_back(colsurfpair); 
	if ( find(_psurfs.begin(), _psurfs.end(), psa) == _psurfs.end() ) {
		_psurfs.push_back(psa);
	}
	if ( find(_psurfs.begin(), _psurfs.end(), psb) == _psurfs.end() ) {
		_psurfs.push_back(psb);
	}
}

void CollisionChecker::checkCollision()
{
	// save the previous contact info and reset the contact info
	for (size_t i=0; i<_psurfs.size(); i++) {
		_psurfs[i]->savePrevContactInfo();
		_psurfs[i]->resetContactInfo();
	}
	// check collision
	for (size_t i=0; i<_colsurfpairs.size(); i++) {
		_colsurfpairs[i].checkCollision();
	}
}

void CollisionChecker::applyContactForces()
{
	for (size_t i=0; i<_colsurfpairs.size(); i++) {
		_colsurfpairs[i].applyContactForces();
	}
}

CollisionChecker::CollidableSurfacePair::CollidableSurfacePair(RigidSurface *psa, RigidSurface *psb, ContactParam cp, bool bbidir, bool breuseprevinfo, bool baveragednormal) : _psurfA(psa), _psurfB(psb), _cp(cp), _bBidirectional(bbidir), _bReusePrevInfo(breuseprevinfo), _bAveragedNormal(baveragednormal)
{ 
	if ( !psa || !psb ) return;

	// reserve memory for optimizing push_back()
	_idxActiveVertexA.reserve(psa->getNumVertices());
	_idxActiveVertexB.reserve(psb->getNumVertices());
	if ( bbidir ) { 
		_cols.reserve(psa->getNumVertices()+psb->getNumVertices()); 
	} else { 
		_cols.reserve(psa->getNumVertices()); 
	}
}

void CollisionChecker::CollidableSurfacePair::checkCollision()
{
	_cols.clear();

	if ( !_psurfA || !_psurfB ) return;
	if ( _psurfA->getNumVertices() == 0 || _psurfB->getNumVertices() == 0 ) return;

	BoundingBox &obbA = _psurfA->getBoundingBox(); // bounding box of surface A w.r.t. {global}
	BoundingBox &obbB = _psurfB->getBoundingBox(); // bounding box of surface B w.r.t. {global}

	SE3 Ta, Tb, invTa, invTb;
	if ( !!_psurfA->_pT ) {
		Ta = *(_psurfA->_pT);
		invTa = Inv(Ta);
	}
	if ( !!_psurfB->_pT ) {
		Tb = *(_psurfB->_pT);
		invTb = Inv(Tb);
	}

	// If the relative pose between the two surfaces is almost identical to the previous one, 
	// run the fast version using the previous collision information (_idxActiveVertexA, _idxActiveVertexB, _idxNearestVertexA, _idxNearestVertexB).
	// (It skips the large and medium scale checks and some part of the small scale check.)
	if ( _bReusePrevInfo ) {
		SE3 Tab = invTa * Tb;
		SE3 dT = _Tab_prev*Inv(Tab);
		gReal L = (obbB.extents[0] > obbB.extents[1] ? (obbB.extents[0] > obbB.extents[2] ? obbB.extents[0] : obbB.extents[2] ) : (obbB.extents[1] > obbB.extents[2] ? obbB.extents[1] : obbB.extents[2]));
		gReal maxposchange = fabs(L) * Norm(Log(dT.GetRotation())) + Norm(dT.GetPosition()); // maximum possible relative position change
		gReal eps = 1E-5; // eps = 0.01mm
		// if maxposchange is less than eps, do collision check with information at the previous time step
		if ( maxposchange < eps ) { 
			_checkCollisionFast(); 
			return;
		}
		_Tab_prev = Tab; // save the current relative pose (this must be updated only when the full collision check is called)
	}

	// initialize
	_idxActiveVertexA.clear();
	_idxActiveVertexB.clear();
	_idxNearestVertexA.clear();
	_idxNearestVertexB.clear();

	// ---- large scale check (between the bounding boxes of the surfaces) ----

	if ( !dcOBBOverlap(obbA.extents, obbA.position.GetArray(), obbA.orientation.GetArray(), obbB.extents, obbB.position.GetArray(), obbB.orientation.GetArray()) ) {
		return;
	}

	// ---- medium scale check (vertices <--> bounding box) to find active vertices and seed vertices

	BoundingBox obbB_A(obbB.extents, invTa * obbB.position, invTa.GetRotation() * obbB.orientation); // bounding box of surface B w.r.t. surface A
	BoundingBox obbA_B(obbA.extents, invTb * obbA.position, invTb.GetRotation() * obbA.orientation); // bounding box of surface A w.r.t. surface B

	for (size_t i=0; i<_psurfA->getNumVertices(); i++) {
		if ( dcPointOBBOverlap(_psurfA->vertices[i].GetArray(), obbB_A.extents, obbB_A.position.GetArray(), obbB_A.orientation.GetArray()) ) {
			_idxActiveVertexA.push_back(i);
		}
	}

	if ( _bBidirectional ) {
		for (size_t i=0; i<_psurfB->getNumVertices(); i++) {
			if ( dcPointOBBOverlap(_psurfB->vertices[i].GetArray(), obbA_B.extents, obbA_B.position.GetArray(), obbA_B.orientation.GetArray()) ) {
				_idxActiveVertexB.push_back(i);
			}
		}
	}

	// if there is no active vertex in both surfaces, there is no collision
	if ( _idxActiveVertexA.size() == 0 && _idxActiveVertexB.size() == 0 ) return;

	// set seed vertices
	std::vector<int> idxseedvertexA;// = _idxActiveVertexA;
	std::vector<int> idxseedvertexB;// = _idxActiveVertexB;

	if ( idxseedvertexB.size() == 0 ) {
		// p = mid point of the active vertices of surface A w.r.t. {surface A}
		// n = averaged normal of the active vertices of surface A w.r.t. {surface A}
		Vec3 p(0,0,0), n(0,0,0);
		for (size_t i=0; i<_idxActiveVertexA.size(); i++) {
			p += _psurfA->vertices[_idxActiveVertexA[i]];
			if ( _psurfA->vertexnormals.size() == _psurfA->vertices.size() ) {
				n += _psurfA->vertexnormals[_idxActiveVertexA[i]];
			} else {
				n += _psurfA->vertices[_idxActiveVertexA[i]] - obbA.position;
			}
		}
		p *= (1./(gReal)_idxActiveVertexA.size());
		n.Normalize();
		// pb = the mid point w.r.t. {surface B}
		// nb = the averaged normal w.r.t. {surface B}
		Vec3 pb = invTb * Ta * p;
		Vec3 nb = invTb * Ta * n;
		// find seed vertices
		for (size_t i=0; i<_idxActiveVertexB.size(); i++) {
			if ( Inner(nb, _psurfB->vertexnormals[_idxActiveVertexB[i]]) < 0 ) {
				idxseedvertexB.push_back(_idxActiveVertexB[i]);
			}
		}
		if ( idxseedvertexB.size() == 0 ) {
			// finds the nearest vertex of surface A from the mid point
			int idx = -1;
			gReal dist = 1E20, dist_tmp;
			for (size_t i=0; i<_psurfB->getNumVertices(); i++) {
				dist_tmp = SquareSum(pb-_psurfB->vertices[i]);
				if ( dist_tmp < dist && Inner(nb, _psurfB->vertexnormals[i]) < 0 ) {
					idx = i;
					dist = dist_tmp;
				}
			}
			// add the nearest vertex into seed
			if ( idx >= 0 ) {
				idxseedvertexB.push_back(idx);
			}
		}
	}
	if ( _bBidirectional && idxseedvertexA.size() == 0 ) {
		// p = mid point of the active vertices of surface B w.r.t. {surface B}
		// n = mid point of the active vertices of surface B w.r.t. {surface B}
		Vec3 p(0,0,0), n(0,0,0);
		for (size_t i=0; i<_idxActiveVertexB.size(); i++) {
			p += _psurfB->vertices[_idxActiveVertexB[i]];
			if ( _psurfB->vertexnormals.size() == _psurfB->vertices.size() ) {
				n += _psurfB->vertexnormals[_idxActiveVertexB[i]];
			} else {
				n += _psurfB->vertices[_idxActiveVertexB[i]] - obbB.position;
			}
		}
		p *= (1./(gReal)_idxActiveVertexB.size());
		n.Normalize();
		// pa = the mid point w.r.t. {surface A}
		// na = the averaged normal w.r.t. {surface A}
		Vec3 pa = invTa * Tb * p;
		Vec3 na = invTa * Tb * n;
		// find seed vertices
		for (size_t i=0; i<_idxActiveVertexA.size(); i++) {
			if ( Inner(na, _psurfA->vertexnormals[_idxActiveVertexA[i]]) < 0 ) {
				idxseedvertexA.push_back(_idxActiveVertexA[i]);
			}
		}
		if ( idxseedvertexA.size() == 0 ) {
			// finds the nearest vertex of surface A from the mid point
			int idx = -1;
			gReal dist = 1E20, dist_tmp;
			for (size_t i=0; i<_psurfA->getNumVertices(); i++) {
				dist_tmp = SquareSum(pa-_psurfA->vertices[i]);
				if ( dist_tmp < dist && Inner(na, _psurfA->vertexnormals[i]) < 0 ) {
					idx = i;
					dist = dist_tmp;
				}
			}
			// add the nearest vertex into seed
			if ( idx >= 0 ) {
				idxseedvertexA.push_back(idx);
			}
		}
	}

	// ---- small scale check (active vertices of surface A/B <--> adjacent faces of the active vertices of surface B/A) ----

	// check collision between the surface A's active vertices and surface B
	_idxNearestVertexB.resize(_idxActiveVertexA.size());
	for (size_t i=0; i<_idxActiveVertexA.size(); i++) {

		// position of the i-th active vertex of surface A 
		Vec3 P = Ta * _psurfA->vertices[_idxActiveVertexA[i]]; // position vector w.r.t. {global}
		Vec3 Pb = invTb * P; // position vector w.r.t. {surface B}
		Vec3 vNb = invTb * Ta * _psurfA->vertexnormals[_idxActiveVertexA[i]]; // vertex normal w.r.t. {surface B}

		// find the nearest vertex of surface B
		gReal dtmp, dmin = 1E20;
		int idxB = -1;
		for (size_t j=0; j<idxseedvertexB.size(); j++) {
			dtmp = Norm(_psurfB->vertices[idxseedvertexB[j]] - Pb);
			if ( dtmp < dmin ) { dmin = dtmp; idxB = idxseedvertexB[j]; }
			//if ( dtmp < dmin && Inner(vNb, _psurfB->vertexnormals[idxseedvertexB[j]]) < 0 ) { dmin = dtmp; idxB = idxseedvertexB[j]; }
			//if ( dtmp < dmin && acos(Inner(vNb, _psurfB->vertexnormals[idxseedvertexB[j]])) > 90./180.*3.14159 ) { dmin = dtmp; idxB = idxseedvertexB[j]; }
		}
		//if ( idxB == -1 ) {
		//	std::cout << "warning:: no seed vertex" << std::endl;
		//	//continue;
		//	//for (size_t j=0; j<_psurfB->getNumVertices(); j++) {
		//	//	dtmp = Norm(_psurfB->vertices[j] - Pb);
		//	//	//if ( dtmp < dmin ) { dmin = dtmp; idxB = idxseedvertexB[j]; }
		//	//	if ( dtmp < dmin && acos(Inner(vNb, _psurfB->vertexnormals[j])) > 90./180.*3.14159 ) { dmin = dtmp; idxB = j; }
		//	//}
		//}
		////if ( Inner(Pb-_psurfB->vertices[idxB], _psurfB->vertexnormals[idxB]) <= 0 ) {
		//	std::cout << "warning 1111" << std::endl;
		//	std::cout << "idxB = " << idxB << std::endl;
		//	std::cout << "Pb = " << Pb << "pobbA = " << obbA_B.position << "v = " << _psurfB->vertices[idxB] << "vn = " << _psurfB->vertexnormals[idxB] << std::endl;
		//	std::cout << "idxseedvertexB = "; for (size_t j=0; j<idxseedvertexB.size(); j++) { std::cout << idxseedvertexB[j] << ", "; } std::cout << std::endl;
		//}
		int idxB_seed = idxB;
		if ( _psurfA->vertexnormals.size() == _psurfA->vertices.size() ) {
			idxB = _psurfB->searchNearestVertex(Pb, idxB, vNb);
		} else {
			idxB = _psurfB->searchNearestVertex(Pb, idxB);
		}
		_idxNearestVertexB[i] = idxB;
		//if ( idxB != idxB_old && _psurfB->getNumVertices() > 1000 && _psurfA->getNumVertices() > 100 ) {
		//	std::cout << "changed!" << std::endl;
		//}
		//if ( Inner(Pb-_psurfB->vertices[idxB], _psurfB->vertexnormals[idxB]) <= 0 ) {
		//	std::cout << "warning 2222" << std::endl;
		//	std::cout << "idxB = " << idxB << std::endl;
		//}

		// check collision with the surface patch of the nearest vertex
		double pd;
		Vec3 Nb(0,0,0);
		int idxfaceB;
		bool bcol;
		if ( _bAveragedNormal ) {
			bcol = _psurfB->_checkCollisionWithSurfacePatch(pd, Nb, idxfaceB, idxB, Pb, vNb);
		} else {
			bcol = _psurfB->_checkCollisionWithSurfacePatch(pd, Nb, idxfaceB, idxB, Pb);
		}

		if ( bcol ) {
			
			// save the collision info (position and normal must be represented in {global})
			_cols.push_back(CollisionInfo(_psurfA, _psurfB, _idxActiveVertexA[i], idxB, P, Tb.GetRotation()*Nb, pd, idxfaceB));

			// if this is a new collision, save the contact point (w.r.t. {surface B}) as the reference position for computing static friction force in World::_handleCollision()
			if ( !_psurfA->_bContactPrev[_idxActiveVertexA[i]] ) {
				_psurfA->_xf_ref[_idxActiveVertexA[i]] = Pb;
			}

			_psurfA->_bContact[_idxActiveVertexA[i]] = 1;
			_psurfA->_collidingVertexIndices.push_back(_idxActiveVertexA[i]);

			if ( idxB_seed >= 0 ) {
				_psurfB->_bColSeedVertex[idxB_seed] = 1;
				_psurfB->_seedVertexIndices.push_back(idxB_seed);
			}
		}
	}

	// check collision between the surface B's active vertices and surface A
	if ( _bBidirectional ) {
		_idxNearestVertexA.resize(_idxActiveVertexB.size());
		for (size_t i=0; i<_idxActiveVertexB.size(); i++) {

			// position of the i-th active vertex of surface B 
			Vec3 P = Tb * _psurfB->vertices[_idxActiveVertexB[i]]; // position vector w.r.t. {global}
			Vec3 Pa = invTa * P; // position vector w.r.t. {surface A}
			Vec3 vNa = invTa * Tb * _psurfB->vertexnormals[_idxActiveVertexB[i]]; // vertex normal vector w.r.t. {surface A}

			// find the nearest vertex of surface A
			gReal dtmp, dmin = 1E20;
			int idxA = -1;
			for (size_t j=0; j<idxseedvertexA.size(); j++) {
				dtmp = Norm(_psurfA->vertices[idxseedvertexA[j]] - Pa);
				if ( dtmp < dmin ) { dmin = dtmp; idxA = idxseedvertexA[j]; }
			}
			int idxA_seed = idxA;
			if ( _psurfB->vertexnormals.size() == _psurfB->vertices.size() ) {
				idxA = _psurfA->searchNearestVertex(Pa, idxA, invTa * Tb * _psurfB->vertexnormals[_idxActiveVertexB[i]]);
			} else {
				idxA = _psurfA->searchNearestVertex(Pa, idxA);
			}
			_idxNearestVertexA[i] = idxA;

			// check collision with the surface patch of the nearest vertex
			double pd;
			Vec3 Na(0,0,0);
			int idxfaceA;
			bool bcol;
			if ( _bAveragedNormal ) {
				bcol = _psurfB->_checkCollisionWithSurfacePatch(pd, Na, idxfaceA, idxA, Pa, vNa);
			} else {
				bcol = _psurfA->_checkCollisionWithSurfacePatch(pd, Na, idxfaceA, idxA, Pa);
			}

			if ( bcol ) {
			
				// save the collision info (position and normal must be represented in {global})
				_cols.push_back(CollisionInfo(_psurfB, _psurfA, _idxActiveVertexB[i], idxA, P, Ta.GetRotation()*Na, pd, idxfaceA));

				// if this is a new collision, save the contact point (w.r.t. {surface A}) as the reference position for computing static friction force in World::_handleCollision()
				if ( !_psurfB->_bContactPrev[_idxActiveVertexB[i]] ) {
					_psurfB->_xf_ref[_idxActiveVertexB[i]] = Pa;
				}

				_psurfB->_bContact[_idxActiveVertexB[i]] = 1;
				_psurfB->_collidingVertexIndices.push_back(_idxActiveVertexB[i]);

				if ( idxA_seed >= 0 ) {
					_psurfA->_bColSeedVertex[idxA_seed] = 1;
					_psurfA->_seedVertexIndices.push_back(idxA_seed);
				}
			}
		}
	}
}

void CollisionChecker::CollidableSurfacePair::_checkCollisionFast()
{
	// do small scale collision check with _idxActiveVertexA, _idxActiveVertexB, _idxNearestVertexA, _idxNearestVertexB

	_cols.clear();

	SE3 Ta, Tb, invTa, invTb;
	if ( !!_psurfA->_pT ) {
		Ta = *(_psurfA->_pT);
		invTa = Inv(Ta);
	}
	if ( !!_psurfB->_pT ) {
		Tb = *(_psurfB->_pT);
		invTb = Inv(Tb);
	}

	// ---- small scale check (active vertices of surface A/B <--> adjacent faces of the active vertices of surface B/A) ----

	// check collision between the surface A's active vertices and surface B
	for (size_t i=0; i<_idxActiveVertexA.size(); i++) {

		// position of the i-th active vertex of surface A 
		Vec3 P = Ta * _psurfA->vertices[_idxActiveVertexA[i]]; // position vector w.r.t. {global}
		Vec3 Pb = invTb * P; // position vector w.r.t. {surface B}
		Vec3 vNb = invTb * Ta * _psurfA->vertexnormals[_idxActiveVertexA[i]]; // vertex normal w.r.t. {surface B}

		// the nearest vertex of surface B
		int idxB = _idxNearestVertexB[i];

		// check collision with the surface patch of the nearest vertex
		double pd;
		Vec3 Nb(0,0,0);
		int idxfaceB;
		bool bcol;
		if ( _bAveragedNormal ) {
			bcol = _psurfB->_checkCollisionWithSurfacePatch(pd, Nb, idxfaceB, idxB, Pb, vNb);
		} else {
			bcol = _psurfB->_checkCollisionWithSurfacePatch(pd, Nb, idxfaceB, idxB, Pb);
		}

		if ( bcol ) {
			
			// save the collision info (position and normal must be represented in {global})
			_cols.push_back(CollisionInfo(_psurfA, _psurfB, _idxActiveVertexA[i], idxB, P, Tb.GetRotation()*Nb, pd, idxfaceB));

			// if this is a new collision, save the contact point (w.r.t. {surface B}) as the reference position for computing static friction force in World::_handleCollision()
			if ( !_psurfA->_bContactPrev[_idxActiveVertexA[i]] ) {
				_psurfA->_xf_ref[_idxActiveVertexA[i]] = Pb;
			}

			_psurfA->_bContact[_idxActiveVertexA[i]] = 1;
			_psurfA->_collidingVertexIndices.push_back(_idxActiveVertexA[i]);

			_psurfB->_bColSeedVertex[idxB] = 1;
			_psurfB->_seedVertexIndices.push_back(idxB);
		}
	}

	// check collision between the surface B's active vertices and surface A
	if ( _bBidirectional ) {
		for (size_t i=0; i<_idxActiveVertexB.size(); i++) {

			// position of the i-th active vertex of surface B 
			Vec3 P = Tb * _psurfB->vertices[_idxActiveVertexB[i]]; // position vector w.r.t. {global}
			Vec3 Pa = invTa * P; // position vector w.r.t. {surface A}
			Vec3 vNa = invTa * Tb * _psurfB->vertexnormals[_idxActiveVertexB[i]]; // vertex normal vector w.r.t. {surface A}

			// the nearest vertex of surface A
			int idxA = _idxNearestVertexA[i];

			// check collision with the surface patch of the nearest vertex
			double pd;
			Vec3 Na(0,0,0);
			int idxfaceA;
			bool bcol;
			if ( _bAveragedNormal ) {
				bcol = _psurfA->_checkCollisionWithSurfacePatch(pd, Na, idxfaceA, idxA, Pa, vNa);
			} else {
				bcol = _psurfA->_checkCollisionWithSurfacePatch(pd, Na, idxfaceA, idxA, Pa);
			}

			if ( bcol ) {
			
				// save the collision info (position and normal must be represented in {global})
				_cols.push_back(CollisionInfo(_psurfB, _psurfA, _idxActiveVertexB[i], idxA, P, Ta.GetRotation()*Na, pd, idxfaceA));

				// if this is a new collision, save the contact point (w.r.t. {surface A}) as the reference position for computing static friction force in World::_handleCollision()
				if ( !_psurfB->_bContactPrev[_idxActiveVertexB[i]] ) {
					_psurfB->_xf_ref[_idxActiveVertexB[i]] = Pa;
				}

				_psurfB->_bContact[_idxActiveVertexB[i]] = 1;
				_psurfB->_collidingVertexIndices.push_back(_idxActiveVertexB[i]);

				_psurfA->_bColSeedVertex[idxA] = 1;
				_psurfA->_seedVertexIndices.push_back(idxA);
			}
		}
	}
}

void CollisionChecker::CollidableSurfacePair::applyContactForces()
{
	double Kp = _cp._Kp, Kd = _cp._Kd, Kfp = _cp._Kfp, Kfd = _cp._Kfd, mu_s = _cp._mu_s, mu_d = _cp._mu_d, k = _cp._k;

	for (size_t i=0; i<_cols.size(); i++) {
		RigidSurface *psa = _cols[i]._psurfA;
		RigidSurface *psb = _cols[i]._psurfB;
		int nc =_cols.size(); // number of contacts
		gReal pd = _cols[i]._penetration_depth; // penetration depth
		Vec3 P = _cols[i]._pos; // contact point
		Vec3 P0 = *(psb->_pT) * psa->_xf_ref[_cols[i]._idxVertexA]; // reference contact point in global frame
		Vec3 N = _cols[i]._normal;

		// compute velocity
		Vec3 Va(0,0,0), Vb(0,0,0);
		SO3 R;
		if ( psa->_pV != NULL ) {
			R = psa->_pT->GetRotation();
			Va = Cross(R * psa->_pV->GetW(), P - psa->_pT->GetPosition()) + R * psa->_pV->GetV();
		} else {
			std::cout << "warning:: velocity info is not accessible!" << std::endl;
		}
		if ( psb->_pV != NULL ) {
			R = psb->_pT->GetRotation();
			Vb = Cross(R * psb->_pV->GetW(), P - psb->_pT->GetPosition()) + R * psb->_pV->GetV();
		} else {
			std::cout << "warning:: velocity info is not accessible!--" << std::endl;
		}
		Vec3 Vab = Va - Vb; // relative velocity
		//Vec3 Vn = Inner(Vab, N) * N; // normal component of the relative velocity
		//Vec3 Vt = Vab - Vn; // tangential component of the relative velocity
		gReal vn = Inner(Vab, N); // normal component of the relative velocity
		Vec3 Vt = Vab - vn*N; // tangential component of the relative velocity
		gReal vs = Norm(Vt); // slip velocity

		// compute normal force (acting on surface A)
		gReal fn = 0;
		if ( vn < 0 ) {
			fn = pd * (Kp - Kd * vn);
		} else {
			fn = pd * Kp;
		}
		fn /= nc;
		
		//gReal _Tp = 0.0005;
		//if ( pd < _Tp ) {
		//	fn = ( 0.5 * Kp / _Tp ) * pd * pd - Kd * vn * pd;
		//} else {
		//	fn = 0.5 * Kp * _Tp + Kp * pd - Kd * vn * pd;
		//}
		//fn /= nc;

		// compute friction force (try a static friction model first)
		Vec3 Ft = Kfp * (P0 - P) - Kfd * Vt; 
		Ft *= 1./ gReal(nc);
		psa->_bStaticFriction[_cols[i]._idxVertexA] = 1; // static friction applied

		// switch to a dynamic friction model if the force exceeds the maximum static friction force or the slip velocity is larger than the tolerance limit 
		if ( Norm(Ft) > mu_s * fn ) {
			if ( vs > 1E-12 ) {
				Ft  = ( -mu_d * fn * ((1. - exp(-k * vs)) / vs) ) * Vt; // dynamic friction
			} else {
				Ft = ( -mu_d * fn * k ) * Vt;
			}
			psa->_xf_ref[_cols[i]._idxVertexA] = Inv(*(psb->_pT)) * P; // update the reference contact point
			psa->_bStaticFriction[_cols[i]._idxVertexA] = 0; // dynamic friction applied
		}

		// total contact force
		Vec3 Fc = fn * N + Ft;
		psa->_fc[_cols[i]._idxVertexA] = Fc; // save the contact force for rendering

		// apply the force to the surfaces
		dse3 F = dse3(Cross(P, Fc), Fc);
		*(psa->_pF) += dAd(*(psa->_pT), F);
		*(psb->_pF) += dAd(*(psb->_pT), -F);
	}
}

#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <math.h>
#include "world.h"
#include "motor.h"
#include "tinyxml.h"
#include "xmlparser.h"
#include "glsub.h"
#include "rigidobject.h"
#include "surfmesh.h"
#include "utilfunc.h"
#include "cdfunc.h"
#include "forceclosure.h"

using namespace std;

static GLUquadricObj *qobj = gluNewQuadric();
static double s_timer = 0;
static double s_liftup_time;
static double s_rest_time;
static World::SimulPhase s_next_state_after_rest;
static int s_idx_force = 0, s_idx_moment = 0;

// shaking
static double s_shake_time;
static double _shake_d, _shake_a, _shake_vm;
static int _num_shakes;

bool World::loadShake(const char *filepath)
{
	_shake_d = 0.1, _shake_a = 5.0, _shake_vm = 5.0, _num_shakes = 1;
	std::ifstream fin(filepath);
	if ( !fin.is_open() ) return false;
	fin >> _shake_d >> _shake_a >> _shake_vm >> _num_shakes;
	fin.close();
	return true;
}

bool World::loadFromXML(const char *filepath)
{
	TiXmlDocument doc(filepath);
	if ( !doc.LoadFile() ) {
		cerr << "error:: failed in loading file: " << filepath << endl;
		return false;
	}

	_xmlsetkeywords();
	_xmlparse(&doc);

	if ( !getReady() ) {
		cerr << "error:: failed in getting ready" << endl;
		return false;
	}

	_estimateBoundingSphere();

	return true;
}

void World::render()
{
	// world coordinate frame
	if ( _b_show_coord_frames ) {
		glsub_draw_coord_sys(qobj, 0.2*getBoundingSphereRadius());
	}

	// ground
	_ground.render();

	if ( !!_phand ) {
		_phand->render();

		if ( _b_show_coord_frames ) {
			// link frames
			vector<RigidBody*> pbodies = _phand->getBodies();
			for (size_t i=0; i<pbodies.size(); i++) {
				glPushMatrix();
				SE3 T = pbodies[i]->getPoseGlobal();
#if GEAR_DOUBLE_PRECISION
				glMultMatrixd(T.GetArray());
#else
				glMultMatrixf(T.GetArray());
#endif
				glsub_draw_coord_sys(qobj, 0.07*getBoundingSphereRadius());
				glPopMatrix();
			}

			// end-effector frame
			if ( !!_phand->_pbody_ee ) {
				glPushMatrix();
				SE3 T = _phand->_pbody_ee->getPoseGlobal() * _phand->_T_ee;
#if GEAR_DOUBLE_PRECISION
				glMultMatrixd(T.GetArray());
#else
				glMultMatrixf(T.GetArray());
#endif
				glsub_draw_coord_sys(qobj, 0.1*getBoundingSphereRadius());
				glPopMatrix();
			}
		}
	}

	if ( !!_pobject ) {
		_pobject->render();

		if ( _b_show_coord_frames ) {

			glPushMatrix();

#if GEAR_DOUBLE_PRECISION
			glMultMatrixd(_pobject->getPose().GetArray());
#else
			glMultMatrixf(_pobject->getPose().GetArray());
#endif
			// local object frame
			glsub_draw_coord_sys(qobj, 0.1*getBoundingSphereRadius());

			// grasp contact center (w.r.t. local object frame)
			if ( _b_show_gcc_obj && _b_set_gcc_obj ) {
				glTranslated(_gcc_obj[0], _gcc_obj[1], _gcc_obj[2]);
				glColor3f(0,1,0);
				gluSphere(qobj, 0.04*getBoundingSphereRadius(), 10, 10);
			}

			glPopMatrix();
		}

		// external force
		double L;
		Vec3 c = _pobject->getPose() * _pobject->getCOM();
		glTranslated(c[0], c[1], c[2]);
		L = _force_scale * Norm(_external_force);
		glColor3f(1,0,0);
		glsub_draw_arrow(qobj, _external_force, L*0.7, L*0.3, 0.01, 0.02, 10, 1, 2);
		L = _moment_scale * Norm(_external_moment);
		glColor3f(1,0,1);
		glsub_draw_arrow_with_double_heads(qobj, _external_moment, L*0.7, L*0.3, 0.01, 0.02, 10, 1, 2);

	}
}

void World::enableRenderingBoundingBoxes(bool b)
{
	for (size_t i=0; i<_colchecker.getSurfaces().size(); i++) {
		_colchecker.getSurfaces()[i]->enableRenderingBoundingBox(b);
	}
}

void World::enableRenderingContactPoints(bool b)
{
	for (size_t i=0; i<_colchecker.getSurfaces().size(); i++) {
		_colchecker.getSurfaces()[i]->enableRenderingContactPoints(b);
	}
}

void World::enableRenderingContactForces(bool b)
{
	for (size_t i=0; i<_colchecker.getSurfaces().size(); i++) {
		_colchecker.getSurfaces()[i]->enableRenderingContactForces(b);
	}
}

void World::enableRenderingCollidingSurfacePatches(bool b)
{
	for (size_t i=0; i<_colchecker.getSurfaces().size(); i++) {
		_colchecker.getSurfaces()[i]->enableRenderingCollidingSurfacePatches(b);
		_colchecker.getSurfaces()[i]->enableRenderingContactPoints(b);
	}
	for (size_t i=0; i<_ground.getSurfaces().size(); i++) {
		_ground.getSurfaces()[i]->enableRenderingCollidingSurfacePatches(false);
	}
}

void World::setRenderingMode(int mode)
{
	int cnt;
	switch ( mode ) {
	case 0: // restore original setting
		cnt=0;
		for (list<GBody*>::iterator iter_pbody = _phand->pBodies.begin(); iter_pbody != _phand->pBodies.end(); iter_pbody++) {
			for (size_t i=0; i<((RigidBody*)(*iter_pbody))->getSurfaces().size(); i++) {
				((RigidBody*)(*iter_pbody))->getSurface(i)->enableRendering(_surf_brender[cnt]);
				((RigidBody*)(*iter_pbody))->getSurface(i)->setDrawType(_surf_draw_types[cnt]);
				cnt++;
			}
		}
		if ( !!_pobject ) {
			for (size_t i=0; i<_pobject->getSurfaces().size(); i++) {
				_pobject->getSurface(i)->enableRendering(_surf_brender[cnt]);
				_pobject->getSurface(i)->setDrawType(_surf_draw_types[cnt]);
				cnt++;
			}
		}
		for (size_t i=0; i<_ground.getSurfaces().size(); i++) {
			_ground.getSurface(i)->enableRendering(_surf_brender[cnt]);
			_ground.getSurface(i)->setDrawType(_surf_draw_types[cnt]);
			cnt++;
		}
		break;
	case 1: // wireframe only
		cnt=0;
		for (list<GBody*>::iterator iter_pbody = _phand->pBodies.begin(); iter_pbody != _phand->pBodies.end(); iter_pbody++) {
			for (size_t i=0; i<((RigidBody*)(*iter_pbody))->getSurfaces().size(); i++) {
				((RigidBody*)(*iter_pbody))->getSurface(i)->enableRendering(_surf_brender[cnt]);
				((RigidBody*)(*iter_pbody))->getSurface(i)->setDrawType(SurfaceMesh::DT_WIRE);
				cnt++;
			}
		}
		if ( !!_pobject ) {
			for (size_t i=0; i<_pobject->getSurfaces().size(); i++) {
				_pobject->getSurface(i)->enableRendering(_surf_brender[cnt]);
				_pobject->getSurface(i)->setDrawType(SurfaceMesh::DT_WIRE);
				cnt++;
			}
		}
		for (size_t i=0; i<_ground.getSurfaces().size(); i++) {
			_ground.getSurface(i)->enableRendering(_surf_brender[cnt]);
			_ground.getSurface(i)->setDrawType(SurfaceMesh::DT_WIRE);
			cnt++;
		}
		break;
	case 2: // show collision meshes only
		cnt=0;
		for (list<GBody*>::iterator iter_pbody = _phand->pBodies.begin(); iter_pbody != _phand->pBodies.end(); iter_pbody++) {
			for (size_t i=0; i<((RigidBody*)(*iter_pbody))->getSurfaces().size(); i++) {
				if ( ((RigidBody*)(*iter_pbody))->getSurface(i)->isEnabledCollision() ) {
					((RigidBody*)(*iter_pbody))->getSurface(i)->enableRendering(true);
					((RigidBody*)(*iter_pbody))->getSurface(i)->setDrawType(SurfaceMesh::DT_WIRE);
				} else {
					((RigidBody*)(*iter_pbody))->getSurface(i)->enableRendering(false);
					((RigidBody*)(*iter_pbody))->getSurface(i)->setDrawType(SurfaceMesh::DT_WIRE);
				}
			}
		}
		if ( !!_pobject ) {
			for (size_t i=0; i<_pobject->getSurfaces().size(); i++) {
				if ( _pobject->getSurface(i)->isEnabledCollision() ) {
					_pobject->getSurface(i)->enableRendering(true);
					_pobject->getSurface(i)->setDrawType(SurfaceMesh::DT_WIRE);
				} else {
					_pobject->getSurface(i)->enableRendering(false);
					_pobject->getSurface(i)->setDrawType(SurfaceMesh::DT_WIRE);
				}
			}
		}
		for (size_t i=0; i<_ground.getSurfaces().size(); i++) {
				if ( _ground.getSurface(i)->isEnabledCollision() ) {
					_ground.getSurface(i)->enableRendering(true);
					_ground.getSurface(i)->setDrawType(SurfaceMesh::DT_WIRE);
				} else {
					_ground.getSurface(i)->enableRendering(false);
					_ground.getSurface(i)->setDrawType(SurfaceMesh::DT_WIRE);
				}
		}
		break;
	case 3: // show all
		cnt=0;
		for (list<GBody*>::iterator iter_pbody = _phand->pBodies.begin(); iter_pbody != _phand->pBodies.end(); iter_pbody++) {
			for (size_t i=0; i<((RigidBody*)(*iter_pbody))->getSurfaces().size(); i++) {
				((RigidBody*)(*iter_pbody))->getSurface(i)->enableRendering(true);
				((RigidBody*)(*iter_pbody))->getSurface(i)->setDrawType(_surf_draw_types[cnt]);
				cnt++;
			}
		}
		if ( !!_pobject ) {
			for (size_t i=0; i<_pobject->getSurfaces().size(); i++) {
				_pobject->getSurface(i)->enableRendering(true);
				_pobject->getSurface(i)->setDrawType(_surf_draw_types[cnt]);
				cnt++;
			}
		}
		break;
		for (size_t i=0; i<_ground.getSurfaces().size(); i++) {
			_ground.getSurface(i)->enableRendering(true);
			_ground.getSurface(i)->setDrawType(_surf_draw_types[cnt]);
			cnt++;
		}
	}
}

bool World::getReady()
{
	for (size_t i=0; i<_ground.pSurfs.size(); i++) {
		if ( !_ground.pSurfs[i]->getReady() ) return false;
	}
	if ( !!_phand ) {
		if ( !_phand->getReady() ) return false;
	}
	if ( !!_pobject ) {
		if ( !_pobject->getReady() ) return false;
	}

	_scanCollisionSurfacePairs();

	_setSimulationSave();

	_estimateBoundingSphere();

	_saveRenderingOptions();

	initSimulation();

	return true;
}

void World::initForces()
{
	if ( !!_phand ) {
		_phand->initBodyForces();
	}
	if ( !!_pobject ) {
		_pobject->initForce();
	}
}

void World::saveState(const char *filepath)
{
	// save state to memory
	_state_double.clear();
	_state_int.clear();
	for (std::list<GCoordinate*>::iterator iter_pcoord = _phand->pCoordinates.begin(); iter_pcoord != _phand->pCoordinates.end(); iter_pcoord++) {
		_state_double.push_back((*iter_pcoord)->q);
		_state_double.push_back((*iter_pcoord)->dq);
		_state_double.push_back((*iter_pcoord)->ddq);
		_state_double.push_back((*iter_pcoord)->tau);
	}
	for (int j=0; j<16; j++) {
		_state_double.push_back(_pobject->_T[j]);
	}
	for (int j=0; j<6; j++) {
		_state_double.push_back(_pobject->_V[j]);
		_state_double.push_back(_pobject->_F[j]);
	}
	for (int i=0; i<_colchecker.getNumSurfaces(); i++) {
		RigidSurface *psurf = _colchecker.getSurfaces()[i];
		for (int j=0; j<psurf->getNumVertices(); j++) {
			_state_int.push_back(psurf->_bColSurfPatch[j]);
			_state_int.push_back(psurf->_bColSeedVertex[j]);
			_state_int.push_back(psurf->_bContact[j]);
			_state_int.push_back(psurf->_bContactPrev[j]);
			_state_double.push_back(psurf->_xf_ref[j][0]);
			_state_double.push_back(psurf->_xf_ref[j][1]);
			_state_double.push_back(psurf->_xf_ref[j][2]);
			_state_int.push_back(psurf->_bStaticFriction[j]);
		}
		_state_int.push_back(psurf->_collidingVertexIndices.size());
		for (size_t j=0; j<psurf->_collidingVertexIndices.size(); j++) {
			_state_int.push_back(psurf->_collidingVertexIndices[j]);
		}
		_state_int.push_back(psurf->_collidingSurfPatchIndices.size());
		for (size_t j=0; j<psurf->_collidingSurfPatchIndices.size(); j++) {
			_state_int.push_back(psurf->_collidingSurfPatchIndices[j]);
		}
		_state_int.push_back(psurf->_seedVertexIndices.size());
		for (size_t j=0; j<psurf->_seedVertexIndices.size(); j++) {
			_state_int.push_back(psurf->_seedVertexIndices[j]);
		}
	}

	// save state to file if requested
	if ( filepath != NULL ) {
		ofstream fout(filepath);
		fout << _state_double.size() << endl;
		for (size_t i=0; i<_state_double.size(); i++) {
			fout << _state_double[i] << " ";
		}
		fout << endl;
		fout << _state_int.size() << endl;
		for (size_t i=0; i<_state_int.size(); i++) {
			fout << _state_int[i] << " ";
		}
		fout << endl;
		fout.close();
	}
}

void World::restoreState(const char *filepath)
{
	// load state from file if requested
	if ( filepath != NULL ) {
		int n=0;
		ifstream fin(filepath);
		fin >> n;
		_state_double.resize(n);
		for (size_t i=0; i<n; i++) {
			fin >> _state_double[i];
		}
		fin >> n;
		_state_int.resize(n);
		for (size_t i=0; i<n; i++) {
			fin >> _state_int[i];
		}
		fin.close();
	}

	// load state from memory
	int cntd=0, cnti=0;
	for (std::list<GCoordinate*>::iterator iter_pcoord = _phand->pCoordinates.begin(); iter_pcoord != _phand->pCoordinates.end(); iter_pcoord++) {
		(*iter_pcoord)->q = _state_double[cntd++];
		(*iter_pcoord)->dq = _state_double[cntd++];
		(*iter_pcoord)->ddq = _state_double[cntd++];
		(*iter_pcoord)->tau = _state_double[cntd++];
	}
	for (int j=0; j<16; j++) {
		_pobject->_T[j] = _state_double[cntd++];
	}
	for (int j=0; j<6; j++) {
		_pobject->_V[j] = _state_double[cntd++];
		_pobject->_F[j] = _state_double[cntd++];
	}
	for (int i=0; i<_colchecker.getNumSurfaces(); i++) {
		RigidSurface *psurf = _colchecker.getSurfaces()[i];
		for (int j=0; j<psurf->getNumVertices(); j++) {
			psurf->_bColSurfPatch[j] = _state_int[cnti++];
			psurf->_bColSeedVertex[j] = _state_int[cnti++];
			psurf->_bContact[j] = _state_int[cnti++];
			psurf->_bContactPrev[j] = _state_int[cnti++];
			psurf->_xf_ref[j][0] = _state_double[cntd++];
			psurf->_xf_ref[j][1] = _state_double[cntd++];
			psurf->_xf_ref[j][2] = _state_double[cntd++];
			psurf->_bStaticFriction[j] = _state_int[cnti++];
		}
		psurf->_collidingVertexIndices.resize(_state_int[cnti++]);
		for (size_t j=0; j<psurf->_collidingVertexIndices.size(); j++) {
			psurf->_collidingVertexIndices[j] = _state_int[cnti++];
		}
		psurf->_collidingSurfPatchIndices.resize(_state_int[cnti++]);
		for (size_t j=0; j<psurf->_collidingSurfPatchIndices.size(); j++) {
			psurf->_collidingSurfPatchIndices[j] = _state_int[cnti++];
		}
		psurf->_seedVertexIndices.resize(_state_int[cnti++]);
		for (size_t j=0; j<psurf->_seedVertexIndices.size(); j++) {
			psurf->_seedVertexIndices[j] = _state_int[cnti++];
		}
	}
	updateKinematics();
}

bool World::stepSimulation(double t)
{
	// init forces
	initForces();

	// check collision
	_colchecker.checkCollision();

	// check if the hand collides with the ground
	if ( isHandCollidingWithGround() ) {
		//cout << "Hand collided with ground at t = " << t << "sec !" << endl;
		return false;
	}

	// apply contact forces
	_colchecker.applyContactForces();

	// simulate hand
	if ( !!_phand ) {

		// close fingers if available
		_phand->updateStaticJointTorques(); 
		_phand->Close(_stepsize, t);

		// actions
		switch ( _simul_phase ) {

		case SS_START:
			_T_h2o_before_grasping = Inv(_phand->getEndEffectorTransform()) * _pobject->getPose();
			_simul_phase = SS_CLOSE_FINGER;
			break;

		case SS_CLOSE_FINGER:
			// if closed, have a rest and set SS_LIFTUP as the next state
			if ( _phand->isClosed() ) {
				_simul_phase = SS_REST;
				s_timer = 0;
				s_rest_time = 0.5;
				s_next_state_after_rest = SS_LIFTUP;
			}
			break;

		case SS_LIFTUP:
			// lift up the hand along z-direction when 0 <= s_timer <= tf
			s_liftup_time = _phand->Move(_stepsize, s_timer, Vec3(0,0,1), 0, _liftup_d, _liftup_vm, _liftup_a); 
			s_timer += _stepsize;
			// if liftup is done, have a rest and set SS_SHAKE as the next state
			if (s_timer >= s_liftup_time ) {
				_simul_phase = SS_REST;
				s_timer = 0;
				s_rest_time = 0.5;
				s_next_state_after_rest = SS_SHAKE;
			}
			break;

		case SS_SHAKE:
			if (!_num_shakes--) {
				// done shaking! rest
				_simul_phase = SS_REST;
				s_timer = 0;
				s_rest_time = 0.5;
				if ( !_apply_external_force || ( _gq_external_forces.size() == 0 && _gq_external_moments.size() == 0 ) ) {
					s_next_state_after_rest = SS_NO_APPLY_FORCE;
				} else {
					s_next_state_after_rest = SS_APPLY_FORCE;
				}
			} else {
				_simul_phase = SS_SHAKE_LEFT;
			}
			break;

		case SS_SHAKE_LEFT:
			// shake left when 0 <= s_timer <= tf
			s_shake_time = _phand->Move(_stepsize, s_timer, Vec3(1, 0, 0), 0, _shake_d, _shake_vm, _shake_a);
			s_timer += _stepsize;
			// if done shaking left, rest and set SS_SHAKE_RIGHT as next state
			if (s_timer >= s_shake_time) {
				_simul_phase = SS_SHAKE_RIGHT;
				s_timer = 0;
				s_rest_time = 0;
			}
			break;

		case SS_SHAKE_RIGHT:
			// shake right when 0 <= s_timer <= tf
			s_shake_time = _phand->Move(_stepsize, s_timer, Vec3(-1, 0, 0), 0, _shake_d, _shake_vm, _shake_a);
			s_timer += _stepsize;
			// if done shaking right, rest and set SS_SHAKE as next state
			if (s_timer >= s_shake_time) {
				_simul_phase = SS_SHAKE;
				s_timer = 0;
				s_rest_time = 0;
			}
			break;

		case SS_REST:
			// rest
			s_timer += _stepsize;
			// if rest time is over, go to the predefined next state
			if ( s_timer >= s_rest_time ) {
				_simul_phase = s_next_state_after_rest;
				s_timer = 0;
				if ( s_next_state_after_rest == SS_LIFTUP ) {
					_T_h2o_before_liftup = Inv(_phand->getEndEffectorTransform()) * _pobject->getPose();
					_gcc_obj = getGraspContactCenter(); // calculate grasp contact center
					_b_set_gcc_obj = 1; // start rendering grasp contact center
				}
				if ( s_next_state_after_rest == SS_NO_APPLY_FORCE ) {
					_num_contact_links_after_liftup = getNumOfContactLinks();
					_min_dist_after_liftup = ComputeMinDist();
					SE3 T_h2o = Inv(_phand->getEndEffectorTransform()) * _pobject->getPose();
					_deviation_pos_final = Norm(_T_h2o_before_grasping * _pobject->getCOM() - T_h2o * _pobject->getCOM());
					_deviation_ori_final = Norm(Log(~(_T_h2o_before_grasping.GetRotation())*T_h2o.GetRotation()));
				}
				if ( s_next_state_after_rest == SS_APPLY_FORCE ) {
					_num_contact_links_after_liftup = getNumOfContactLinks();
					_min_dist_after_liftup = ComputeMinDist();
					SE3 T_h2o = Inv(_phand->getEndEffectorTransform()) * _pobject->getPose();
					_deviation_pos_final = Norm(_T_h2o_before_grasping * _pobject->getCOM() - T_h2o * _pobject->getCOM());
					_deviation_ori_final = Norm(Log(~(_T_h2o_before_grasping.GetRotation())*T_h2o.GetRotation()));

					// prepare to apply external forces
					_T_g2o_before_applying_force = _pobject->getPose();
					s_idx_force = s_idx_moment = 0;
					saveState();
					_phand->lockMotors(); // lock motors before force test
					if ( _num_contact_links_after_liftup < 2 ) {
						_simul_phase = SS_DONE;
					}
				}
			}
			break;

		case SS_NO_APPLY_FORCE:
			_simul_phase = SS_DONE;
			break;

		case SS_DONE:
			break;
		}
	}

	if ( !!_pobject ) {

		_external_force.SetZero();
		_external_moment.SetZero();

		// set external force
		if ( _simul_phase == SS_APPLY_FORCE ) {
			if ( s_idx_force < _gq_external_forces.size() ) {
				_external_force = _gq_external_forces[s_idx_force].getForce(s_timer);
			} else if ( s_idx_moment < _gq_external_moments.size() ) {
				_external_moment = _gq_external_moments[s_idx_moment].getMomemt(s_timer);
			}
			_pobject->addGlobalForceAtLocalPosition(_external_force, _pobject->getCOM());
			_pobject->addGlobalMoment(_external_moment);
		}

		// object dynamics
		_pobject->addGravityForce(_gravity);
		_pobject->stepSimulation(_stepsize);

		// post-process for measuring grasp quality
		if ( _simul_phase == SS_CLOSE_FINGER || _simul_phase == SS_LIFTUP ) {
			SE3 T_h2o = Inv(_phand->getEndEffectorTransform()) * _pobject->getPose();
			double devi_pos = Norm(_T_h2o_before_grasping * _pobject->getCOM() - T_h2o * _pobject->getCOM());
			double devi_ori = Norm(Log(~(_T_h2o_before_grasping.GetRotation())*T_h2o.GetRotation()));
			if ( devi_pos > _deviation_pos_max_grasping ) {
				_deviation_pos_max_grasping = devi_pos;
			}
			if ( devi_ori > _deviation_ori_max_grasping ) {
				_deviation_ori_max_grasping = devi_ori;
			}
		}
		if ( _simul_phase == SS_LIFTUP ) {
			SE3 T_h2o = Inv(_phand->getEndEffectorTransform()) * _pobject->getPose();
			//double devi_pos = Norm(_T_h2o_before_liftup * _pobject->getCOM() - T_h2o * _pobject->getCOM());
			double devi_pos = Norm(_T_h2o_before_liftup * _gcc_obj - T_h2o * _gcc_obj);
			double devi_ori = Norm(Log(~(_T_h2o_before_liftup.GetRotation())*T_h2o.GetRotation()));
			if ( devi_pos > _deviation_pos_max_liftup ) {
				_deviation_pos_max_liftup = devi_pos;
			}
			if ( devi_ori > _deviation_ori_max_liftup ) {
				_deviation_ori_max_liftup = devi_ori;
			}
		}
		if ( _simul_phase == SS_APPLY_FORCE ) {
			//double devi_pos = Norm(_T_g2o_before_applying_force * _pobject->getCOM() - _pobject->getPose() * _pobject->getCOM());
			double devi_pos = Norm(_T_g2o_before_applying_force * _gcc_obj - _pobject->getPose() * _gcc_obj);
			double devi_ori = Norm(Log(~(_T_g2o_before_applying_force.GetRotation()) * _pobject->getOrientation()));
			bool bgonext = false;
			if ( s_idx_force < _gq_external_forces.size() ) {
				switch ( _gq_external_forces[s_idx_force]._gqm_type ) {
				case FRM_MAX_DEVIATION:
					if ( devi_pos > _deviation_pos_max_externalforce[s_idx_force] ) { 
						_deviation_pos_max_externalforce[s_idx_force] = devi_pos; 
					}
					if ( _gq_external_forces[s_idx_force].isTimeOut(s_timer) ) {
						bgonext = true;
					}
					break;
				case FRM_RESISTANT_TIME:
					if ( devi_pos >= _deviation_limit_pos_externalforce || _gq_external_forces[s_idx_force].isTimeOut(s_timer) ) {
						if ( _time_resistant_externalforce[s_idx_force] < 0 ) { // _time_resistant_externalforce[] was initialized with -1
							_time_resistant_externalforce[s_idx_force] = s_timer;
						}
						bgonext = true;
					}
					break;
				case FRM_ALL:
					if ( devi_pos > _deviation_pos_max_externalforce[s_idx_force] ) { 
						_deviation_pos_max_externalforce[s_idx_force] = devi_pos; 
					}
					if ( devi_pos >= _deviation_limit_pos_externalforce ) {
						if ( _time_resistant_externalforce[s_idx_force] < 0 ) { // _time_resistant_externalforce[] was initialized with -1
							_time_resistant_externalforce[s_idx_force] = s_timer;
						}
					}
					if ( _gq_external_forces[s_idx_force].isTimeOut(s_timer) ) {
						if ( _time_resistant_externalforce[s_idx_force] < 0 ) { // _time_resistant_externalforce[] was initialized with -1
							_time_resistant_externalforce[s_idx_force] = s_timer;
						}
						bgonext = true;
					}
					break;
				}
				if ( bgonext || isSimulationDiverged() ) {
					s_idx_force++;
					s_timer = 0;
					restoreState();
				}
			} else if ( s_idx_moment < _gq_external_moments.size() ) {
				switch ( _gq_external_moments[s_idx_moment]._gqm_type ) {
				case FRM_MAX_DEVIATION:
					if ( devi_ori > _deviation_ori_max_externalmoment[s_idx_moment] ) {
						_deviation_ori_max_externalmoment[s_idx_moment] = devi_ori;
					}
					if ( _gq_external_moments[s_idx_moment].isTimeOut(s_timer) ) {
						bgonext = true;
					}
					break;
				case FRM_RESISTANT_TIME:
					if ( devi_ori >= _deviation_limit_ori_externalmoment || _gq_external_moments[s_idx_moment].isTimeOut(s_timer) ) {
						if ( _time_resistant_externalmoment[s_idx_moment] < 0 ) { // _time_resistant_externalmoment[] was initialized with -1
							_time_resistant_externalmoment[s_idx_moment] = s_timer;
						}
						bgonext = true;
					}
					break;
				case FRM_ALL:
					if ( devi_ori > _deviation_ori_max_externalmoment[s_idx_moment] ) {
						_deviation_ori_max_externalmoment[s_idx_moment] = devi_ori;
					}
					if ( devi_ori >= _deviation_limit_ori_externalmoment ) {
						if ( _time_resistant_externalmoment[s_idx_moment] < 0 ) { // _time_resistant_externalmoment[] was initialized with -1
							_time_resistant_externalmoment[s_idx_moment] = s_timer;
						}
					}
					if ( _gq_external_moments[s_idx_moment].isTimeOut(s_timer) ) {
						if ( _time_resistant_externalmoment[s_idx_moment] < 0 ) { // _time_resistant_externalmoment[] was initialized with -1
							_time_resistant_externalmoment[s_idx_moment] = s_timer;
						}
						bgonext = true;
					}
					break;
				}
				if ( bgonext || isSimulationDiverged() ) {
					s_idx_moment++;
					s_timer = 0;
					restoreState();
				}
			}

			if (s_idx_force >= _gq_external_forces.size() && s_idx_moment >= _gq_external_moments.size() ) {
				_simul_phase = SS_DONE; // finish simulation!
				s_timer = 0;
				s_idx_force = s_idx_moment = 0;
				restoreState();
			} else {
				s_timer += _stepsize;
			}
		}
	}

	// check if the simulation blows up
	if ( isSimulationDiverged() ) { 
		//cout << "Simulation diverged at t = " << t << "sec !" << endl;
		return false;
	}

	return true;
}

void World::initSimulation()
{
	if ( !_phand || !_pobject ) return;

	for (size_t i=0; i<_colchecker.getNumSurfaces(); i++) {
		_colchecker.getSurfaces()[i]->resetContactInfo();
		for (size_t j=0; j<_colchecker.getSurfaces()[i]->getNumVertices(); j++) {
			_colchecker.getSurfaces()[i]->_bContactPrev[j] = 0;
		}
	}

	_phand->initSimulation();
	_pobject->initSimulation();

	_simul_phase = SS_START;

	_num_contact_links_after_liftup = 0;
	_min_dist_after_liftup = 0;

	_gcc_obj.SetZero();
	_T_h2o_before_grasping.SetIdentity();
	_T_h2o_before_liftup.SetIdentity();
	_T_g2o_before_applying_force.SetIdentity();
	_deviation_pos_final = _deviation_ori_final = 0;
	_deviation_pos_max_grasping = _deviation_ori_max_grasping = 0;
	_deviation_pos_max_liftup = _deviation_ori_max_liftup = 0;
		
	_deviation_pos_max_externalforce.resize(_gq_external_forces.size());
	_time_resistant_externalforce.resize(_gq_external_forces.size());
	for (size_t i=0; i<_gq_external_forces.size(); i++) {
		_deviation_pos_max_externalforce[i] = -1;
		_time_resistant_externalforce[i] = -1;
	}
	_deviation_ori_max_externalmoment.resize(_gq_external_moments.size());
	_time_resistant_externalmoment.resize(_gq_external_moments.size());
	for (size_t i=0; i<_gq_external_moments.size(); i++) {
		_deviation_ori_max_externalmoment[i] = -1;
		_time_resistant_externalmoment[i] = -1;
	}

	_external_force.SetZero();
	_external_moment.SetZero();

	s_timer = 0;
	s_idx_force = s_idx_moment = 0;

	_b_set_gcc_obj = 0;
}

void World::setObjectPoseUncertainty(int sample_size, Vec3 sigma_pos, double sigma_angle, Vec3 axis, SO3 R_principal_axes_pos, SO3 R_principal_axes_ori)
{
	_obj_pose_uncertainty.set(sample_size, sigma_pos, sigma_angle, axis, R_principal_axes_pos, R_principal_axes_ori);
}

void World::setObjectPoseUncertaintyWithHalfNormalMean(int sample_size, Vec3 half_normal_mean_pos, double half_normal_mean_angle, Vec3 axis, SO3 R_principal_axes_pos, SO3 R_principal_axes_ori)
{
	// How to obtain the standard deviation of normal distribution from the half normal distribution mean:
	// sigma = sqrt(pi/2)*x_hn where sigma = standard deviation of normal distribution, x_hn = mean of the half normal distribution
	// reference: "The Half-Normal distribution method for measurement error: two case studies" by J. Martin Bland
	gReal c = sqrt(3.14159/2.); // convert half normal distribution mean into standard deviation of normal distribution
	_obj_pose_uncertainty.set(sample_size, c*half_normal_mean_pos, c*half_normal_mean_angle, axis, R_principal_axes_pos, R_principal_axes_ori);
}

void World::scoreGraspQuality(std::vector<double> &qscores)
{
	qscores.resize(_num_score_types);

	// if hand collides with ground, set the scores with zeros and return
	if ( isHandCollidingWithGround() ) {
		for (size_t i=0; i<qscores.size(); i++) {
			qscores[i] = 0;
		}
		return;
	}

	// measure the grasp quality: number of contact links, quality values based on pose deviation
	if ( _num_contact_links_after_liftup >= 2 ) {
		if ( _num_contact_links_after_liftup == 2 ) {
			qscores[0] = 0.5;
		} else {
			qscores[0] = 1.0;
		}
		//qscores[1] = _deviation_pos_max_liftup < _deviation_limit_pos ? 1 - _deviation_pos_max_liftup / _deviation_limit_pos : 0;
		//qscores[2] = _deviation_ori_max_liftup < _deviation_limit_ori ? 1 - _deviation_ori_max_liftup / _deviation_limit_ori : 0;
		//qscores[1] = _deviation_pos_max_grasping < _deviation_limit_pos ? 1 - _deviation_pos_max_grasping / _deviation_limit_pos : 0;
		//qscores[2] = _deviation_ori_max_grasping < _deviation_limit_ori ? 1 - _deviation_ori_max_grasping / _deviation_limit_ori : 0;
		qscores[1] = _deviation_pos_final < _deviation_limit_pos ? 1 - _deviation_pos_final / _deviation_limit_pos : 0;
		qscores[2] = _deviation_ori_final < _deviation_limit_ori ? 1 - _deviation_ori_final / _deviation_limit_ori : 0;
	} else {
		qscores[0] = qscores[1] = qscores[2] = 0;
	}

	// meaasure the grasp quality: minimum resistant time, maximum pose deviation during applying external forces and moments
	double min_resi_time_force = 1E10, min_resi_time_moment = 1E10;
	double max_pos_dev_force = -1, max_ori_dev_moment = -1;
	for (size_t j=0; j<_gq_external_forces.size(); j++) {
		switch ( _gq_external_forces[j]._gqm_type ) {
		case FRM_MAX_DEVIATION:
			if ( _deviation_pos_max_externalforce[j] > max_pos_dev_force ) {
				max_pos_dev_force = _deviation_pos_max_externalforce[j];
			}
			break;
		case FRM_RESISTANT_TIME:
			if ( _time_resistant_externalforce[j] < min_resi_time_force ) {
				min_resi_time_force = _time_resistant_externalforce[j];
			}
			break;
		case FRM_ALL:
			if ( _deviation_pos_max_externalforce[j] > max_pos_dev_force ) {
				max_pos_dev_force = _deviation_pos_max_externalforce[j];
			}
			if ( _time_resistant_externalforce[j] < min_resi_time_force ) {
				min_resi_time_force = _time_resistant_externalforce[j];
			}
			break;
		}
	}
	for (size_t j=0; j<_gq_external_moments.size(); j++) {
		switch ( _gq_external_moments[j]._gqm_type ) {
		case FRM_MAX_DEVIATION:
			if ( _deviation_ori_max_externalmoment[j] > max_ori_dev_moment ) {
				max_ori_dev_moment = _deviation_ori_max_externalmoment[j];
			}
			break;
		case FRM_RESISTANT_TIME:
			if ( _time_resistant_externalmoment[j] < min_resi_time_moment ) {
				min_resi_time_moment = _time_resistant_externalmoment[j];
			}
			break;
		case FRM_ALL:
			if ( _deviation_ori_max_externalmoment[j] > max_ori_dev_moment ) {
				max_ori_dev_moment = _deviation_ori_max_externalmoment[j];
			}
			if ( _time_resistant_externalmoment[j] < min_resi_time_moment ) {
				min_resi_time_moment = _time_resistant_externalmoment[j];
			}
			break;
		}
	}

	qscores[3] = qscores[4] = qscores[5] = qscores[6] = 0;
	if ( min_resi_time_force >= 0 && min_resi_time_force < 100 ) {
		qscores[3] = min_resi_time_force;
	}
	if ( min_resi_time_moment >= 0 && min_resi_time_moment < 100 ) {
		qscores[4] = min_resi_time_moment;
	}
	if ( max_pos_dev_force >= 0 ) {
		qscores[5] = max_pos_dev_force < _deviation_limit_pos_externalforce ? 1 - max_pos_dev_force / _deviation_limit_pos_externalforce : 0;
	}
	if ( max_ori_dev_moment >= 0 ) {
		qscores[6] = max_ori_dev_moment < _deviation_limit_ori_externalmoment ? 1 - max_ori_dev_moment / _deviation_limit_ori_externalmoment : 0;
	}

	// measure the grasp quality: minimum distance from the origin to the convex hull faces of the contact wrenches (after lift-up)
	qscores[7] = _min_dist_after_liftup;
}

bool World::measureGraspQuality(std::vector<double> &quality_scores, const GraspSet::Grasp &grasp, bool bsavesimul, const char *folderpath)
{
	if ( !_phand || !_pobject ) return false;
	if ( _obj_pose_uncertainty.size() <= 0 ) {
		cerr << "error:: sample size = 0" << endl;
		return false;
	}

	double tf = 1000;
	vector<double> qscores_sum(_num_score_types, 0.0), qscores(_num_score_types);

	_phand->showMotorMessage(false);

	// save the current world (hand configuration, hand end-effector pose, object pose)
	vector<double> q_0; _phand->getJointValues(q_0);
	SE3 T_ee_0 = _phand->getEndEffectorTransform();
	SE3 T_obj_0 = _pobject->getPose();

	// grasp set (grasps with uncertainty)
	string gfilepath = folderpath; gfilepath += "_with_uncertainty.gst";
	GraspSet gs; 
	for (size_t i=0; i<_obj_pose_uncertainty.size(); i++) {
		SE3 T_obj; // object pose with the error from uncertainty
		T_obj.SetPosition(_obj_pose_uncertainty._poses[i].GetPosition() + T_obj_0.GetPosition());
		T_obj.SetRotation(_obj_pose_uncertainty._poses[i].GetRotation() * T_obj_0.GetRotation()); // R * exp([Rt*w]) = R * Rt * exp([w]) * R = exp([w]) * R where exp([w]) is a rotational error in {global}
		GraspSet::Grasp gra;
		gra._T = Inv(T_obj_0) * T_obj * grasp._T;	// relative pose from the object to the end-effector
		gra._preshape = grasp._preshape;
		gra._score = 0;
		gs._grasps.push_back(gra);
	}
	gs.save(gfilepath.c_str());

	// start simulation with uncertainty
	string logfile = folderpath; logfile += "_log.m";
	ofstream fout(logfile.c_str());
	cout << "sample_index, grasp quality scores [sA, sBp, sBr, sCf, sCt, sDf, sDt, sE]" << endl;
	fout << "% sample_index, grasp quality scores [sA, sBp, sBr, sCf, sCt, sDf, sDt, sE], num of contact links after liftup" << endl;
	fout << "%    sA = score based on number of contact links" << endl;
	fout << "%    sBp/sBr = scores based on final pose (position/orientation) deviation after grasping (finger closing and liftup)" << endl;
	fout << "%    sCf/sCt = scores based on resisting time to external forces/moments" << endl;
	fout << "%    sDf/sDt = scores based on maximum position/orientation deviation by external forces/moments" << endl;
	fout << "%    sE = scores based on the force-closure analysis after grasping (minimum distance from the origin to the convex hull of the contact wrenches)" << endl;
	fout << endl;
	fout << "simlogdata = [" << endl;

	int cnt = 0;
	for (size_t i=0; i<gs.size(); i++) {
		// set simulation file path
		string simfilepath;
		if ( !!folderpath ) {
			stringstream format; format << "%0" << ceil(log10((float)gs.size())) << "d";
			char str[10]; sprintf(str, format.str().c_str(), i);
			simfilepath = string(folderpath) + "/sim_" + str + ".sim";
		}

		// set initial condition
		initSimulation();
		_pobject->setPose(T_obj_0); 
		_phand->placeHandWithEndEffectorTransform(T_obj_0 * gs._grasps[i]._T);
		if ( !_phand->setJointValues(gs._grasps[i]._preshape) ) { 
			cerr << setw(5) << i << " : failed in setting joint values with preshape! skipped!" << endl; 
			fout << "%" << setw(5) << i << " : failed in setting joint values with preshape! skipped!" << endl;
			continue;
		}

		// run simulation
		bool bsuccess;
		if ( bsavesimul && simfilepath.size() > 0 ) {
			bsuccess = runSimulation(tf, simfilepath.c_str());
		} else {
			bsuccess = runSimulation(tf);
		}
		if ( !bsuccess && isSimulationDiverged() ) {
			cerr << setw(5) << i << " : simulation diverged! skipped!" << endl;
			fout << "%" << setw(5) << i << " : simulation diverged! skipped!" << endl;
			continue;
		}

		// score grasp quality
		scoreGraspQuality(qscores);

		// sum the scores for averaging later
		for (size_t j=0; j<qscores.size(); j++) {
			qscores_sum[j] += qscores[j];
		}

		// print scores
		cout << setw(5) << i << " ";
		fout << setw(5) << i << " ";
		if ( qscores.size() > 0 ) {
			cout << setw(5) << qscores[0] << " ";
			fout << setw(12) << qscores[0] << " ";
			for (size_t j=1; j<qscores.size(); j++) {
				cout << setw(8) << qscores[j] << " ";
				fout << setw(12) << qscores[j] << " ";
			}
		}
		cout << endl;
		fout << setw(12) << _num_contact_links_after_liftup << endl;

		cnt++; // count scoring for averaging later
	}

	fout << "];" << endl;
	quality_scores.resize(qscores_sum.size());
	for (size_t i=0; i<quality_scores.size(); i++) {
		quality_scores[i] = qscores_sum[i] / double(cnt);
	}
	cout << "averaged quality scores = [";
	fout << "quality_scores_average = [";
	for (size_t i=0; i<quality_scores.size(); i++) {
		cout << quality_scores[i] << ", ";
		fout << quality_scores[i] << ", ";
	}
	cout << "];" << endl;
	fout << "];" << endl;
	fout.close();

	// restore the simulation world
	initSimulation();
	_phand->placeHandWithEndEffectorTransform(T_ee_0);
	_phand->setJointValues(q_0);
	_pobject->setPose(T_obj_0);

	// reset current surface contact info not to confuse rendering in replay
	for (size_t i=0; i<_colchecker.getSurfaces().size(); i++) {
		_colchecker.getSurfaces()[i]->resetContactInfo();
	}

	return true;
}

bool World::measureGraspQualityWithoutUncertainty(const GraspSet &gs, const char *folderpath)
{
	if ( !_phand || !_pobject ) return false;
	size_t ddd = gs.size();

	if ( gs.size() == 0 ) {
		cout << "warning:: empty grasp set!" << endl;
		return true;
	}

	double tf = 1000;
	vector<double> qscores_sum(_num_score_types, 0.0), qscores(_num_score_types);

	_phand->showMotorMessage(false);

	// save the current world (hand configuration, hand end-effector pose, object pose)
	vector<double> q_0; _phand->getJointValues(q_0);
	SE3 T_ee_0 = _phand->getEndEffectorTransform();
	SE3 T_obj_0 = _pobject->getPose();

	// start simulation
	string logfile = folderpath; logfile += "_log.m";
	ofstream fout(logfile.c_str());
	cout << "sample_index, grasp quality scores [qs0, qs1(pos), qs2(ori), rtf, rtm], num_contact" << endl;
	fout << "% sample_index, grasp quality scores [qs0, qs1(pos), qs2(ori), rtf, rtm], num_contact" << endl;
	fout << endl;
	fout << "simlogdata = [" << endl;

	int cnt = 0;
	for (size_t i=0; i<gs.size(); i++) {

		// set initial condition
		initSimulation();
		_pobject->setPose(T_obj_0); 
		_phand->placeHandWithEndEffectorTransform(T_obj_0 * gs._grasps[i]._T);
		if ( !_phand->setJointValues(gs._grasps[i]._preshape) ) { 
			cerr << "failed in setting joint values with preshape!" << endl; 
			continue;
		}

		// run simulation
		bool bsuccess;
		if ( !!folderpath ) {
			stringstream format; format << "%0" << ceil(log10((float)gs.size())) << "d";
			char str[10]; sprintf(str, format.str().c_str(), cnt);
			stringstream filepath; filepath << string(folderpath) << "/sim_" << str << ".sim";
			bsuccess = runSimulation(tf, filepath.str().c_str());
		} else {
			bsuccess = runSimulation(tf);
		}
		if ( !bsuccess && isSimulationDiverged() ) {
			cerr << setw(5) << i << " : simulation diverged! skipped!" << endl;
			fout << "%" << setw(5) << i << " : simulation diverged! skipped!" << endl;
			continue;
		}

		// score grasp quality
		scoreGraspQuality(qscores);

		// sum the scores for averaging later
		for (size_t j=0; j<qscores.size(); j++) {
			qscores_sum[j] += qscores[j];
		}

		// print scores
		cout << setw(5) << i << " ";
		fout << setw(5) << i << " ";
		if ( qscores.size() > 0 ) {
			cout << setw(5) << qscores[0] << " ";
			fout << setw(12) << qscores[0] << " ";
			for (size_t j=1; j<qscores.size(); j++) {
				cout << setw(8) << qscores[j] << " ";
				fout << setw(12) << qscores[j] << " ";
			}
		}
		cout << endl;
		fout << setw(12) << _num_contact_links_after_liftup << endl;

		cnt++; // count scoring for averaging later
	}

	fout << "];" << endl;

	cout << "quality scores (average) = [";
	fout << "quality_scores_average = [";
	for (size_t i=0; i<qscores_sum.size(); i++) {
		cout << qscores_sum[i] / double(cnt) << ", ";
		fout << qscores_sum[i] / double(cnt) << ", ";
	}
	cout << "];" << endl;
	fout << "];" << endl;
	fout.close();

	// restore the simulation world
	initSimulation();
	_phand->placeHandWithEndEffectorTransform(T_ee_0);
	_phand->setJointValues(q_0);
	_pobject->setPose(T_obj_0);

	// reset current surface contact info not to confuse rendering in replay
	for (size_t i=0; i<_colchecker.getSurfaces().size(); i++) {
		_colchecker.getSurfaces()[i]->resetContactInfo();
	}

	return true;
}

void World::_init_fingerprint(FingerPrint &fp)
{
	fp._hit_cnt_ref = 0;
	fp._face_hit_cnt.resize(_pobject->_pSurfs.size());
	for (size_t i=0; i<fp._face_hit_cnt.size(); i++) {
		fp._face_hit_cnt[i].resize(_pobject->_pSurfs[i]->getNumFaces());
		for (size_t j=0; j<fp._face_hit_cnt[i].size(); j++) {
			fp._face_hit_cnt[i][j] = 0;
		}
	}
}

bool World::calcFingerPrintHistory(const GraspSet &gs, GraspSet &gs_finalconfig)
{
	gs_finalconfig.clear();
	if ( !_phand || !_pobject ) return false;
	if ( gs.size() == 0 ) {
		cout << "warning:: empty grasp set!" << endl;
		return true;
	}

	GraspSet::Grasp gfc; // grasp at the final configuration
	double tf = 1000;
	_phand->showMotorMessage(false);

	// save the current world (hand configuration, hand end-effector pose, object pose)
	vector<double> q_0; _phand->getJointValues(q_0);
	SE3 T_ee_0 = _phand->getEndEffectorTransform();
	SE3 T_obj_0 = _pobject->getPose();
	bool apply_external_force_tmp = _apply_external_force;

	// memory for temporal fingerprint and accumulated fingerprint
	FingerPrint fp_tmp, fp_acc;
	_init_fingerprint(fp_acc);
	
	// start simulation
	_apply_external_force = false; // do not apply external force for calculating finger print (this will be restored later)
	cout << "calculating finger print of the grasp set..." << endl;
	for (size_t i=0; i<gs.size(); i++) {

		cout << i << "/" << gs.size() << endl;

		// set initial condition
		initSimulation();
		_pobject->setPose(T_obj_0); 
		_phand->placeHandWithEndEffectorTransform(T_obj_0 * gs._grasps[i]._T);
		if ( !_phand->setJointValues(gs._grasps[i]._preshape) ) { 
			cerr << "error:: failed in setting joint values with preshape!" << endl; 
			continue;
		}

		// run simulation
		bool bsuccess = runSimulation(tf);
		if ( !bsuccess && isSimulationDiverged() ) {
			cerr << " : simulation diverged!" << endl;
			continue;
		}

		// init temporal fingerprint
		_init_fingerprint(fp_tmp);

		// set temporal fingerprint
		for (size_t j=0; j<_idx_obj_hand_surf_pairs.size(); j++) {
			CollisionChecker::CollidableSurfacePair &colsurfpair = _colchecker.getCollisionSurfacePairs()[_idx_obj_hand_surf_pairs[j]];
			int idxobjsurf = -1;
			for (size_t k=0; k<_pobject->_pSurfs.size(); k++) {
				if ( _pobject->_pSurfs[k] == colsurfpair._psurfB ) {
					idxobjsurf = k;
					break;
				}
			}
			if ( idxobjsurf < 0 ) {
				cerr << "error:: failed in finding object surface!" << endl;
				return false;
			}
			for (size_t k=0; k<colsurfpair._cols.size(); k++) {
				if ( colsurfpair._cols[k]._idxFaceB < 0 || colsurfpair._cols[k]._idxFaceB >= fp_tmp._face_hit_cnt[idxobjsurf].size() ) {
					cerr << "error:: invalid face index!" << endl;
					return false;
				}
				fp_tmp._face_hit_cnt[idxobjsurf][colsurfpair._cols[k]._idxFaceB] = 1;
			}
		}

		// accumulate fingerprint data
		for (size_t j=0; j<fp_tmp._face_hit_cnt.size(); j++) {
			for (size_t k=0; k<fp_tmp._face_hit_cnt[j].size(); k++) {
				fp_acc._face_hit_cnt[j][k] += fp_tmp._face_hit_cnt[j][k];
			}
		}
		fp_acc._hit_cnt_ref++;

		// add the accumulated fingerprint to the fingerprint history
		_fingerprinthistory.add(fp_acc);

		// save the final grasp configuration
		_phand->getJointValues(gfc._preshape);
		gfc._T = Inv(_pobject->getPose()) * _phand->getEndEffectorTransform();
		gfc._score = 0;
		gs_finalconfig._grasps.push_back(gfc);
	}

	// restore the simulation world
	initSimulation();
	_phand->placeHandWithEndEffectorTransform(T_ee_0);
	_phand->setJointValues(q_0);
	_pobject->setPose(T_obj_0);
	_apply_external_force = apply_external_force_tmp;

	// reset current surface contact info not to confuse rendering in replay
	for (size_t i=0; i<_colchecker.getSurfaces().size(); i++) {
		_colchecker.getSurfaces()[i]->resetContactInfo();
	}

	return true;
}

bool World::checkFingerPrintHistory()
{
	for (list<FingerPrint>::iterator iter = _fingerprinthistory._fingerprints.begin(); iter != _fingerprinthistory._fingerprints.end(); iter++) {
		if ( iter->_face_hit_cnt.size() != _pobject->_pSurfs.size() ) return false;
		for (size_t i=0; i<iter->_face_hit_cnt.size(); i++) {
			if ( iter->_face_hit_cnt[i].size() != _pobject->_pSurfs[i]->getNumFaces() ) return false;
		}
	}
	return true;
}

void World::computeGraspQualityErrorBar(int num_seed, int max_sample_size, const GraspSet::Grasp &grasp, const char *filepath)
{
	double tf = 1000;
	vector<double> qscores(_num_score_types), mean_qscores_avg(_num_score_types), sd_qscores_avg(_num_score_types);
	vector< vector<double> > qscores_sum(num_seed), qscores_avg(num_seed);
	for (int s=0; s<num_seed; s++) {
		qscores_sum[s].resize(_num_score_types);
		qscores_avg[s].resize(_num_score_types);
		for (int j=0; j<_num_score_types; j++) {
			qscores_sum[s][j] = 0;
			qscores_avg[s][j] = 0;
		}
	}

	_phand->showMotorMessage(false);

	// save the current world (hand configuration, hand end-effector pose, object pose)
	vector<double> q_0; _phand->getJointValues(q_0);
	SE3 T_ee_0 = _phand->getEndEffectorTransform();
	SE3 T_obj_0 = _pobject->getPose();

	// generate pose uncertainty models
	std::vector<PoseUncertainty> pu(num_seed);
	for (int i=0; i<num_seed; i++) {
		srand((unsigned)time(NULL));
		pu[i].set(max_sample_size, _obj_pose_uncertainty._sigma_pos, _obj_pose_uncertainty._sigma_angle, _obj_pose_uncertainty._axis, _obj_pose_uncertainty._R_principal_axes_pos, _obj_pose_uncertainty._R_principal_axes_ori);
	}

	// file saving
	ofstream fout(filepath);
	fout << "% num of seed = " << num_seed << endl;
	fout << "% max sample size = " << max_sample_size << endl;
	fout << "% grasp data: ";
	SO3 R = grasp._T.GetRotation();
	Vec3 p = grasp._T.GetPosition();
	for (size_t j=0; j<9; j++) {
		fout << R[j] << " ";
	}
	for (size_t j=0; j<3; j++) {
		fout << p[j] << " ";
	}
	for (size_t j=0; j<grasp._preshape.size(); j++) { fout << grasp._preshape[j] << " "; }
	fout << grasp._score << endl;
	fout << endl;
	fout << "% sample_index, mean average scores [qs0, qs1(pos), qs2(ori), rtf, rtm], sd of average scores" << endl;
	fout << endl;
	fout << "simlogdata = [" << endl;

	for (int i=0; i<max_sample_size; i++) {

		for (int s=0; s<num_seed; s++) {

			// g = grasp with uncertainty
			GraspSet::Grasp g;
			SE3 T_obj; // object pose with the error from uncertainty
			T_obj.SetPosition(pu[s]._poses[i].GetPosition() + T_obj_0.GetPosition());
			T_obj.SetRotation(pu[s]._poses[i].GetRotation() * T_obj_0.GetRotation()); // R * exp([Rt*w]) = R * Rt * exp([w]) * R = exp([w]) * R where exp([w]) is a rotational error in {global}
			g._T = Inv(T_obj_0) * T_obj * grasp._T;	// relative pose from the object to the end-effector
			g._preshape = grasp._preshape;
			g._score = 0;

			// set initial condition
			initSimulation();
			_pobject->setPose(T_obj_0); 
			_phand->placeHandWithEndEffectorTransform(T_obj_0 * g._T);
			if ( !_phand->setJointValues(g._preshape) ) { 
				cerr << "failed in setting joint values with preshape!" << endl; 
				continue;
			}

			// run simulation
			bool bsuccess = runSimulation(tf);

			// score grasp quality
			scoreGraspQuality(qscores);

			// sum the scores
			for (size_t j=0; j<qscores.size(); j++) {
				qscores_sum[s][j] += qscores[j];
			}

			// average the scores
			for (size_t j=0; j<qscores.size(); j++) {
				qscores_avg[s][j] = qscores_sum[s][j] / double(i+1);
			}
		}

		// find mean of the averaged scores at the i-th sampling
		for (int j=0; j<qscores.size(); j++) {
			double sum = 0;
			sum = 0;
			for (int s=0; s<num_seed; s++) {
				sum += qscores_avg[s][j];
			}
			mean_qscores_avg[j]  = sum / double(num_seed);
		}

		// find sample standard deviation of the averaged scores at the i-th sampling
		for (int j=0; j<qscores.size(); j++) {
			double var = 0;
			for (int s=0; s<num_seed; s++) {
				var += (qscores_avg[s][j] - mean_qscores_avg[j]) * (qscores_avg[s][j] - mean_qscores_avg[j]);
			}
			var /= double(num_seed-1);
			sd_qscores_avg[j] = sqrt(var);
		}

		// print mean and sd of the averaged scores at the i-th sampling
		cout << setw(5) << i << " ";
		fout << setw(5) << i << " ";
		for (size_t j=0; j<qscores.size(); j++) {
			cout << setw(12) << mean_qscores_avg[j] << " ";
			fout << setw(12) << mean_qscores_avg[j] << " ";
		}
		cout << endl << "      ";
		for (size_t j=0; j<qscores.size(); j++) {
			cout << setw(12) << sd_qscores_avg[j] << " ";
			fout << setw(12) << sd_qscores_avg[j] << " ";
		}
		cout << endl;
		fout << endl;
	}

	fout << "];" << endl;
	fout.close();

	// restore the simulation world
	initSimulation();
	_phand->placeHandWithEndEffectorTransform(T_ee_0);
	_phand->setJointValues(q_0);
	_pobject->setPose(T_obj_0);

	// reset current surface contact info not to confuse rendering in replay
	for (size_t i=0; i<_colchecker.getSurfaces().size(); i++) {
		_colchecker.getSurfaces()[i]->resetContactInfo();
	}
}

bool World::runSimulation(double tf, const char *filepath)
{
	double t=0;
	int counter_save = 0, max_counter_save = int(1./(double(_datasavefreq)*_stepsize));;
	bool bsuccess = true;

	_simuldata.clear_data_on_memory(); // clear memory for data saving
	_simuldata.write_data_into_memory(t); // save the first frame

	// run simulation
	while (1) {
		if ( !stepSimulation(t) ) {
			bsuccess = false;
			break;
		}
		t += _stepsize;

		counter_save++;
		if ( counter_save >= max_counter_save ) {
			_simuldata.write_data_into_memory(t);
			counter_save = 0;
		}
		if ( isSimulationDone() ) {
			break;
		}
		if ( t > tf ) {
			cout << "warning;: simulation stopped at the time limit! (tf = " << tf << ")" << endl;
			break;
		}
	}

	// file save
	if ( !!filepath ) {
		_simuldata.set_filepath_for_writing(filepath);
		_simuldata.save_data_on_memory_into_file();
	}

	return bsuccess;
}

int World::getNumOfContactLinks()
{
	int n=0;
	for (size_t i=0; i<_idx_surf_pairs_by_links.size(); i++) {
		bool bcontact = false;
		for (size_t j=0; j<_idx_surf_pairs_by_links[i].size(); j++) {
			if ( _colchecker.getCollisionSurfacePairs()[_idx_surf_pairs_by_links[i][j]]._cols.size() > 0 ) {
				bcontact = true;
				break;
			}
		}
		if ( bcontact ) n++;
	}
	return n;
}

Vec3 World::getGraspContactCenter()
{
	Vec3 gcc(0,0,0); // grasp contact center w.r.t. {global}
	int nlc = 0;
	for (size_t i=0; i<_idx_surf_pairs_by_links.size(); i++) {
		int nc = 0;
		Vec3 c(0,0,0);
		for (size_t j=0; j<_idx_surf_pairs_by_links[i].size(); j++) {
			for (size_t k=0; k<_colchecker.getCollisionSurfacePairs()[_idx_surf_pairs_by_links[i][j]]._cols.size(); k++) {
				c += _colchecker.getCollisionSurfacePairs()[_idx_surf_pairs_by_links[i][j]]._cols[k]._pos; // contact position in {global}
				nc++;
			}
		}
		if ( nc > 0 ) {
			c *= 1./double(nc);
			gcc += c;
			nlc++;
		}
	}

	if ( nlc > 0 ) {
		gcc *= 1./double(nlc);
	}

	return Inv(_pobject->getPose()) * gcc; // return grasp contact center w.r.t. {object}
}


bool World::isObjectMoving(double &v, double &w, double eps_v, double eps_w)
{
	v = Norm(_pobject->getFilteredVelocity().GetV());
	w = Norm(_pobject->getFilteredVelocity().GetW());
	if ( v > eps_v || w > eps_w ) return true;
	return false;
}

bool World::isHandCollidingWithGround()
{
	for (size_t i=0; i<_idx_hand_ground_surf_pairs.size(); i++) {
		if ( _colchecker.getCollisionSurfacePairs()[_idx_hand_ground_surf_pairs[i]]._cols.size() > 0 ) {
			return true;
		}
	}
	return false;
}

bool World::isHandCollidingWithGround_BoundingBoxCheckOnly()
{
	for (size_t i=0; i<_idx_hand_ground_surf_pairs.size(); i++) {
		CollisionChecker::CollidableSurfacePair &surfpair = _colchecker.getCollisionSurfacePairs()[_idx_hand_ground_surf_pairs[i]];
		BoundingBox &obbA = surfpair._psurfA->getBoundingBox(); // bounding box of surface A w.r.t. {global}
		BoundingBox &obbB = surfpair._psurfB->getBoundingBox(); // bounding box of surface B w.r.t. {global}
		if ( dcOBBOverlap(obbA.extents, obbA.position.GetArray(), obbA.orientation.GetArray(), obbB.extents, obbB.position.GetArray(), obbB.orientation.GetArray()) ) {
			return true;
		}
	}
	return false;
}

double World::ComputeMinDist()
{
	return AnalyzeContacts().mindist;
}

GRASPANALYSIS World::AnalyzeContacts()
{
	vector<Vec3> cp, cn; // contact positions and normals

	// get contacts on the object
	for (size_t i=0; i<_idx_obj_hand_surf_pairs.size(); i++) {
		CollisionChecker::CollidableSurfacePair &colsurfpair = _colchecker.getCollisionSurfacePairs()[_idx_obj_hand_surf_pairs[i]];
		for (size_t j=0; j<colsurfpair._cols.size(); j++) {
			cp.push_back(colsurfpair._cols[j]._pos);
			cn.push_back(colsurfpair._cols[j]._normal);
		}
	}

	return _AnalyzeContacts(cp, cn, _cp_hand_object._mu_s, 8);
}

GRASPANALYSIS World::_AnalyzeContacts(std::vector<Vec3> positions, std::vector<Vec3> normals, double frictioncoeff, int numfrictionconeaxes)
{
	if ( positions.size() != normals.size() ) {
		cerr << "size mismatch: contact positions and normals" << endl;
		return 	GRASPANALYSIS();
	}

	// force-closure analysis
	vector<CONTACT> c(positions.size());
	for (size_t i=0; i<c.size(); i++) {
		c[i].pos = Vector(positions[i][0], positions[i][1], positions[i][2]);
		c[i].norm = Vector(normals[i][0], normals[i][1], normals[i][2]);
	}
	return _AnalyzeContacts3D(c, frictioncoeff, numfrictionconeaxes);
}

void World::_delete_elements()
{
	// delete the elements created by new operator
	if ( !!_phand ) {
		delete _phand;
	}
	if ( !!_pobject ) {
		delete _pobject;
	}
}

void World::_xmlsetkeywords()
{
	//system body joint endeffector motor
	//object body
	//body geom geometry
	//joint connection
	//import
	//geom
	//geometry
	//connection
	//endeffector
	//motor breakaway
	//breakaway
	//hand body joint endeffector motor
	//simulation contact liftup externalforce externalmoment
	//contact
	//ground geom geometry
	//liftup
	//externalforce
	//externalmoment

	vector< vector< string > > keywords;
	vector<string> s;

	s.clear(); s.push_back("system"); s.push_back("body"); s.push_back("joint"); s.push_back("endeffector"); s.push_back("motor"); keywords.push_back(s);
	s.clear(); s.push_back("object"); s.push_back("body"); keywords.push_back(s); 
	s.clear(); s.push_back("body"); s.push_back("geom"); s.push_back("geometry"); keywords.push_back(s);
	s.clear(); s.push_back("joint"); s.push_back("connection"); keywords.push_back(s);
	s.clear(); s.push_back("import"); keywords.push_back(s);
	s.clear(); s.push_back("geom"); keywords.push_back(s);
	s.clear(); s.push_back("geometry"); keywords.push_back(s);
	s.clear(); s.push_back("connection"); keywords.push_back(s);
	s.clear(); s.push_back("endeffector"); keywords.push_back(s);
	s.clear(); s.push_back("motor"); s.push_back("breakaway"); keywords.push_back(s);
	s.clear(); s.push_back("breakaway"); keywords.push_back(s);
	s.clear(); s.push_back("hand"); s.push_back("body"); s.push_back("joint");	s.push_back("endeffector"); s.push_back("motor"); keywords.push_back(s);
	s.clear(); s.push_back("simulation"); s.push_back("contact"); s.push_back("liftup"); s.push_back("externalforce"); s.push_back("externalmoment"); keywords.push_back(s);
	s.clear(); s.push_back("contact"); keywords.push_back(s);
	s.clear(); s.push_back("ground"); s.push_back("geom"); s.push_back("geometry"); keywords.push_back(s);
	s.clear(); s.push_back("liftup"); keywords.push_back(s);
	s.clear(); s.push_back("externalforce"); keywords.push_back(s);
	s.clear(); s.push_back("externalmoment"); keywords.push_back(s);

	xmlLoadKeywords(keywords);
}

void World::_xmlparse(TiXmlNode *node)
{
	if ( !node ) return;
	TiXmlNode* child = 0;
	while( child = node->IterateChildren( child ) ) {
		if ( child->Type() == TiXmlNode::TINYXML_ELEMENT ) {
			string str = child->Value();
			transform(str.begin(), str.end(), str.begin(), ::tolower);
			if ( str == "hand" || str == "system" ) {
				if ( !_xmlparse_hand((TiXmlElement*)child) ) {
					cerr << "error:: failed in parsing hand system" << " (line " << child->Row() << ")" << endl;
					break;
				}
			}
			else if ( str == "object" ) {
				if ( !_xmlparse_object((TiXmlElement*)child) ) {
					cerr << "error:: failed in parsing object" << " (line " << child->Row() << ")" << endl;
					break;
				}
			}
			else if ( str == "ground" ) {
				if ( !_xmlparse_ground((TiXmlElement*)child) ) {
					cerr << "error:: failed in parsing ground" << " (line " << child->Row() << ")" << endl;
					break;
				}
			}
			else if ( str == "simulation" ) {
				if ( !_xmlparse_simulation((TiXmlElement*)child) ) {
					cerr << "error:: failed in parsing simulation" << " (line " << child->Row() << ")" << endl;
					break;
				}
			}
			else {
				_xmlparse(child);
			}
		}
	}
}

bool World::_xmlparse_ground(TiXmlElement *pelement)
{
	if ( !xmlParseBody(pelement, &_ground) ) {
		cerr << "error:: failed in parsing ground" << endl;
		return false;
	}
	return true;
}

bool World::_xmlparse_object(TiXmlElement *pelement)
{
	vector<xmlElementProperty> properties;
	xmlScanElementProperties(pelement, properties);

	bool bstatic = false;
	Vec3 translation(0,0,0);
	SO3 rotation;
	string name;
	
	for (size_t i=0; i<properties.size(); i++) {
		string key = properties[i].key;
		string val = properties[i].value;
		int row = properties[i].row;

		if ( key == "name" ) {
			name = val;
		}
		else if ( key == "type" ) {
			if ( val == "dynamic" ) {
				bstatic = false;
			} else if ( val == "static" ) {
				bstatic = true;
			} else {
				cerr << "error:: unknown type \"" << val << "\":: type must be (static|dynamic)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "translation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				translation = narr.get_Vec3();
			} else {
				cerr << "error:: translation = [" << val << "]:: numeric value size mismatch:: size must be 3::" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "rotation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 9 ) {
				rotation = narr.get_SO3();
			} else {
				cerr << "error:: rotation = [" << val << "]:: numeric value size mismatch:: size must be 9::" << " (line " << row << ")" << endl;
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
		else if ( key == "body" ) {
			_pobject = xmlParseObject(properties[i].pxmlelement);
			if ( !_pobject ) {
				cerr << "error:: failed in parsing body" << " (line " << row << ")" << endl;
				return false;
			}
		}
	}

	if ( !_pobject ) {
		cerr << "error:: body undefined." << endl;
		return false;
	}

	_pobject->setName(name); // overwrite name
	_pobject->setPose(SE3(rotation,translation));
	_pobject->setStatic(bstatic);

	return true;
}

bool World::_xmlparse_hand(TiXmlElement *pelement)
{
	vector<xmlElementProperty> properties;
	xmlScanElementProperties(pelement, properties);

	if ( !!_phand ) {
		cerr << "error:: hand has already defined!" << endl;
		return false;
	}
	_phand = new RobotHand;

	bool bstatic = false;
	string basebodyname;
	Vec3 translation(0,0,0);
	SO3 rotation;
	vector<gReal> q_init;

	for (size_t i=0; i<properties.size(); i++) {
		string key = properties[i].key;
		string val = properties[i].value;
		int row = properties[i].row;

		if ( key == "name" ) {
			_phand->setName(val);
		}
		else if ( key == "translation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				translation = narr.get_Vec3();
			} else {
				cerr << "error:: translation = [" << val << "]:: numeric value size mismatch:: size must be 3::" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "rotation" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 9 ) {
				rotation = narr.get_SO3();
			} else {
				cerr << "error:: rotation = [" << val << "]:: numeric value size mismatch:: size must be 9::" << " (line " << row << ")" << endl;
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
		else if ( key == "jointconfigdeg" ) {
			xmlNumberArray narr(val);
			q_init.resize(narr.size());
			for (size_t j=0; j<q_init.size(); j++) {
				q_init[j] = narr.x[j]*3.14159265/180.;
			}
		}
		else if ( key == "jointconfigrad" ) {
			xmlNumberArray narr(val);
			q_init = narr.x;
		}
		else if ( key == "body" ) {
			RigidBody *pbody = xmlParseBody(properties[i].pxmlelement);
			if ( !pbody ) {
				cerr << "error:: failed in parsing body" << " (line " << row << ")" << endl;
				return false;
			}
			_phand->_pbodies_new.push_back(pbody);
		}
		else if ( key == "joint" ) {
			GJoint *pjoint = xmlParseJoint(properties[i].pxmlelement, _phand->_pbodies_new);
			if ( !pjoint ) {
				cerr << "error:: failed in parsing joint" << " (line " << row << ")" << endl;
				return false;
			}
			_phand->_pjoints_new.push_back(pjoint);
		}
		else if ( key == "endeffector" ) {
			vector<xmlElementProperty> properties_ee;
			xmlScanElementProperties(properties[i].pxmlelement, properties_ee);
			for (size_t j=0; j<properties_ee.size(); j++) {
				string eekey = properties_ee[j].key;
				string eeval = properties_ee[j].value;
				int eerow = properties_ee[j].row;
				if ( eekey == "body" ) {
					for (size_t k=0; k<_phand->_pbodies_new.size(); k++) {
						if ( _phand->_pbodies_new[k]->getName() == eeval ) {
							_phand->_pbody_ee = _phand->_pbodies_new[k];
							break;
						}
					}
				}
				else if ( eekey == "translation" ) {
					xmlNumberArray narr(eeval);
					if ( narr.size() == 3 ) {
						_phand->_T_ee.SetPosition(narr.get_Vec3());
					} else {
						cerr << "error:: translation = [" << val << "]:: numeric value size mismatch:: size must be 3::" << " (line " << eerow << ")" << endl;
						return false;
					}
				}
				else if ( eekey == "rotation" ) {
					xmlNumberArray narr(eeval);
					if ( narr.size() == 9 ) {
						_phand->_T_ee.SetRotation(narr.get_SO3());
					} else {
						cerr << "error:: rotation = [" << val << "]:: numeric value size mismatch:: size must be 9::" << " (line " << eerow << ")" << endl;
						return false;
					}
				}
				else if ( eekey == "axisrotation" ) {
					xmlNumberArray narr(eeval);
					if ( narr.size() == 4 ) {
						Vec3 n(narr.x[0], narr.x[1], narr.x[2]); n.Normalize();
						gReal theta = narr.x[3] * 3.14159265 / 180.;
						_phand->_T_ee.SetRotation(_phand->_T_ee.GetRotation() * Exp(theta*n)); // accumulative rotation
					} else {
						cerr << "error:: axisrotation = [" << eeval << "]:: numeric value size mismatch:: size must be 4::" << " (line " << eerow << ")" << endl;
						return false;
					}
				}
				else if ( eekey == "quaternion" || eekey == "quat" ) {
					xmlNumberArray narr(eeval);
					if ( narr.size() == 4 ) {
						gReal quat[4]; quat[0] = narr.x[0]; quat[1] = narr.x[1]; quat[2] = narr.x[2]; quat[3] = narr.x[3];
						_phand->_T_ee.SetRotation(Quat(quat));
					} else {
						cerr << "error:: quaternion = [" << eeval << "]:: numeric value size mismatch:: size must be 4::" << " (line " << eerow << ")" << endl;
						return false;
					}
				}
			}
		}
		else if ( key == "motor" ) {
			string name;
			int closingdir=0, mafiltersize=0;
			double speed=0, maxspeed=0, stalltorque=0, breakawaytorque=0;
			vector<GCoordinate*> pjointcoords;
			vector<double> ratios, ratios2;
			bool bbreakaway = false, autolock = false;

			vector<xmlElementProperty> properties_motor;
			xmlScanElementProperties(properties[i].pxmlelement, properties_motor);
			for (size_t j=0; j<properties_motor.size(); j++) {
				string mkey = properties_motor[j].key;
				string mval = properties_motor[j].value;
				int mrow = properties_motor[j].row;
				if ( mkey == "name" ) {
					name = mval;
				}
				else if ( mkey == "closingdirection" || mkey == "closingdir" ) {
					xmlNumberArrayInt narr(mval);
					if ( narr.size() == 1 ) {
						closingdir = narr.x[0];
					} else {
						cerr << "error:: closingdirection = [" << val << "]:: numeric value size mismatch:: size must be 1::" << " (line " << mrow << ")" << endl;
						return false;
					}
				}
				else if ( mkey == "speed" ) {
					xmlNumberArray narr(mval);
					if ( narr.size() == 1 ) {
						speed = narr.x[0];
					} else {
						cerr << "error:: speed = [" << val << "]:: numeric value size mismatch:: size must be 1::" << " (line " << mrow << ")" << endl;
						return false;
					}
				}
				else if ( mkey == "maxspeed" ) {
					xmlNumberArray narr(mval);
					if ( narr.size() == 1 ) {
						maxspeed = narr.x[0];
					} else {
						cerr << "error:: maxspeed = [" << val << "]:: numeric value size mismatch:: size must be 1::" << " (line " << mrow << ")" << endl;
						return false;
					}
				}
				else if ( mkey == "stalltorque" || mkey == "maxtorque" ) {
					xmlNumberArray narr(mval);
					if ( narr.size() == 1 ) {
						stalltorque = narr.x[0];
					} else {
						cerr << "error:: " << mkey << " = [" << val << "]:: numeric value size mismatch:: size must be 1::" << " (line " << mrow << ")" << endl;
						return false;
					}
				}
				else if ( mkey == "autolock" ) {
					if ( mval == "true" ) {
						autolock = true;
					} else if ( mval == "false" ) {
						autolock = false;
					} else {
						cerr << "error:: unsupported value \"" << mval << "\":: value must be (true|false)" << " (line " << mrow << ")" << endl;
						return false;
					}
				}
				else if ( mkey == "mafiltersize" ) {
					xmlNumberArrayInt narr(mval);
					if ( narr.size() == 1 && narr.x[0] > 0 ) {
						mafiltersize = narr.x[0];
					} else {
						cerr << "error:: " << mkey << " = [" << mval << "]:: numeric value mismatch:: size must be 1 and value must be positive::" << " (line " << mrow << ")" << endl;
						return false;
					}
				}
				else if ( mkey == "joints" ) {
					pjointcoords.clear();
					xmlStringArray strarr(mval);
					for (size_t k=0; k<strarr.size(); k++) {
						GJoint *pjoint = NULL;
						for (size_t m=0; m<_phand->_pjoints_new.size(); m++) {
							if ( _phand->_pjoints_new[m]->getName() == strarr.x[k] ) {
								pjoint = _phand->_pjoints_new[m];
								break;
							}
						}
						if ( !pjoint ) {
							cerr << "error:: undefined joint = [" << strarr.x[k] << "]" << " (line " << mrow << ")" << endl;
							return false;
						}
						for (list<GCoordinate*>::iterator iter_pcoord = pjoint->pCoordinates.begin(); iter_pcoord != pjoint->pCoordinates.end(); iter_pcoord++) {
							pjointcoords.push_back(*iter_pcoord);
						}
					}
				}
				else if ( mkey == "ratios" ) {
					xmlNumberArray narr(mval);
					ratios.resize(narr.size());
					for (size_t k=0; k<ratios.size(); k++) {
						ratios[k] = narr.x[k];
					}
				}
				else if ( mkey == "breakaway" ) {
					bbreakaway = true;
					vector<xmlElementProperty> properties_breakaway;
					xmlScanElementProperties(properties_motor[j].pxmlelement, properties_breakaway);
					for (size_t k=0; k<properties_breakaway.size(); k++) {
						string bkey = properties_breakaway[k].key;
						string bval = properties_breakaway[k].value;
						int brow = properties_breakaway[k].row;
						if ( bkey == "breakawaytorque" || bkey == "switchtorque" ) {
							if ( bkey == "switchtorque" ) {
								cout << "warning:: the keyword 'switchtorque' has been deprecated! Use 'breakawaytorque' instead." << endl;
							}
							xmlNumberArray narr(bval);
							if ( narr.size() == 1 ) {
								breakawaytorque = narr.x[0];
							} else {
								cerr << "error:: breakawaytorque = [" << val << "]:: numeric value size mismatch:: size must be 1::" << " (line " << brow << ")" << endl;
								return false;
							}
						}
						else if ( bkey == "ratios" ) {
							xmlNumberArray narr(bval);
							ratios2.resize(narr.size());
							for (size_t k=0; k<ratios2.size(); k++) {
								ratios2[k] = narr.x[k];
							}
						}
					}
				}
			}

			Motor motor;
			if ( !motor.setMotor(name, maxspeed, stalltorque, pjointcoords, ratios, autolock) ) {
				cerr << "error:: failed in setting motor properties" << endl;
				return false;
			}
			if ( bbreakaway && !motor.setBreakAway(breakawaytorque, ratios2) ) {
				cerr << "error:: failed in setting breakaway for motor" << endl;
				return false;
			}
			if ( mafiltersize > 0 ) {
				motor.setTorqueFilterBufferSize(mafiltersize);
			}
			motor.setClosingDirection(closingdir);
			motor.setSpeed(speed);
			_phand->_motors.push_back(motor);
		}
	}

	if ( _phand->_pbodies_new.size() == 0 ) {
		cerr << "error:: no body defined in the system" << endl;
		return false;
	}

	// if end-effector body has not been set yet, set the first body
	if ( !_phand->_pbody_ee ) {
		_phand->_pbody_ee = _phand->_pbodies_new[0];
	}

	// root joint
	GJoint *prootjoint = new GJointFree;
	prootjoint->connectBodies(&_ground, _phand->_pbodies_new[0]);
	prootjoint->setPositionAndOrientation(SE3(rotation,translation), SE3());
	_phand->_pjoints_new.push_back(prootjoint);

	// build system
	if ( !_phand->buildSystem(&_ground) ) {
		cerr << "error:: failed in building hand system" << endl;
		return false;
	}

	// check body and joint
	for (size_t i=0; i<_phand->_pbodies_new.size(); i++) {
		if ( find(_phand->pBodies.begin(), _phand->pBodies.end(), _phand->_pbodies_new[i]) == _phand->pBodies.end() ) {
			cout << "warning:: body \"" << _phand->_pbodies_new[i]->getName() << "\" is not included in the hand system" << endl;
		}
	}
	for (size_t i=0; i<_phand->_pjoints_new.size(); i++) {
		if ( find(_phand->pJoints.begin(), _phand->pJoints.end(), _phand->_pjoints_new[i]) == _phand->pJoints.end() ) {
			cout << "warning:: joint \"" << _phand->_pjoints_new[i]->getName() << "\" is not included in the hand system" << endl;
		}
	}

	// set joint coordinates in the order of joint defined in xml file (root joint will not be included here)
	_phand->_pjointcoords.clear();
	for (size_t i=0; i<_phand->_pjoints_new.size(); i++) {
		if ( _phand->_pjoints_new[i] == prootjoint )
			continue;
		for (list<GCoordinate*>::iterator iter_pcoord = _phand->_pjoints_new[i]->pCoordinates.begin(); iter_pcoord != _phand->_pjoints_new[i]->pCoordinates.end(); iter_pcoord++) {
			_phand->_pjointcoords.push_back(*iter_pcoord);
		}
	}

	// set initial joint configuration 
	if ( q_init.size() == 0 ) {
		q_init = vector<double>(_phand->_pjointcoords.size(), 0);
	}
	if ( _phand->_pjointcoords.size() != q_init.size() ) {
		cerr << "error:: failed in setting initial joint configuration:: size mismatch:: number of joint coordinates = " << _phand->_pjointcoords.size() << ", number of input values (jointconfigdeg or jointconfigrad) = " << q_init.size() << endl;
		return false;
	}
	for (size_t i=0; i<_phand->_pjointcoords.size(); i++) {
		_phand->_pjointcoords[i]->q = q_init[i];
	}
	_phand->updateKinematics();

	return true;
}

bool World::_xmlparse_simulation(TiXmlElement *pelement)
{
	vector<xmlElementProperty> properties;
	xmlScanElementProperties(pelement, properties);

	for (size_t i=0; i<properties.size(); i++) {
		string key = properties[i].key;
		string val = properties[i].value;
		int row = properties[i].row;

		if ( key == "gravity" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 3 ) {
				setGravity(Vec3(narr.x[0], narr.x[1], narr.x[2]));
			} else {
				cerr << "error:: gravity = [" << val << "]:: numeric value size mismatch:: size must be 3::" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "stepsize" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 1 ) {
				setStepSize(narr.x[0]);
			} else {
				cerr << "error:: stepsize = [" << val << "]:: numeric value size mismatch:: size must be 1::" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "savefreq" ) {
			xmlNumberArrayInt narr(val);
			if ( narr.size() == 1 ) {
				_datasavefreq = narr.x[0];
			} else {
				cerr << "error:: datasavefreq = [" << val << "]:: numeric value size mismatch:: size must be 1::" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "savecontact" ) {
			if ( val == "true" ) {
				_datasavecontact = true;
			} else if ( val == "false" ) {
				_datasavecontact = false;
			} else {
				cerr << "error:: unsupported value \"" << val << "\":: value must be (true|false)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "ignorehandgroundcollision" ) {
			if ( val == "true" ) {
				_ignore_hand_ground_collision = true;
			} else if ( val == "false" ) {
				_ignore_hand_ground_collision = false;
			} else {
				cerr << "error:: unsupported value \"" << val << "\":: value must be (true|false)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "contact" ) {
			vector<xmlElementProperty> cproperties;
			xmlScanElementProperties(properties[i].pxmlelement, cproperties);

			enum ContactType { CT_NONE, CT_HAND_OBJECT, CT_HAND_GROUND, CT_OBJECT_GROUND };
			ContactType ct = CT_NONE;
			CollisionChecker::ContactParam cp;
			cp._k = _c/_stepsize;

			for (size_t j=0; j<cproperties.size(); j++) {
				string ckey = cproperties[j].key;
				string cval = cproperties[j].value;
				int crow = cproperties[j].row;

				if ( ckey == "type" ) {
					if ( cval == "hand-object" || cval == "hand_object" ) {
						ct = CT_HAND_OBJECT;
					}
					else if ( cval == "hand-ground" || cval == "hand_ground" ) {
						ct =  CT_HAND_GROUND;
					}
					else if ( cval == "object-ground" || cval == "object_ground" ) {
						ct = CT_OBJECT_GROUND;
					}
					else {
						cerr << "error:: type = [" << cval << "]:: unknown contact type:: type must be one of (hand-object, hand-ground, object-ground) :: " << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( ckey == "kp" ) {
					xmlNumberArray narr(cval);
					if ( narr.size() == 1 ) {
						cp._Kp = narr.x[0];
					} else {
						cerr << "error:: kp = [" << cval << "]:: numeric value size mismatch:: size must be 1::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( ckey == "kd" ) {
					xmlNumberArray narr(cval);
					if ( narr.size() == 1 ) {
						cp._Kd = narr.x[0];
					} else {
						cerr << "error:: kd = [" << cval << "]:: numeric value size mismatch:: size must be 1::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( ckey == "kfp" ) {
					xmlNumberArray narr(cval);
					if ( narr.size() == 1 ) {
						cp._Kfp = narr.x[0];
					} else {
						cerr << "error:: kfp = [" << cval << "]:: numeric value size mismatch:: size must be 1::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( ckey == "kfd" ) {
					xmlNumberArray narr(cval);
					if ( narr.size() == 1 ) {
						cp._Kfd = narr.x[0];
					} else {
						cerr << "error:: kfd = [" << cval << "]:: numeric value size mismatch:: size must be 1::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( ckey == "friction" ) {
					xmlNumberArray narr(cval);
					if ( narr.size() == 1 ) {
						cp._mu_s = cp._mu_d = narr.x[0];
					} else if ( narr.size() == 2 ) {
						cp._mu_s = narr.x[0]; cp._mu_d = narr.x[1];
					} else {
						cerr << "error:: friction = [" << cval << "]:: numeric value size mismatch:: size must be 1 or 2(static, dynamic)::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
			}

			switch (ct) {
			case CT_HAND_OBJECT:
				_cp_hand_object = cp;
				break;
			case CT_HAND_GROUND:
				_cp_hand_ground = cp;
				break;
			case CT_OBJECT_GROUND:
				_cp_object_ground = cp;
				break;
			case CT_NONE:
				_cp_hand_object = _cp_hand_ground = _cp_object_ground = cp;
				break;
			}
		}
		else if ( key == "liftup" ) {
			vector<xmlElementProperty> lproperties;
			xmlScanElementProperties(properties[i].pxmlelement, lproperties);

			for (size_t j=0; j<lproperties.size(); j++) {
				string lkey = lproperties[j].key;
				string lval = lproperties[j].value;
				int crow = lproperties[j].row;

				if ( lkey == "d" ) {
					xmlNumberArray narr(lval);
					if ( narr.size() == 1 ) {
						_liftup_d = narr.x[0];
					} else {
						cerr << "error:: d = [" << lval << "]:: numeric value size mismatch:: size must be 1::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( lkey == "a" ) {
					xmlNumberArray narr(lval);
					if ( narr.size() == 1 ) {
						_liftup_a = narr.x[0];
					} else {
						cerr << "error:: a = [" << lval << "]:: numeric value size mismatch:: size must be 1::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( lkey == "vm" ) {
					xmlNumberArray narr(lval);
					if ( narr.size() == 1 ) {
						_liftup_vm = narr.x[0];
					} else {
						cerr << "error:: vm = [" << lval << "]:: numeric value size mismatch:: size must be 1::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
			}
		}
		else if ( key == "applyexternalforce" ) {
			if ( val == "true" ) {
				_apply_external_force = true;
			} else if ( val == "false" ) {
				_apply_external_force = false;
			} else {
				cerr << "error:: unsupported value \"" << val << "\":: value must be (true|false)" << " (line " << row << ")" << endl;
				return false;
			}
		}
		else if ( key == "externalforce" || key == "externalmoment" ) {
			ForceResponseMeasureType gqmtype = FRM_ALL;
			ForceApplyType ftype = FA_LINEAR;
			double mag = 0, dur = 0;
			vector<Vec3> dirs;

			vector<xmlElementProperty> lproperties;
			xmlScanElementProperties(properties[i].pxmlelement, lproperties);
			for (size_t j=0; j<lproperties.size(); j++) {
				string lkey = lproperties[j].key;
				string lval = lproperties[j].value;
				int crow = lproperties[j].row;

				if ( lkey == "measure" ) {
					if ( lval == "deviation" ) {
						gqmtype = FRM_MAX_DEVIATION;
					}
					else if ( lval == "time" ) {
						gqmtype = FRM_RESISTANT_TIME;
					}
					else if ( lval == "all" ) {
						gqmtype = FRM_ALL;
					}
					else {
						cerr << "error:: " << lkey << " = [" << lval << "]:: unknown type::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( lkey == "type" || lkey == "forcetype" || lkey == "momenttype" ) {
					if ( lval == "constant" ) {
						ftype = FA_CONSTANT;
					}
					else if ( lval == "linear" ) {
						ftype = FA_LINEAR;
					}
					else {
						cerr << "error:: " << lkey << " = [" << lval << "]:: unknown type::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( lkey == "magnitude" ) {
					xmlNumberArray narr(lval);
					if ( narr.size() == 1 ) {
						mag = narr.x[0];
					} else {
						cerr << "error:: " << lkey << " = [" << lval << "]:: numeric value size mismatch:: size must be 1::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( lkey == "duration" ) {
					xmlNumberArray narr(lval);
					if ( narr.size() == 1 ) {
						dur = narr.x[0];
					} else {
						cerr << "error:: " << lkey << " = [" << lval << "]:: numeric value size mismatch:: size must be 1::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
				else if ( lkey == "direction" ) {
					xmlNumberArray narr(lval);
					if ( narr.size() > 0 && narr.size() % 3 == 0 ) {
						int n = narr.size() / 3;
						for (int i=0; i<n; i++) {
							dirs.push_back(Vec3(narr.x[3*i], narr.x[3*i+1], narr.x[3*i+2]));
						}
					} else {
						cerr << "error:: " << lkey << " = [" << lval << "]:: numeric value size mismatch:: size must be positive and multiple of 3::" << " (line " << crow << ")" << endl;
						return false;
					}
				}
			}

			if ( key == "externalforce" ) {
				for (size_t j=0; j<dirs.size(); j++) {
					_gq_external_forces.push_back(ExternalForce(gqmtype, ftype, dirs[j], mag, dur));
				}
			} 
			else if ( key == "externalmoment" ) {
				for (size_t j=0; j<dirs.size(); j++) {
					_gq_external_moments.push_back(ExternalMoment(gqmtype, ftype, dirs[j], mag, dur));
				}
			}
		}
		else if ( key == "posedeviationlimit" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 2 ) {
				_deviation_limit_pos = narr.x[0];
				_deviation_limit_ori = (3.14159/180.)*narr.x[1];
				_deviation_limit_pos_externalforce = _deviation_limit_pos;
				_deviation_limit_ori_externalmoment = _deviation_limit_ori;
			}
			else if ( narr.size() == 4 ) {
				_deviation_limit_pos = narr.x[0];
				_deviation_limit_ori = (3.14159/180.)*narr.x[1];
				_deviation_limit_pos_externalforce = narr.x[2];
				_deviation_limit_ori_externalmoment = (3.14159/180.)*narr.x[3];
			}
			else {
				cerr << "error:: posedeviationlimit = [" << val << "]:: numeric value size mismatch:: size must be 2 or 4 (pos ori [pos ori])::" << " (line " << row << ")" << endl;
				cout << "        pos = position deviation limit in liftup and applying external force(unit: m)" << endl;
				cout << "        ori = orientation deviation limit in liftup and applying external moment (unit: deg)" << endl;
				return false;
			}
		}
		else if ( key == "objectposeuncertainty" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 8 ) {
				setObjectPoseUncertainty((int)narr.x[0], Vec3(narr.x[1],narr.x[2],narr.x[3]), (3.14159/180.)*narr.x[4], Vec3(narr.x[5],narr.x[6],narr.x[7]));
			} else {
				cerr << "error:: objectposeuncertainty = [" << val << "]:: numeric value size mismatch:: size must be 8 (n px py pz ang wx wy wz)::" << " (line " << row << ")" << endl;
				cout << "        n = sampling size" << endl;
				cout << "        p = standard deviation for position in meter" << endl;
				cout << "        ang = standard deviation for rotation angle in degree" << endl;
				cout << "        w = rotation axis (e.g., [0 0 1] for z-axis rotation, [1,1,1] for random axis)" << endl;
				return false;
			}
		}
		else if ( key == "objectposeuncertainty_halfnormalmean" ) {
			xmlNumberArray narr(val);
			if ( narr.size() == 8 ) {
				setObjectPoseUncertaintyWithHalfNormalMean((int)narr.x[0], Vec3(narr.x[1],narr.x[2],narr.x[3]), (3.14159/180.)*narr.x[4], Vec3(narr.x[5],narr.x[6],narr.x[7]));
			} else {
				cerr << "error:: objectposeuncertainty_halfnormalmean = [" << val << "]:: numeric value size mismatch:: size must be 8 (n px py pz ang wx wy wz)::" << " (line " << row << ")" << endl;
				cout << "        n = sampling size" << endl;
				cout << "        p = half normal distribution mean for position in meter" << endl;
				cout << "        ang = half normal distribution mean for rotation angle in degree" << endl;
				cout << "        w = rotation axis (e.g., [0 0 1] for z-axis rotation, [1,1,1] for random axis)" << endl;
				return false;
			}
		}
		else if ( key == "targetconfig" ) {
			if ( !_phand ) {
				cerr << "error:: hand system not defined yet!" << endl;
				return false;
			}
			xmlNumberArray narr(val);
			if ( narr.size() == _phand->_motors.size() ) {
				for (size_t j=0; j<_phand->_motors.size(); j++) {
					if ( _phand->_motors[j]._ratios.size() > 0 && fabs(_phand->_motors[j]._ratios[0]) > 1E-8 ) {
						_phand->_motors[j].setTargetPosition(narr.x[j] / _phand->_motors[j]._ratios[0]);
					}
				}
			} else {
				cerr << "error:: targetconfig = [" << val << "]:: numeric value size mismatch:: size must be the number of motors::" << " (line " << row << ")" << endl;
			}
		}
		else {
			cout << "warning:: unknown keyword: " << key << " (line " << row << ")" << endl;
		}
	}

	return true;
}

void World::_estimateBoundingSphere()
{
	vector<Vec3> nodes;
	if ( !!_phand ) {
		for (list<GBody*>::iterator iter_pbody = _phand->pBodies.begin(); iter_pbody != _phand->pBodies.end(); iter_pbody++) {
			nodes.push_back((*iter_pbody)->getPositionCOMGlobal());
		}
		for (list<GJoint*>::iterator iter_pjoint = _phand->pJoints.begin(); iter_pjoint != _phand->pJoints.end(); iter_pjoint++) {
			nodes.push_back((*iter_pjoint)->T_global.GetPosition());
		}
	}
	if ( !!_pobject ) {
		nodes.push_back(_pobject->getPosition());
	}

	if ( nodes.size()==0 ) {
		_bs_radius = 0;
		_bs_center.SetZero();
		return;
	}
	if ( nodes.size()==1 ) {
		_bs_radius = 0;
		_bs_center = nodes[0];
		return;
	}

	// _bs_center = average position of the nodes
	_bs_center.SetZero();
	for (size_t i=0; i<nodes.size(); i++) {
		_bs_center += nodes[i];
	}
	_bs_center *= gReal(1./(double)nodes.size());

	// _bs_radius = half of the maximum distance between nodes
	_bs_radius = 0;
	for (size_t i=0; i<nodes.size(); i++) {
		for (size_t j=i+1; j<nodes.size(); j++) {
			gReal dist = Norm(nodes[i]-nodes[j]);
			if ( dist > _bs_radius ) {
				_bs_radius = dist;
			}
		}
	}
	_bs_radius *= 0.5;
}

void World::_setSimulationSave()
{
#if GEAR_DOUBLE_PRECISION
	std::vector<double *> ptr_data_real;
#else
	std::vector<float *> ptr_data_real;
#endif
	std::vector<int *> ptr_data_int;

	// save hand
	if ( !!_phand ) {
		// save root joint positioning
		for (int i=0; i<16; i++) {
			ptr_data_real.push_back(&(_phand->getBase()->pBaseJoint->T_left[i]));
			ptr_data_real.push_back(&(_phand->getBase()->pBaseJoint->T_right[i]));
		}

		// save system coordinates
		for (std::list<GCoordinate*>::iterator iter_pcoord = _phand->pCoordinates.begin(); iter_pcoord != _phand->pCoordinates.end(); iter_pcoord++) {
			ptr_data_real.push_back(&((*iter_pcoord)->q));
			ptr_data_real.push_back(&((*iter_pcoord)->dq));
			ptr_data_real.push_back(&((*iter_pcoord)->ddq));
			ptr_data_real.push_back(&((*iter_pcoord)->tau));
		}

		// save coordinate chart for spherical joints
		for (std::list<GJoint*>::iterator iter_pjoint = _phand->pJoints.begin(); iter_pjoint != _phand->pJoints.end(); iter_pjoint++) {
			if ( (*iter_pjoint)->jointType == GJoint::GJOINT_SPHERICAL ) {
				ptr_data_int.push_back(&(((GJointSpherical*)(*iter_pjoint))->coord_chart));
			}
			if ( (*iter_pjoint)->jointType == GJoint::GJOINT_FREE_ST ) {
				ptr_data_int.push_back(&(((GJointFreeST*)(*iter_pjoint))->spherical_joint.coord_chart));
			}
			if ( (*iter_pjoint)->jointType == GJoint::GJOINT_FREE_TS ) {
				ptr_data_int.push_back(&(((GJointFreeTS*)(*iter_pjoint))->spherical_joint.coord_chart));
			}
		}

		// save external force of each link
		for (size_t i=0; i<_phand->_pbodies_new.size(); i++) {
			for (int j=0; j<6; j++) {
				ptr_data_real.push_back(&(_phand->_pbodies_new[i]->Fe[j]));
			}
		}

		// save motor information 
		for (size_t i=0; i<_phand->_motors.size(); i++) {
			for (size_t j=0; j<_phand->_motors[i]._data_debugging.size(); j++) {
				ptr_data_real.push_back(&(_phand->_motors[i]._data_debugging[j]));
			}
		}

	}

	// save objects
	if ( !!_pobject ) {
		for (int j=0; j<16; j++) {
			ptr_data_real.push_back(&(_pobject->_T[j]));
		}
		for (int j=0; j<6; j++) {
			ptr_data_real.push_back(&(_pobject->_V[j]));
			ptr_data_real.push_back(&(_pobject->_F[j]));
		}
	}

	vector<RigidSurface*> psurfs = _colchecker.getSurfaces();

	// save bounding boxes of the surfaces
	for (size_t i=0; i<psurfs.size(); i++) {
		for (int j=0; j<3; j++) {
			ptr_data_real.push_back(&((psurfs[i]->getBoundingBox().position[j])));
		}
		for (int j=0; j<9; j++) {
			ptr_data_real.push_back(&((psurfs[i]->getBoundingBox().orientation[j])));
		}
	}

	// save external force
	for (int i=0; i<3; i++) {
		ptr_data_real.push_back(&(_external_force[i]));
		ptr_data_real.push_back(&(_external_moment[i]));
	}

	// save contacts
	if ( _datasavecontact ) {
		for (size_t i=0; i<psurfs.size(); i++) {
			_simuldata.add_varying_data_int(&(psurfs[i]->_bContact), &(psurfs[i]->_collidingVertexIndices));
			_simuldata.add_varying_data_int(&(psurfs[i]->_bStaticFriction), &(psurfs[i]->_collidingVertexIndices));
			_simuldata.add_varying_data_Vec3(&(psurfs[i]->_xf_ref), &(psurfs[i]->_collidingVertexIndices));
			_simuldata.add_varying_data_Vec3(&(psurfs[i]->_fc), &(psurfs[i]->_collidingVertexIndices));
			_simuldata.add_varying_data_int(&(psurfs[i]->_bColSurfPatch), &(psurfs[i]->_collidingSurfPatchIndices));
			_simuldata.add_varying_data_int(&(psurfs[i]->_bColSeedVertex), &(psurfs[i]->_seedVertexIndices));
		}
	}

	// save grasp contact center
	for (int i=0; i<3; i++) {
		ptr_data_real.push_back(&(_gcc_obj[i]));
	}
	ptr_data_int.push_back(&_b_set_gcc_obj);

	// clear and set data address
	_simuldata.clear_data_address_all();
#if GEAR_DOUBLE_PRECISION
	_simuldata.set_data_address_double(ptr_data_real);
#else
	_simuldata.set_data_address_float(ptr_data_real);
#endif
	_simuldata.set_data_address_int(ptr_data_int);
}

void World::_scanCollisionSurfacePairs()
{
	_colchecker.clearCollisionSurfacePairs();

	// collision surface pairs (hand <--> object)
	if ( !!_phand && !!_pobject && _pobject->isEnabledCollision() ) {
		_idx_surf_pairs_by_links.resize(_phand->_pbodies_new.size());
		for (size_t j=0; j<_phand->_pbodies_new.size(); j++) {
			if ( !_phand->_pbodies_new[j]->isEnabledCollision() )
				continue;
			for (size_t k=0; k<_phand->_pbodies_new[j]->pSurfs.size(); k++) {
				if ( !_phand->_pbodies_new[j]->pSurfs[k]->isEnabledCollision() )
					continue;
				for (size_t m=0; m<_pobject->_pSurfs.size(); m++) {
					if ( !_pobject->_pSurfs[m]->isEnabledCollision() )
						continue;
					_colchecker.addCollisionSurfacePair(_phand->_pbodies_new[j]->pSurfs[k], _pobject->_pSurfs[m], _cp_hand_object, false, true, true);
					_idx_obj_hand_surf_pairs.push_back(_colchecker.getCollisionSurfacePairs().size()-1);
					_idx_surf_pairs_by_links[j].push_back(_colchecker.getCollisionSurfacePairs().size()-1);
				}
			}
		}
	}

	// collision surface pairs (hand <--> ground)
	if ( !!_phand && _ground.isEnabledCollision() && !_ignore_hand_ground_collision ) {
		for (size_t j=0; j<_phand->_pbodies_new.size(); j++) {
			if ( !_phand->_pbodies_new[j]->isEnabledCollision() )
				continue;
			for (size_t k=0; k<_phand->_pbodies_new[j]->pSurfs.size(); k++) {
				if ( !_phand->_pbodies_new[j]->pSurfs[k]->isEnabledCollision() )
					continue;
				for (size_t m=0; m<_ground.pSurfs.size(); m++) {
					if ( !_ground.pSurfs[m]->isEnabledCollision() )
						continue;
					_colchecker.addCollisionSurfacePair(_phand->_pbodies_new[j]->pSurfs[k], _ground.pSurfs[m], _cp_hand_ground, false, true, false); 
					_idx_hand_ground_surf_pairs.push_back(_colchecker.getCollisionSurfacePairs().size()-1);
				}
			}
		}
	}

	// collision surface pairs (object <--> ground)
	if ( !!_pobject && _pobject->isEnabledCollision() ) {
		for (size_t j=0; j<_pobject->_pSurfs.size(); j++) {
			if ( !_pobject->_pSurfs[j]->isEnabledCollision() ) 
				continue;
			for (size_t l=0; l<_ground.pSurfs.size(); l++) {
				if ( !_ground.pSurfs[l]->isEnabledCollision() ) 
					continue;
				_colchecker.addCollisionSurfacePair(_pobject->_pSurfs[j], _ground.pSurfs[l], _cp_object_ground, false, true, false);
				_idx_obj_ground_surf_pairs.push_back(_colchecker.getCollisionSurfacePairs().size()-1);
			}
		}
	}

	// contact force rendering 
	if ( !!_phand ) { 
		for (size_t j=0; j<_phand->_pbodies_new.size(); j++) {
			for (size_t k=0; k<_phand->_pbodies_new[j]->pSurfs.size(); k++) {
				_phand->_pbodies_new[j]->pSurfs[k]->_bRenderingContactForcesReversed = true; // render contact forces by the hand (i.e., acting on the object)
			}
		}
	}
	if ( !!_pobject ) {
		for (size_t j=0; j<_pobject->_pSurfs.size(); j++) {
			_pobject->_pSurfs[j]->_bRenderingContactForcesReversed = false; // render contact forces acting on the object
		}
	}
}

void World::_saveRenderingOptions()
{
	_surf_draw_types.clear();
	_surf_brender.clear();
	for (list<GBody*>::iterator iter_pbody = _phand->pBodies.begin(); iter_pbody != _phand->pBodies.end(); iter_pbody++) {
		for (size_t i=0; i<((RigidBody*)(*iter_pbody))->getSurfaces().size(); i++) {
			_surf_draw_types.push_back(((RigidBody*)(*iter_pbody))->getSurface(i)->getDrawType());
			_surf_brender.push_back(((RigidBody*)(*iter_pbody))->getSurface(i)->isEnabledRendering());
		}
	}
	if ( !!_pobject ) {
		for (size_t i=0; i<_pobject->getSurfaces().size(); i++) {
			_surf_draw_types.push_back(_pobject->getSurface(i)->getDrawType());
			_surf_brender.push_back(_pobject->getSurface(i)->isEnabledRendering());
		}
	}
	for (size_t i=0; i<_ground.getSurfaces().size(); i++) {
		_surf_draw_types.push_back(_ground.getSurface(i)->getDrawType());
		_surf_brender.push_back(_ground.getSurface(i)->isEnabledRendering());
	}
}



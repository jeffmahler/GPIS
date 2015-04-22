#ifndef _WORLD_
#define _WORLD_

#include <vector>
#include <string>
#include "gear.h"
#include "robothand.h"
#include "rigidbody.h"
#include "rigidobject.h"
#include "simuldata.h"
#include "colchecker.h"
#include "tinyxml.h"
#include "graspset.h"
#include "poseuncertainty.h"
#include "externalforce.h"
#include "forceclosure.h"
#include "fingerprint.h"

class World//: public GElement
{
public:
	enum SimulPhase { SS_START, SS_REST, SS_CLOSE_FINGER, SS_LIFTUP, SS_NO_APPLY_FORCE, SS_APPLY_FORCE, SS_DONE };
	//enum ForceApplyType { FA_FORCE_X, FA_FORCE_Y, FA_FORCE_Z, FA_MOMENT_X, FA_MOMENT_Y, FA_MOMENT_Z };
	//enum ForceApplyDir { FA_POSITIVE, FA_NEGATIVE };

	// elements in the world
	RobotHand *_phand; 
	RigidObject *_pobject;

	// ground
	RigidBody _ground;

	// gravity
	Vec3 _gravity;

	// simulation step size
	double _stepsize;

	// simulation phase
	SimulPhase _simul_phase; 

	// collision checker
	CollisionChecker _colchecker;
	bool _ignore_hand_ground_collision; // set true to ignore hand-ground collision

	// contact parameters
	CollisionChecker::ContactParam _cp_hand_object, _cp_hand_ground, _cp_object_ground;
	double _c; // for setting CollisionChecker::ContactParam::_k automatically (_k = _c/_stepsize)

	// simulation data save
	SimulData _simuldata;
	int _datasavefreq;
	bool _datasavecontact;

	// for saving/restoring simulation state
	std::vector<double> _state_double; 
	std::vector<int> _state_int;

	// bounding sphere for view sizing
	double _bs_radius;
	Vec3 _bs_center;

	// object pose uncertainty for grasp quality evaluation 
	PoseUncertainty _obj_pose_uncertainty;

	// liftup trajectory
	double _liftup_d, _liftup_a, _liftup_vm; // liftup distance, acceleration, and max velocity

	// set of external forces and moments for grasp quality test
	bool _apply_external_force;
	std::vector<ExternalForce> _gq_external_forces;
	std::vector<ExternalMoment> _gq_external_moments;

	// pose deviation limits for computing grasp quality (unit: meter, radian)
	double _deviation_limit_pos, _deviation_limit_ori;
	double _deviation_limit_pos_externalforce, _deviation_limit_ori_externalmoment;

	// simulation result for computing grasp quality
	int _num_contact_links_after_liftup;	// number of contact links after liftup
	double _min_dist_after_liftup;		// minimum distance from the origin to the convex hull faces of the contact wrenches
	Vec3 _gcc_obj;						// grasp contact center (after grasping and before liftup) w.r.t. {object}
	SE3 _T_h2o_before_grasping;			// object pose before grasping (relative to the hand's end-effector frame)
	SE3 _T_h2o_before_liftup;			// object pose before liftup (relative to the hand's end-effector frame)
	SE3 _T_g2o_before_applying_force;	// object pose before applying force (w.r.t. {global})
	double _deviation_pos_final, _deviation_ori_final; // final pose deviation from initial pose
	double _deviation_pos_max_grasping, _deviation_ori_max_grasping; // max pose deviation during grasping
	double _deviation_pos_max_liftup, _deviation_ori_max_liftup;	// max pose deviation during liftup
	std::vector<double> _deviation_pos_max_externalforce, _deviation_ori_max_externalmoment;	// max pose deviations by external forces/moments
	std::vector<double> _time_resistant_externalforce, _time_resistant_externalmoment;	// times resistant to external forces/moments

	// finger print
	FingerPrintHistory _fingerprinthistory;
	//std::vector< std::vector<int> > _face_hit_cnt_obj_surfs;	// _face_hit_cnt_obj_surfs[i][j] = number of contact hits at the j-th face of the i-th surface of the object
	//int _hit_cnt_ref;											// reference value for the hit count

	// internal variables
	std::vector<int> _idx_obj_hand_surf_pairs, _idx_hand_ground_surf_pairs, _idx_obj_ground_surf_pairs; // surface pair indices in _colchecker.getCollisionSurfacePairs()
	std::vector< std::vector<int> > _idx_surf_pairs_by_links; // indices of the surface pairs
	std::vector<SurfaceMesh::DrawType> _surf_draw_types;
	std::vector<bool> _surf_brender;
	Vec3 _external_force, _external_moment;	// current external force and moment applied to the object's center of mass
	int _num_score_types; // number of score types
	double _force_scale, _moment_scale; // scale for rendering external force and moment
	int _b_set_gcc_obj; // show if gcc has been set or not
	bool _b_show_gcc_obj;
	bool _b_show_coord_frames; // show the coordinate frames for the links, object, and the inertial frame

public:
	World() : _phand(NULL), _gravity(0,0,-9.81), _simul_phase(SS_START), _stepsize(0.001), _ignore_hand_ground_collision(false), _c(0.1), _datasavefreq(120), _datasavecontact(false), _bs_radius(0.5), _bs_center(0,0,0)
		, _liftup_d(0.1), _liftup_a(0.5), _liftup_vm(0.5)
		, _gcc_obj(0,0,0)
		, _deviation_limit_pos(0.05), _deviation_limit_ori(30./180.*3.14159)
		, _deviation_limit_pos_externalforce(0.01), _deviation_limit_ori_externalmoment(10./180.*3.14159)
		, _deviation_pos_final(0), _deviation_ori_final(0)
		, _deviation_pos_max_grasping(0), _deviation_ori_max_grasping(0)
		, _deviation_pos_max_liftup(0), _deviation_ori_max_liftup(0), _num_contact_links_after_liftup(0) 
		, _num_score_types(8), _force_scale(0.002), _moment_scale(0.1), _b_set_gcc_obj(0), _b_show_gcc_obj(false)
		, _apply_external_force(true)
		, _b_show_coord_frames(false)
	{ 
			_cp_hand_object._k = _c/_stepsize; _cp_hand_ground._k = _c/_stepsize; _cp_object_ground._k = _c/_stepsize; 
	}
	~World() { _delete_elements(); }

	bool loadFromXML(const char *filepath);

	void setGravity(Vec3 g) { _gravity = g; }
	Vec3& getGravity() { return _gravity; }
	
	void setStepSize(double h) { _stepsize = h; _cp_hand_object._k = _c/_stepsize; _cp_hand_ground._k = _c/_stepsize; _cp_object_ground._k = _c/_stepsize; }
	double getStepSize() { return _stepsize; }

	bool getReady();

	bool stepSimulation(double t); // proceed the simulation at time t with the time step _stepsize

	bool isSimulationDone() { return (_simul_phase == SS_DONE); }	// return true if simulation is done

	bool isSimulationDiverged() { return ( SquareSum(_pobject->_V) > 1E6 ); }

	void render();

	void initForces();
	void updateKinematics() { if ( !!_phand ) { _phand->updateKinematics(); } }

	void enableRenderingBoundingBoxes(bool b);
	void enableRenderingContactPoints(bool b); 
	void enableRenderingContactForces(bool b); 
	void enableRenderingCollidingSurfacePatches(bool b);

	void setRenderingMode(int mode); // 0(original), 1(wireframe only), 2(show hidden geometries)

	double getBoundingSphereRadius() { return _bs_radius; }
	Vec3 &getBoundingSphereCenter() { return _bs_center; }

	RigidObject* getObject() { return _pobject; }
	RobotHand* getHand() { return _phand; }

	// ------- grasp quality measurement using physics simulation ----------

	void initSimulation();
	void setLiftUpMotion(double d, double a, double vm) { _liftup_d = d; _liftup_a = a; _liftup_vm = vm; }
	void setObjectPoseUncertainty(int sample_size, Vec3 sigma_pos, double sigma_angle, Vec3 axis, SO3 R_principal_axes_pos=SO3(), SO3 R_principal_axes_ori=SO3());
	void setObjectPoseUncertaintyWithHalfNormalMean(int sample_size, Vec3 half_normal_mean_pos, double half_normal_mean_angle, Vec3 axis, SO3 R_principal_axes_pos=SO3(), SO3 R_principal_axes_ori=SO3());
	bool measureGraspQuality(std::vector<double> &quality_scores, const GraspSet::Grasp &grasp, bool bsavesimul=false, const char *folderpath=NULL);
	bool measureGraspQualityWithoutUncertainty(const GraspSet &gs, const char *folderpath=NULL);
	bool runSimulation(double tf, const char *filepath=NULL);	// tf = simulation time limit, filepath = file path for saving simulation result
	void scoreGraspQuality(std::vector<double> &qscores);
	int getNumOfContactLinks();
	bool isObjectMoving(double &v, double &w, double eps_v=0.001, double eps_w=0.1);
	bool isHandCollidingWithGround();
	bool isHandCollidingWithGround_BoundingBoxCheckOnly();
	void saveState(const char *filepath = NULL);		// save the current simulation state to memory (and file, optionally)
	void restoreState(const char *filepath = NULL);		// restore the saved simulation state from memory (or file, optionally)
	void computeGraspQualityErrorBar(int num_seed, int max_sample_size, const GraspSet::Grasp &grasp, const char *filepath);
	Vec3 getGraspContactCenter(); // get grasp contact center w.r.t. {object}

	// ------- force-closure analysis using current contact information (call _colchecker.checkCollision() if needed)
	double ComputeMinDist(); // compute minimum distance from the origin to the convex hull faces of the contact wrenches
	GRASPANALYSIS AnalyzeContacts();

	// ------- other functions ----------------------------------------------
	bool calcFingerPrintHistory(const GraspSet &gs, GraspSet &gs_finalconfig);	// calculate accumulated fingerprint history on the object by running grasping simulation for each grasp given in gs, and also save the final grasp configuration to gs_finalconfig
	bool checkFingerPrintHistory();						// check fingerprint history

	// ------- sub-functions -----------------------------------------------------------------

	void _delete_elements();

	void _xmlsetkeywords();
	void _xmlparse(TiXmlNode *pnode);
	bool _xmlparse_object(TiXmlElement *pelement);
	bool _xmlparse_hand(TiXmlElement *pelement);
	bool _xmlparse_ground(TiXmlElement *pelement);
	bool _xmlparse_simulation(TiXmlElement *pelement);

	void _estimateBoundingSphere();

	void _setSimulationSave();

	void _scanCollisionSurfacePairs(); // scan collision surface pairs (systems <--> objects, objects <--> objects)

	void _saveRenderingOptions();

	GRASPANALYSIS _AnalyzeContacts(std::vector<Vec3> positions, std::vector<Vec3> normals, double frictioncoeff, int numfrictionconeaxes);

	void _init_fingerprint(FingerPrint &fp); // init fingerprint memory for object
};

#endif


% computes all grasps for apc objects
config = struct();

% hardcoded pr2 params
pr2_grip_width_m = 0.1;
pr2_grip_offset = [-0.0375, 0, 0]'; % offset from points at which pr2 contacts on closing...

% io dirs
config.root_dir = 'data/apc';
config.out_dir = 'results/apc';
config.sdf_filename = 'optimized_poisson_texture_mapped_mesh_clean_25.sdf';
config.obj_filename = 'optimized_poisson_texture_mapped_mesh_clean.obj';

% bandit pose params
config.num_perturbations = 100;
config.num_random_grasps = 100;
config.sigma_trans = 0.02; % in meters grid cells
config.sigma_rot = 0.25;
config.pose_sampling_2d = false;
config.vis_pose_perturb = false;

% visualization
config.arrow_length = 3;
config.arrow_length_3d = 0.05;
config.step_size = 1;
config.plot_grasps = 1;
config.scale = 10;
config.plate_width = 2;
config.eps = 0;
config.plot_all_grasps_2d = false;
config.plot_all_grasps_3d = false;
config.vis_best_grasps = true;
config.vis_sdf = false;
config.vis_antipodal = false;

% candidate grasp computation
config.num_samples = 2;
config.pr2_grasp_width_m = pr2_grip_width_m;
config.friction_coef = 0.5;
config.n_cone_faces = 2;
config.dir_prior = 1.0;
config.alpha_thresh = pi / 8;
config.rho_scale = 0.9;

% random params for candidate grasps
config.theta_res = 2 * pi / 20;
config.surf_thresh = 0.004;
config.grasp_offset = pr2_grip_offset;
config.constrain_2d = true;

% bandit params
config.use_uniform_space_part = false;
config.num_candidate_grasps = 1;
config.num_bins = 20;
config.epsilon = 1e-1;
config.max_iters = 10000;

%results = compute_apc_grasps(config);
rng(200);
object_name = 'mommys_helper_outlet_plugs';
results = compute_mesh_grasps(object_name, config);

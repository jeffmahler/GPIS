% script to run hyperparam tuning

dim = 25;
dataDir = 'data/google_objects';
shapeNames = {'loofa', 'marker', 'squirt_bottle', 'stapler', 'tape', 'water'};
outputDir = 'results/google_objects/test';
newShape = false;
scale = 2;

%% experiment config
experimentConfig = struct();
experimentConfig.numGrasps = 5;
experimentConfig.frictionCoef = 0.5;

experimentConfig.min_mcir = 2;
experimentConfig.max_mcir = 6;
experimentConfig.inc_mcir = 2;

experimentConfig.min_ipc = 0.5;
experimentConfig.max_ipc = 0.5;
experimentConfig.inc_ipc = 1;

experimentConfig.min_itbs = 5;
experimentConfig.max_itbs = 5;
experimentConfig.inc_itbs = 1;

experimentConfig.min_tsr = 0.2;
experimentConfig.max_tsr = 0.8;
experimentConfig.inc_tsr = 0.2;

experimentConfig.min_ter = 1.25;
experimentConfig.max_ter = 3.25;
experimentConfig.inc_ter = 1;

experimentConfig.min_nu = 0.025;
experimentConfig.max_nu = 0.325;
experimentConfig.inc_nu = 0.1;


%% optimization parameters
cf = cos(atan(experimentConfig.frictionCoef));

cfg = struct();
cfg.max_iter = 10;
cfg.max_penalty_iter = 5;
cfg.max_merit_coeff_increases = 5;
cfg.merit_coeff_increase_ratio = 2;
cfg.initial_penalty_coeff = 0.5;
cfg.initial_trust_box_size = 5;
cfg.trust_shrink_ratio = .75;
cfg.trust_expand_ratio = 2.0;
cfg.min_approx_improve = 0.1;
cfg.min_trust_box_size = 0.1;
cfg.callback = @plot_surface_grasp_points;
cfg.full_hessian = true;
cfg.cnt_tolerance = 1e-4;
cfg.ineq_tolerance = [1-cf; 1-cf; 0.1];
cfg.eq_tolerance = [1e-1; 1e-1; 5e-2; 5e-2; 0.1];
cfg.com_tol = 2.0;
cfg.scale = scale;
cfg.min_init_dist = 8;
cfg.lambda = 0.00;
cfg.nu = 0.1;
cfg.fric_coef = 0; % use no-slip constraint to force antipodality but allow solutions within the friction cone


%%

hyperResults = run_hyperparam_search(shapeNames, dataDir, experimentConfig, cfg);
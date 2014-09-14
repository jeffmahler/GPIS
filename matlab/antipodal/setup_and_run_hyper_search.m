% script to run hyperparam tuning

dim = 25;
dataDir = 'data/google_objects/google_update_versions';
shapeNames = {'alka_seltzer', 'deodorant', 'oatmeal'};%, 'stapler', 'tape', 'water'};
gripScales = {1.2, 1.0, 0.8};
outputDir = 'results/hyper_tuning';
newShape = false;
scale = 2;

%% experiment config
experimentConfig = struct();
experimentConfig.numGrasps = 10;
experimentConfig.frictionCoef = 0.5;
experimentConfig.surfaceThresh = 0.15;
experimentConfig.arrowLength = 4;
experimentConfig.gripWidth = 20;
experimentConfig.graspSigma = 0.25;
experimentConfig.numBadContacts = 10;

experimentConfig.visSampling = false;
experimentConfig.visOptimization = true;
experimentConfig.visGrasps = true;
experimentConfig.rejectUnsuccessfulGrasps = false;

experimentConfig.gripWidth = dim; % dimension of grasp when scale = 1.0
experimentConfig.plateScale = 0.05;
experimentConfig.objScale = 0.95;
experimentConfig.qScale = 100;

experimentConfig.min_mcir = 1.1;
experimentConfig.max_mcir = 1.6;
experimentConfig.inc_mcir = 0.5;

experimentConfig.min_ipc = 0.5;
experimentConfig.max_ipc = 1.0;
experimentConfig.inc_ipc = 0.5;

experimentConfig.min_itbs = 1;
experimentConfig.max_itbs = 11;
experimentConfig.inc_itbs = 5;

experimentConfig.min_tsr = 0.4;
experimentConfig.max_tsr = 0.8;
experimentConfig.inc_tsr = 0.2;

experimentConfig.min_ter = 1.25;
experimentConfig.max_ter = 2.25;
experimentConfig.inc_ter = 0.5;

experimentConfig.min_nu = 0.5;
experimentConfig.max_nu = 2.0;
experimentConfig.scale_nu = 2.0;

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
cfg.min_approx_improve = 0.05;
cfg.min_trust_box_size = 0.1;
cfg.callback = @plot_surface_grasp_points;
cfg.full_hessian = true;
cfg.cnt_tolerance = 1e-4;
cfg.ineq_tolerance = [1-cf; 1-cf; 0.1; 0.1];
cfg.eq_tolerance = [1e-1; 1e-1; 5e-2; 5e-2; 0.1];
cfg.com_tol = 2.0;
cfg.scale = scale;
cfg.min_init_dist = 8;
cfg.lambda = 0.00;
cfg.nu = 1.0;
cfg.beta = 2.0;
cfg.fric_coef = 0; % use no-slip constraint to force antipodality but allow solutions within the friction cone
cfg.grip_width = experimentConfig.gripWidth; % max distance between grasp points in pixels

%% run the actual experiment
%rng(300);

hyperResults = run_hyperparam_search(shapeNames, gripScales, dataDir, ...
    outputDir, experimentConfig, cfg);

filename = sprintf('%s/hyper_results.mat', outputDir);
save(filename, 'hyperResults');
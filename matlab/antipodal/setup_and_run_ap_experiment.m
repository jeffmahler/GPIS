% script for setting up and running an antipodal experiment

dim = 25;
% CHANGE BELOW WHEN CREATION IS OVER!
dataDir = 'data/google_objects/scratch';
shapeNames = {'tape'};%{'can_opener', 'loofa', 'marker', 'plane', 'squirt_bottle', 'stapler', 'tape', 'water'};%{'marker'};
gripScales = {0.75};%{0.8, 1.2, 1.2, 3.0, 0.6, 1.0, 2.0, 0.75};
outputDir = 'results/optimization';
meanCompDir = 'results/mean_vs_predicted_exp/results_final';
newShape = true;
scale = 4;
createGpis = true;

%% experiment config
experimentConfig = struct();
experimentConfig.graspIters = 0;
experimentConfig.frictionCoef = 0.5;
experimentConfig.numSamples = 1000;
experimentConfig.surfaceThresh = 0.15;
experimentConfig.arrowLength = 4;
experimentConfig.loaScale = 1.75;
experimentConfig.graspSigma = 0.25;
experimentConfig.numBadContacts = 10;
experimentConfig.visSampling = false;
experimentConfig.visOptimization = true;
experimentConfig.visGrasps = true;
experimentConfig.rejectUnsuccessfulGrasps = false;

experimentConfig.smoothWin = 1;
experimentConfig.smoothSig = 0.001;

experimentConfig.gripWidth = dim; % dimension of grasp when scale = 1.0
experimentConfig.plateScale = 0.05;
experimentConfig.objScale = 0.95;
experimentConfig.qScale = 100;

experimentConfig.evalUcGrasps = true;
experimentConfig.evalRandSampleFcGrasps = true;

%% variance parameters
varParams = struct();
varParams.y_thresh1_low = 19;
varParams.y_thresh1_high = 24;
varParams.x_thresh1_low = 10;
varParams.x_thresh1_high = 13;

varParams.y_thresh2_low = 6;
varParams.y_thresh2_high = 10;
varParams.x_thresh2_low = 10;
varParams.x_thresh2_high = 13;

varParams.y_thresh3_low = 22;
varParams.y_thresh3_high = 21;
varParams.x_thresh3_low = 12;
varParams.x_thresh3_high = 16;

varParams.occ_y_thresh1_low = 25;
varParams.occ_y_thresh1_high = 24;
varParams.occ_x_thresh1_low = 0;
varParams.occ_x_thresh1_high = 13;

varParams.occ_y_thresh2_low = 21;
varParams.occ_y_thresh2_high = 20;
varParams.occ_x_thresh2_low = 1;
varParams.occ_x_thresh2_high = 7;

varParams.transp_y_thresh1_low = 6;
varParams.transp_y_thresh1_high = 24;
varParams.transp_x_thresh1_low = 0;
varParams.transp_x_thresh1_high = 11;
% varParams.y_thresh1_low = dim;
% varParams.y_thresh1_high = dim;
% varParams.x_thresh1_low = dim;
% varParams.x_thresh1_high = dim;
% 
% varParams.y_thresh2_low = dim;
% varParams.y_thresh2_high = dim;
% varParams.x_thresh2_low = dim;
% varParams.x_thresh2_high = dim;
% 
% varParams.y_thresh3_low = dim;
% varParams.y_thresh3_high = dim;
% varParams.x_thresh3_low = dim;
% varParams.x_thresh3_high = dim;

varParams.occlusionScale = 1000;
varParams.transpScale = 5.0;
varParams.noiseScale = 0.2;
varParams.interiorRate = 0.1;
varParams.specularNoise = true;
varParams.sparsityRate = 0.4;
varParams.sparseScaling = 1000;
varParams.edgeWin = 2;

varParams.noiseGradMode = 'None';
varParams.horizScale = 1;
varParams.vertScale = 1;

%% training parameters
trainingParams = struct();
trainingParams.activeSetMethod = 'Full';
trainingParams.activeSetSize = 100;
trainingParams.beta = 10;
trainingParams.numIters = 1;
trainingParams.eps = 1e-2;
trainingParams.levelSet = 0;
trainingParams.surfaceThresh = experimentConfig.surfaceThresh;
trainingParams.scale = scale;
trainingParams.numSamples = 20;
trainingParams.trainHyp = false;
trainingParams.hyp = struct();
trainingParams.hyp.cov = [1, 0];
trainingParams.hyp.mean = [0; 0; -0.5];
trainingParams.hyp.lik = log(0.1);


%% optimization parameters
cf = cos(atan(experimentConfig.frictionCoef));

cfg = struct();
cfg.max_iter = 10;
cfg.max_penalty_iter = 5;
cfg.max_merit_coeff_increases = 5;
cfg.merit_coeff_increase_ratio = 1.6;
cfg.initial_penalty_coeff = 0.5;
cfg.initial_trust_box_size = 10;
cfg.trust_shrink_ratio = .4;
cfg.trust_expand_ratio = 1.25;
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
cfg.nu = 0.75;
cfg.beta = 2.0;
cfg.fric_coef = 0; % use no-slip constraint to force antipodality but allow solutions within the friction cone

% cfgArray = load('results/google_objects/hyp_results_3444.mat');
% cfg = cfgArray.hyperResults{1,55}.cfg;
% cfg.scale = scale;
% cfg.nu = 1.0;
%% run experiment
%rng(300);

allShapeResults = cell(1,size(shapeNames,2));

for i = 1:size(shapeNames,2)
    filename = shapeNames{i};
    gripScale = gripScales{i};
    fprintf('Running experiment for shape %s\n', filename);
    
    % Run experiment on next shape
    [experimentResults, gpModel, shapeParams, shapeSamples] = ...
        run_antipodal_experiment(dim, filename, gripScale, dataDir, ...
                                 outputDir, meanCompDir, newShape, ...
                                 experimentConfig, varParams, ...
                                 trainingParams, cfg, createGpis);
                             
    % Store results
    shapeResult = struct();
    shapeResult.experimentResults = experimentResults;
    shapeResult.gpModel = gpModel;
    shapeResult.shapeParams = shapeParams;
    shapeResult.shapeSamples = shapeSamples;
    allShapeResults{i} = shapeResult;
end
                         
% script for setting up and running an antipodal experiment

dim = 25;
dataDir = 'data/google_objects';%/google_update_versions';
shapeNames = {'measure'};%{'marker'};
outputDir = 'data/google_update_versions/';
newShape = true;
scale = 2;
createGpis = true;

%% experiment config
experimentConfig = struct();
experimentConfig.graspIters = 0;
experimentConfig.frictionCoef = 0.5;
experimentConfig.numSamples = 1000;
experimentConfig.surfaceThresh = 0.15;
experimentConfig.arrowLength = 4;
experimentConfig.loaScale = 1.75;
experimentConfig.numBadContacts = 10;
experimentConfig.visSampling = false;
experimentConfig.visOptimization = true;
experimentConfig.visGrasps = true;
experimentConfig.rejectUnsuccessfulGrasps = false;

experimentConfig.evalUcGrasps = true;
experimentConfig.evalRandSampleFcGrasps = true;

%% variance parameters
varParams = struct();
varParams.y_thresh1_low = 0;
varParams.y_thresh1_high = 10;
varParams.x_thresh1_low = 0;
varParams.x_thresh1_high = 10;
% 
varParams.y_thresh2_low = 12;
varParams.y_thresh2_high = dim;
varParams.x_thresh2_low = 5;
varParams.x_thresh2_high = 7;
% 
varParams.y_thresh3_low = 8;
varParams.y_thresh3_high = 22;
varParams.x_thresh3_low = 16;
varParams.x_thresh3_high = 24;

% varParams.y_thresh1_low = dim;
% varParams.y_thresh1_high = 0;
% varParams.x_thresh1_low = dim;
% varParams.x_thresh1_high = 0;

% varParams.y_thresh2_low = dim;
% varParams.y_thresh2_high = 0;
% varParams.x_thresh2_low = dim;
% varParams.x_thresh2_high = 0;

% varParams.y_thresh3_low = dim;
% varParams.y_thresh3_high = 0;
% varParams.x_thresh3_low = dim;
% varParams.x_thresh3_high = 0;

varParams.occlusionScale = 1000;
varParams.noiseScale = 0.1;
varParams.interiorRate = 0.2;
varParams.specularNoise = true;
varParams.sparsityRate = 0.75;
varParams.sparseScaling = 1000;
varParams.edgeWin = 1;

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
cfg.merit_coeff_increase_ratio = 2;
cfg.initial_penalty_coeff = 0.25;
cfg.initial_trust_box_size = 15;
cfg.trust_shrink_ratio = .75;
cfg.trust_expand_ratio = 2.0;
cfg.min_approx_improve = 0.1;
cfg.min_trust_box_size = 0.5;
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
cfg.beta = 2.0;
cfg.fric_coef = 0; % use no-slip constraint to force antipodality but allow solutions within the friction cone

% cfgArray = load('results/google_objects/hyp_results_3444.mat');
% cfg = cfgArray.hyperResults{1,55}.cfg;
% cfg.scale = scale;
% cfg.nu = 1.0;
%% run experiment
rng(100);

allShapeResults = cell(1,size(shapeNames,2));

for i = 1:size(shapeNames,2)
    filename = shapeNames{i};
    fprintf('Running experiment for shape %s\n', filename);
    
  
    [gpModel, shapeParams, shapeSamples, constructionResults] = ...
        create_experiment_object(dim, filename, dataDir, newShape, ...
                                 experimentConfig, varParams, trainingParams, ...
                                 cfg.scale);
                             
     save('data/google_update_versions/measure_cup_gpis.mat','gpModel');
     save('data/google_update_versions/measure_cup_samples.mat','shapeSamples');
     save('data/google_update_versions/measure_cup_construction.mat','constructionResults');
     save('data/google_update_versions/measure_cup_variance_params.mat','varParams');
     save('data/google_update_versions/measure_cup.mat','shapeParams');
                                                      
    % Store results
    shapeResult = struct();
    shapeResult.experimentResults = experimentResults;
    shapeResult.gpModel = gpModel;
    shapeResult.shapeParams = shapeParams;
    shapeResult.shapeSamples = shapeSamples;
    allShapeResults{i} = shapeResult;
end
                         
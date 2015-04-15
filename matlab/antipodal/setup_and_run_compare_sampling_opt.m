% script for setting up and running an antipodal experiment

dim = 25;
% CHANGE BELOW WHEN CREATION IS OVER!
dataDir = 'data/google_objects/icra';

% BELOW ARE FINAL CONFIG FOR ICRA
shapeNames = {'knob', 'marker', 'nail', 'plane', 'squirt_bottle', 'splitter', 'switch', 'tape'};
%{'can_opener', 'deodorant', 'marker', 'plane', 'squirt_bottle', 'stapler', 'tape', 'water'};
gripScales = {0.95, 0.75, 0.95, 0.85, 0.4, 1.0, 0.95, 0.85};
%{0.38, 0.6, 0.8, 1.2, 0.4, 0.6, 0.85, 0.4};
% TEST CONFIG
%shapeNames = {'tape'};
%gripScales = {0.4};
% shapeNames = {'tape'};%{'can_opener', 'loofa', 'marker', 'plane', 'squirt_bottle', 'stapler', 'tape', 'water'};%{'marker'};
% gripScales = {0.5};%{0.8, 1.2, 1.2, 3.0, 0.6, 1.0, 2.0, 0.75};
outputDir = 'results/optimization';
meanCompDir = 'results/mean_vs_predicted_exp/icra_long';
newShape = false;
createGpis = false;
scale = 4;
sampleIters = 20;

%% experiment config
experimentConfig = struct();
experimentConfig.trials = 3;
experimentConfig.graspIters = 20;%10;
experimentConfig.frictionCoef = 0.4;
experimentConfig.numSamples = 1000;
experimentConfig.surfaceThresh = 0.15;
experimentConfig.arrowLength = 4;
experimentConfig.loaScale = 1.75;
experimentConfig.graspSigma = 1.5;
experimentConfig.numBadContacts = 10;
experimentConfig.visSampling = false;
experimentConfig.visOptimization = true;
experimentConfig.visGrasps = true;
experimentConfig.rejectUnsuccessfulGrasps = false;

experimentConfig.smoothWin = 1;
experimentConfig.smoothSig = 0.001;
experimentConfig.shapeSampleTime = 1.5; % time to sample shapes

experimentConfig.gripWidth = dim; % dimension of grasp when scale = 1.0
experimentConfig.plateScale = 0.075;
experimentConfig.objScale = 0.95;
experimentConfig.qScale = 1;

experimentConfig.evalUcGrasps = true;
experimentConfig.evalRandSampleFcGrasps = true;

%% variance parameters
varParams = struct();
varParams.y_thresh1_low = 5;
varParams.y_thresh1_high = 24;
varParams.x_thresh1_low = 1;
varParams.x_thresh1_high = 25;

varParams.y_thresh2_low = 25;
varParams.y_thresh2_high = 24;
varParams.x_thresh2_low = 1;
varParams.x_thresh2_high = 25;

varParams.y_thresh3_low = 26;
varParams.y_thresh3_high = 25;
varParams.x_thresh3_low = 26;
varParams.x_thresh3_high = 25;

varParams.occ_y_thresh1_low = 26;
varParams.occ_y_thresh1_high = 12;
varParams.occ_x_thresh1_low = 1;
varParams.occ_x_thresh1_high = 25;

varParams.occ_y_thresh2_low = 26;
varParams.occ_y_thresh2_high = 25;
varParams.occ_x_thresh2_low = 26;
varParams.occ_x_thresh2_high = 25;

varParams.transp_y_thresh1_low = 26;
varParams.transp_y_thresh1_high = 25;
varParams.transp_x_thresh1_low = 26;
varParams.transp_x_thresh1_high = 25;

varParams.transp_y_thresh2_low = 26;
varParams.transp_y_thresh2_high = 25;
varParams.transp_x_thresh2_low = 26;
varParams.transp_x_thresh2_high = 25;

varParams.occlusionScale = 1000;
varParams.transpScale = 4;
varParams.noiseScale = 0.4;
varParams.interiorRate = 0.05;
varParams.specularNoise = true;
varParams.sparsityRate = 0.05;
varParams.sparseScaling = 1000;
varParams.edgeWin = 2;

varParams.noiseGradMode = 'None';
varParams.horizScale = 1;
varParams.vertScale = 1;

%% training parameters
trainingParams = struct();
trainingParams.activeSetMethod = 'Full';
trainingParams.activeSetSize = 1;
trainingParams.beta = 10;
trainingParams.firstIndex = 150;
trainingParams.numIters = 0;
trainingParams.eps = 1e-2;
trainingParams.delta = 1e-2;
trainingParams.levelSet = 0;
trainingParams.surfaceThresh = experimentConfig.surfaceThresh;
trainingParams.scale = scale;
trainingParams.numSamples = 20;
trainingParams.trainHyp = false;
trainingParams.downsample = 1;
trainingParams.hyp = struct();
trainingParams.hyp.cov = [log(exp(1)), log(1)];
trainingParams.hyp.mean = [0; 0; -1];
trainingParams.hyp.lik = log(0.1);
trainingParams.useGradients = true;
trainingParams.image = ...
    imread('/Users/jeff/Documents/Research/implicit_surfaces/docs/icra_drafts/new_examples/DSC_0491.JPG');
trainingParams.cdim = scale * dim;

%% optimization parameters
cf = cos(atan(experimentConfig.frictionCoef));

cfg = struct();
cfg.max_iter = 10;
cfg.max_penalty_iter = 6;
cfg.max_merit_coeff_increases = 5;
cfg.merit_coeff_increase_ratio = 2.0;
cfg.initial_penalty_coeff = 0.25;
cfg.initial_trust_box_size = 10;
cfg.trust_shrink_ratio = .4;
cfg.trust_expand_ratio = 1.25;
cfg.min_approx_improve = 0.05;
cfg.min_trust_box_size = 0.5;
cfg.callback = @plot_surface_grasp_points;
cfg.full_hessian = false;
cfg.cnt_tolerance = 1e-4;
cfg.ineq_tolerance = [1-cf; 1-cf; 0.1; 0.1];
cfg.eq_tolerance = [10e0; 10e0; 5e-2; 5e-2; 0.1];
cfg.prog_tolerance = 1.0;
cfg.com_tol = 2.0;
cfg.scale = scale;
cfg.arrow_length = experimentConfig.arrowLength;
cfg.min_init_dist = 8;
cfg.lambda = 0.00;
cfg.nu = 2.0;
cfg.beta = 2.0;
cfg.fric_coef = 0; % use no-slip constraint to force antipodality but allow solutions within the friction cone

% cfgArray = load('results/google_objects/hyp_results_3444.mat');
% cfg = cfgArray.hyperResults{1,55}.cfg;
% cfg.scale = scale;
% cfg.nu = 1.0;
%% run experiment
%rng(260);
allShapeResults = cell(1,size(shapeNames,2));
avgSpeedups = zeros(size(shapeNames,2),1);
for i = 1:size(shapeNames,2)
    filename = shapeNames{i};
    gripScale = gripScales{i};
    fprintf('Running experiment for shape %s\n', filename);
    
    % Run experiment on next shape
    [experimentResults, gpModel, shapeParams, shapeSamples] = ...
        run_compare_sampling_opt(dim, filename, gripScale, dataDir, ...
                                 outputDir, meanCompDir, sampleIters, newShape, ...
                                 experimentConfig, varParams, ...
                                 trainingParams, cfg, createGpis);
                             
    
    avgSpeedups(i) = ...
        mean(experimentResults.allSampleConvTimes) ./ mean(experimentResults.allOptConvTimes);
    
    % Store results
    shapeResult = struct();
    shapeResult.experimentResults = experimentResults;
    shapeResult.gpModel = gpModel;
    shapeResult.shapeParams = shapeParams;
    shapeResult.shapeSamples = shapeSamples;
    allShapeResults{i} = shapeResult;
end
                         
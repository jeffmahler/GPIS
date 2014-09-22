% get pr2 grasps

sourceDir = 'data/pr2_registration/tape1';
numImages = 10;
cfg.gridDim = 25;
cfg.scale = 4;
cfg.objName = 'pc_tape3';

% params for constructing pointcloud from images
cfg = struct();
cfg.cbSquareMeters = 0.0293; % in meters
cfg.depthThresh = -1e-4;
cfg.truncation = 2;
cfg.noiseScale = 5.0;
cfg.disagreePenalty = 5;
cfg.numSamples = 1000;
cfg.insideScale = 5;
cfg.colorSlack = 20;

% gpis training parameters
trainingParams = struct();
trainingParams.activeSetMethod = 'Full';
trainingParams.activeSetSize = 100;
trainingParams.beta = 10;
trainingParams.numIters = 1;
trainingParams.eps = 1e-2;
trainingParams.levelSet = 0;
trainingParams.surfaceThresh = 0.15;
trainingParams.scale = cfg.scale;
trainingParams.numSamples = 20;
trainingParams.trainHyp = false;
trainingParams.hyp = struct();
trainingParams.hyp.cov = [1, 0];
trainingParams.hyp.mean = [0; 0; -0.5];
trainingParams.hyp.lik = log(0.1);

[pr2GpModel, shapeSamples, constructionResults, transformResults] = ...
    gpis_from_depth(sourceDir, numImages, cfg, trainingParams);

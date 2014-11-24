% get pr2 grasps

sourceDir = 'data/pr2_registration/tape4';
numImages = 10;

% params for constructing pointcloud from images
cfg = struct();
cfg.gridDim = 25;
cfg.scale = 4;
cfg.objName = 'pc_tape3';

cfg.cbSquareMeters = 0.0283; % in meters
cfg.depthThresh = 2e-3;
cfg.truncation = 1;
cfg.noiseScale = 5.0;
cfg.disagreePenalty = 10;
cfg.numSamples = 1000;
cfg.insideScale = 1;
cfg.colorSlack = 15;

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

%%
% constructionResults.newSurfaceImage = ...
%     create_tsdf_image_sampled(constructionResults.predGrid, ...
%         shapeSamples, cfg.scale, 1.0, false, false);
% figure(88);
% imshow(constructionResults.newSurfaceImage);

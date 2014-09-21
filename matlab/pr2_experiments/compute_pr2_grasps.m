% get pr2 grasps

sourceDir = 'data/pr2_registration/tape1';
numImages = 2;

cfg = struct();
cfg.cbSize = [5, 4]; % dimensions of cb in squares (height and width)
cfg.cbSquareMeters = 0.0293; % in meters
cfg.depthThresh = 1e-3;
cfg.truncation = 10;

pr2GpModel = gpis_from_depth(sourceDir, numImages, cfg);
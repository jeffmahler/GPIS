% Script for generating a 2d GPIS
% INSTRUCTIONS:
%   1. Specify a type of shape
%   2. Click surface, interior, and exterior points using GUI to generate
%   the TSDF
%   3. Create GPIS representation
%   4. Predict values for the entire space and threshold to find the surface
clear; clc; close all;

%% Collect manually specified tsdf

% param specification
shape = 'Polygons';
gridDim = 200;
scale = 2;
fill = true;

% active set params
numIters = 5;
thresh = 0.25;
beta = 30;
h = 0;
eps = 1e-2;

% experiment params
activeSetSizes = [250];%, 250, 500, 1000];
numSizes = size(activeSetSizes, 2);
methods = {'LevelSet'}; %, 'Subsample', 'LevelSet'};
numMethods = size(methods,2);
selectionTimes = {zeros(numSizes,1), zeros(numSizes,1), zeros(numSizes,1), zeros(numSizes,1)};
testErrors = {[], [], [], []};
surfaceTestErrors = {[], [], [], []};

[truePoints, trueTsdf, trueShapeImage] = auto_tsdf(shape, gridDim);

%temp
%activeSetSizes = [size(truePoints,1)];

%% Test all methodologies and set sizes
for i = 1:size(methods,2)
    M = methods{i};
    fprintf('Testing method %s\n', M);
    selectionTimes{1} = zeros(numSizes,1);
    for j = 1:size(activeSetSizes,2)
 
% Learn active set and create GPIS representation
K = activeSetSizes(j);
fprintf('Testing size %d\n', K);

tic;
[gpModel, activePoints, testIndices, testPredTsdf, testPredVars] = ...
    select_active_set(M, truePoints, trueTsdf, K, numIters, h, beta, eps);
elapsedTime = toc;

selectionTimes{i}(j) = elapsedTime;

fprintf('Active set learning for %d elements took %f sec\n', K, elapsedTime);

% Test the squared error from the true tsdf values for the whole TSDF
tsdfReconstructionError = evaluate_errors(testPredTsdf, trueTsdf(testIndices), K);
testErrors{i} = [testErrors{i} tsdfReconstructionError];
save(sprintf('results/active_set/tsdfReconstructionError%s%d.mat', M, K), 'tsdfReconstructionError'); % save errors

fprintf('Entire TSDF: Mean error: %f, std error %f\n', tsdfReconstructionError.meanError, ...
    tsdfReconstructionError.stdError);

% Test the squared error from the true tsdf values near the surface
[surfaceTestIndices, surfaceTestPredTsdf, surfaceTestPredVars] = ...
    predict_2d_surface(truePoints, trueTsdf, gpModel, testIndices);
surfaceReconstructionError = evaluate_errors(surfaceTestPredTsdf, ...
    trueTsdf(surfaceTestIndices), K);
surfaceTestErrors{i} = [surfaceTestErrors{i} surfaceReconstructionError];
save(sprintf('results/active_set/surfaceReconstructionError%s%d.mat', M, K), 'surfaceReconstructionError'); % save errors

fprintf('Near Surface: Mean error: %f, std error %f\n', surfaceReconstructionError.meanError, ...
    surfaceReconstructionError.stdError);

% Display resulting TSDF
[allPoints, allTsdf, allVars, surfacePoints, surfaceTsdf, surfaceVars] ...
    = predict_2d_grid( gpModel, gridDim, thresh);

numTest = size(allPoints, 1);
testColors = repmat(ones(numTest,1) - ...
    abs(allTsdf) / max(abs(allTsdf)) .* ones(numTest,1), 1, 3);
testVarColors = repmat(ones(numTest,1) - ...
    abs(allVars) / max(abs(allVars)) .* ones(numTest,1), 1, 3);

testImage = reshape(testColors(:,1), gridDim, gridDim); 
testImage = imresize(testImage, scale*size(testImage));
testVarImage = reshape(testVarColors(:,1), gridDim, gridDim); 
testVarImage = imresize(testVarImage, scale*size(testVarImage));
testImageDarkened = max(0, testImage - 0.3*ones(scale*gridDim, scale*gridDim)); % darken

figure;
imshow(testImageDarkened);
hold on;
scatter(scale*activePoints(1,1), scale*activePoints(1,2), 150.0, 'x', 'LineWidth', 1.5);
scatter(scale*activePoints(:,1), scale*activePoints(:,2), 50.0, 'x', 'LineWidth', 1.5);
hold off;
title(sprintf('Predicted Absolute TSDF for %d Active Elements Selected Using %s Method (White = Surface)', K, M));

% Display additional info 
% figure;
% subplot(1,3,1);
% imshow(testImageDarkened);
% hold on;
% scatter(scale*activePoints(:,1), scale*activePoints(:,2), 50.0, 'x', 'LineWidth', 1.5);
% hold off;
% title('Predicted Absolute TSDF  with Samples(White = 0, Black = 1)');
% subplot(1,3,2);
% imshow(testVarImage);
% title('Predicted Variance');
% subplot(1,3,3);
% imshow(trueShapeImage);
% title('True Shape');

% Write to file
imwrite(testImageDarkened, sprintf('results/active_set/tsdf%s%d.jpg', M, K));

    end
end

%% Aggregate data (or do this manually)
plotMethod = 2;
times = zeros(numSizes, numMethods);
colors = ['r', 'g', 'b', 'c'];

figure;
for k = 1:numMethods
    times(:, k) = selectionTimes{k}';
    plot(activeSetSizes, log(times(:,k)), colors(k), 'LineWidth', 2);
    hold on;
end

title('Active set selection time (log) versus set size');
xlabel('# Elements');
ylabel('Time (sec)');
legend(methods{1}, methods{2}, methods{3}, methods{4}, 'Location', 'Best');


%%
means = zeros(numSizes,1);
stds = zeros(numSizes,1);

for k = 1:numSizes
    means(k) = testErrors{plotMethod}(k).meanError;
    stds(k) = testErrors{plotMethod}(k).stdError;
end

figure;
errorbar(activeSetSizes, means, stds, 'LineWidth', 2);
title('Active set size versus full TSDF reconstruction error');
xlabel('# Elements');
ylabel('Error (signed distance)');

%% surface errors
means = zeros(numSizes, numMethods);
stds = zeros(numSizes, numMethods);
colors = ['r', 'g', 'b', 'c'];

figure;
for l = 1:numMethods
    for k = 1:numSizes
        means(k, l) = testErrors{l}(k).meanError;
        stds(k, l) = testErrors{l}(k).stdError;
    end
    errorbar(activeSetSizes, means(:,l), stds(:,l), colors(l), 'LineWidth', 2);
    hold on;
end

title('Active set size versus surface reconstruction error');
xlabel('# Elements');
ylabel('Error (signed distance)');
legend(methods{1}, methods{2}, methods{3}, methods{4}, 'Location', 'Best');




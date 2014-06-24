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
gridDim = 25;
name = 'shape6.csv';
scale = 2;
fill = true;

% active set params
numIters = 5;
thresh = 0.25;
beta = 500;
h = 0;
eps = 1e-2;

% experiment params
activeSetSizes = [75];%, 250, 500, 1000];
numSizes = size(activeSetSizes, 2);
methods = {'LevelSet'}; %, 'Subsample', 'LevelSet'};
numMethods = size(methods,2);
selectionTimes = {zeros(numSizes,1), zeros(numSizes,1), zeros(numSizes,1), zeros(numSizes,1)};
testErrors = {[], [], [], []};
surfaceTestErrors = {[], [], [], []};

[truePoints, trueTsdf, trueShapeImage] = auto_tsdf(shape, gridDim, name);
trueTsdfGrid = reshape(trueTsdf, gridDim, gridDim);
[Gx, Gy] = imgradientxy(trueTsdfGrid, 'CentralDifference');
trueNormals = [Gx(:) Gy(:)];

%temp
%activeSetSizes = [size(truePoints,1)];

%% Test all methodologies and set sizes
rng(100);
for i = 1:size(methods,2)
    M = methods{i};
    fprintf('Testing method %s\n', M);
    selectionTimes{1} = zeros(numSizes,1);
    for j = 1:size(activeSetSizes,2)
 
        % Learn active set and create GPIS representation
        K = activeSetSizes(j);
        fprintf('Testing size %d\n', K);

        tic;
        [gpModel, activePoints, testIndices, testPredTsdf, testPredNormals, testPredVars] = ...
            select_active_set(M, truePoints, trueTsdf, K, numIters, h, ...
            beta, eps, trueNormals);
        elapsedTime = toc;

        selectionTimes{i}(j) = elapsedTime;

        fprintf('Active set learning for %d elements took %f sec\n', K, elapsedTime);

        % Test the squared error from the true tsdf values for the whole TSDF
        numTest = size(testIndices,1);
        tsdfReconstructionError = ...
            evaluate_errors(testPredTsdf, trueTsdf(testIndices), K);
        testErrors{i} = [testErrors{i} tsdfReconstructionError];
        save(sprintf('results/active_set/tsdfReconstructionError%s%d.mat', M, K), ...
            'tsdfReconstructionError'); % save errors

        normalError = ...
            evaluate_errors(testPredNormals, ...
                reshape(trueNormals(testIndices,:), 2*numTest, 1), K);
        %testErrors{i} = [testErrors{i} tsdfReconstructionError];
        save(sprintf('results/active_set/normalError%s%d.mat', M, K), ...
            'normalError'); % save errors

        
        fprintf('Entire TSDF: Mean error: %f, std error %f\n', ...
            tsdfReconstructionError.meanError, ...
            tsdfReconstructionError.stdError);
        fprintf('Normals: Mean error: %f, std error %f\n', ...
            normalError.meanError, ...
            normalError.stdError); 

        
        % Test the squared error from the true tsdf values near the surface
%         [surfaceTestIndices, surfaceTestPredTsdf, surfaceTestPredVars] = ...
%             predict_2d_surface(truePoints, trueTsdf, gpModel, testIndices);
%         surfaceReconstructionError = evaluate_errors(surfaceTestPredTsdf, ...
%             trueTsdf(surfaceTestIndices), K);
%         surfaceTestErrors{i} = [surfaceTestErrors{i} surfaceReconstructionError];
%         save(sprintf('results/active_set/surfaceReconstructionError%s%d.mat', M, K), ...
%             'surfaceReconstructionError'); % save errors
% 
%         fprintf('Near Surface: Mean error: %f, std error %f\n', ...
%             surfaceReconstructionError.meanError, ...
%             surfaceReconstructionError.stdError);

        % Display resulting TSDF
        [allPoints, allTsdf, allNorms, allVars, surfacePoints, surfaceTsdf, surfaceVars] ...
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
        subplot(1,2,1);
        imshow(testImageDarkened);
        hold on;
        scatter(scale*activePoints(1,1), scale*activePoints(1,2), 150.0, 'x', 'LineWidth', 1.5);
        scatter(scale*activePoints(:,1), scale*activePoints(:,2), 50.0, 'x', 'LineWidth', 1.5);
        hold off;
        subplot(1,2,2);
        imshow(trueTsdfGrid);
        title(sprintf('Predicted Absolute TSDF for %d Active Elements Selected Using %s Method (White = Surface)', K, M));

        % Write to file
        imwrite(testImageDarkened, sprintf('results/active_set/tsdf%s%d.jpg', M, K));

    end
end

%%
subplot(1,3,1);
imshow(trueTsdfGrid);
subplot(1,3,2);
imshow(abs(Gx));
subplot(1,3,3);
imshow(abs(Gy));

%% get grasp points
surf_thresh = 0.1;
[x_surface_i, x_surface_j] = ...
    find(abs(reshape(allTsdf, gridDim, gridDim)) < surf_thresh);
n_surf = size(x_surface_i, 1);
ind1 = uint16(rand * n_surf);
ind2 = uint16(rand * n_surf);
x_init = [x_surface_j(ind1); x_surface_i(ind1);...
          x_surface_j(ind2); x_surface_i(ind2)];
%%
x_grasp = find_antipodal_grasp_points(x_init, gpModel, testImageDarkened, ...
    gridDim, scale);


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

%% Plot K vs times
figure(20);
plot(activeSetSizes, selectionTimes{1}, '-bo', activeSetSizes, selectionTimes{2}, '-gx', 'MarkerSize', 15, 'LineWidth', 2);
title('Computation Time Versus Set Size for 40,000 Data Points', 'FontSize', 15);
xlabel('Active Set Size', 'FontSize', 15);
ylabel('Time (sec)', 'FontSize', 15);
legend('Serial', 'GPU (Chol)', 'Location', 'Best');

%% Plot N vs times
selectionTimesSerial = [0.2078, 0.4642, 1.9013, 15.3608, 64.0320, 251.9920]';
selectionTimesGpu = [0.0887, 0.2583, 1.0540, 4.0220, 15.5049, 60.1130]';
numPoints = [25^2, 50^2, 100^2, 200^2, 400^2, 800^2]';

figure(21);
plot(numPoints, selectionTimesSerial, '-bo', numPoints, selectionTimesGpu, '-gx', 'MarkerSize', 15, 'LineWidth', 2);
title('Computation Time Versus Number of Candidate Points for 100 Active Points', 'FontSize', 15);
xlabel('Number of Candidate Points', 'FontSize', 15);
ylabel('Time (sec)', 'FontSize', 15);
legend('Serial', 'GPU (Chol)', 'Location', 'Best');

%% Plot B vs times
batchSelectionTimesGpu = [155.5570, 25.3074, 18.5861, 8.4950, 4.8721, 4.1361, 4.0337, 4.0087, 3.9083]';
batchSizes = [1, 8, 16, 64, 128, 256, 512, 1024, 2048]';

figure(22);
loglog(batchSizes, batchSelectionTimesGpu, '-bo', 'MarkerSize', 15, 'LineWidth', 2);
title('Computation Time Versus Batch Size for 40,000 Data Points', 'FontSize', 15);
xlabel('Batch Size', 'FontSize', 15);
ylabel('Time (sec)', 'FontSize', 15);


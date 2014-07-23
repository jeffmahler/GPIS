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
scale = 2;
fill = true;
name = 'water';
shapeName = sprintf('data/google_objects/%s.mat', name);
tsdfName = sprintf('data/google_objects/%s_tsdf.csv', name);
pointsName = sprintf('data/google_objects/%s_points.csv', name);

% active set params
numIters = 1;%000;
thresh = 0.25;

h = 0;

beta = 10;
eps = 1e-2;

% experiment params
activeSetSizes = [250];%, 250, 500, 1000];
numSizes = size(activeSetSizes, 2);
methods = {'LevelSet'}; %, 'Subsample', 'LevelSet'};
numMethods = size(methods,2);
selectionTimes = {zeros(numSizes,1), zeros(numSizes,1), zeros(numSizes,1), zeros(numSizes,1)};
testErrors = {[], [], [], []};
surfaceTestErrors = {[], [], [], []};

pts = create_shape(gridDim);

%%
[truePoints, trueTsdf, trueNormals, trueNoise, trueShape, trueShapeImage] = ...
    auto_tsdf(shape, gridDim, tsdfName, pts);
trueTsdfGrid = reshape(allTsdf, gridDim, gridDim);
 
% write points to a file for later
save(shapeName, 'trueShape'); 
csvwrite(pointsName, reshape(pts, 2, size(pts,2)/2)');

%% Test all methodologies and set sizes
rng(100);
for i = 1:size(methods,2)
    M = methods{i};
    fprintf ('Testing method %s\n', M);
    selectionTimes{1} = zeros(numSizes,1);
    for j = 1:size(activeSetSizes,2)
 
        % Learn active set and create GPIS representation
        K = activeSetSizes(j);
        fprintf('Testing size %d\n', K);

        tic;
        [gpModel, activePoints, testIndices, testPredTsdf, testPredNormals, testPredVars] = ...
            select_active_set(M, truePoints, trueTsdf, K, numIters, h, ...
            beta, eps, trueNormals, trueNoise);
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

        allTsdfGrid = reshape(allTsdf, gridDim, gridDim);
        numTest = size(allPoints, 1);
        testColors = repmat(ones(numTest,1) - ...
            abs(allTsdf) / max(abs(allTsdf)) .* ones(numTest,1), 1, 3);
        testVarColors = repmat(ones(numTest,1) - ...
            abs(allVars) / max(abs(allVars)) .* ones(numTest,1), 1, 3);

        testImage = reshape(testColors(:,1), gridDim, gridDim); 
        testImage2 = imresize(testImage, scale*size(testImage));
        testVarImage = reshape(testVarColors(:,1), gridDim, gridDim); 
        testVarImage = imresize(testVarImage, scale*size(testVarImage));
        testImageDarkened = max(0, testImage2 - 0.3*ones(scale*gridDim, scale*gridDim)); % darken

        colorImage = 255*ones(gridDim, gridDim, 'uint8');
        tsdfImage = imresize(testImageDarkened, 0.5);
        allVarsGrid = reshape(allVars, gridDim, gridDim);
        minVar = min(min(allVarsGrid));
        maxVar = max(max(allVarsGrid));
        varImage = zeros(gridDim, gridDim, 3, 'uint8');
        varImage(:,:,2) = uint8(0*colorImage) + uint8((maxVar - allVarsGrid) / (1*(maxVar - minVar)) .* double(colorImage));
        varImage(:,:,1) = uint8(0*colorImage) + uint8((allVarsGrid - minVar) / (1*(maxVar - minVar)) .* double(colorImage));
        combImage = zeros(gridDim, gridDim, 3, 'uint8');
        combImage(:,:,1) = uint8(double(varImage(:,:,1)) .* tsdfImage);
        combImage(:,:,2) = uint8(double(varImage(:,:,2)) .* tsdfImage);
        combImageBig = imresize(combImage, scale);
        
        figure;
        subplot(1,2,1);
        imshow(trueTsdfGrid);
        title(sprintf('True TSDF Grid'));
       % hold on;
       % scatter(scale*activePoints(1,1), scale*activePoints(1,2), 150.0, 'x', 'LineWidth', 1.5);
       % scatter(scale*activePoints(:,1), scale*activePoints(:,2), 50.0, 'x', 'LineWidth', 1.5);
       % hold off;
        subplot(1,2,2);
        
        imshow(testImageDarkened);
        title(sprintf('Predicted TSDF Grid'));
        figure; 
        imshow(testImageDarkened);

       % Write to file
        imwrite(testImageDarkened, sprintf('results/active_set/tsdf%s%d.jpg', M, K));

    end
end


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
[x_grasp, x_all_iters] = find_antipodal_grasp_points(x_init, gpModel, combImageBig, ...
    gridDim, scale);

%% Plot the grasp points
figure(2);
d = 2;
xs1 = x_init(1:d,1);
xs2 = x_init(d+1:2*d,:);

subplot(1,2,1);
imshow(combImageBig); 
title('Initial Grasp', 'FontSize', 15);
hold on;
plot(scale*xs1(1,:), scale*xs1(2,:), 'rx-', 'MarkerSize', 20, 'LineWidth', 1.5);
plot(scale*xs2(1,:), scale*xs2(2,:), 'gx-', 'MarkerSize', 20, 'LineWidth', 1.5);
hold off;

xs1 = x_grasp(1:d,1);
xs2 = x_grasp(d+1:2*d,:);

subplot(1,2,2);
imshow(combImageBig);
title('Final Grasp', 'FontSize', 15);
hold on;
plot(scale*xs1(1,:), scale*xs1(2,:), 'rx-', 'MarkerSize', 20, 'LineWidth', 1.5);
plot(scale*xs2(1,:), scale*xs2(2,:), 'gx-', 'MarkerSize', 20, 'LineWidth', 1.5);
hold off;



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


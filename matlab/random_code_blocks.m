% Random things I like to do at the end of gpid_2d but they're not
% necessary

%% Create a new shape
dim = 25;
dataDir = 'data/google_objects';
filename = 'test_square';
[points, com] = new_shape(filename, dataDir, dim);

%%
varParams = struct();
varParams.y_thresh1_low = dim;
varParams.y_thresh2_low = dim;
varParams.y_thresh1_high = 0;
varParams.y_thresh2_high = 0;
varParams.x_thresh1_low = dim;
varParams.x_thresh2_low = dim;
varParams.x_thresh1_high = 0;
varParams.x_thresh2_high = 0;

varParams.occlusionScale = 1000;
varParams.noiseScale = 0.1;
varParams.specularNoise = false;
varParams.sparsityRate = 0.0;

varParams.noiseGradMode = 'NONE';
varParams.horizScale = 1;
varParams.vertScale = 1;

%%
trainingParams = struct();
trainingParams.activeSetMethod = 'LevelSet';
trainingParams.activeSetSize = 50;
trainingParams.beta = 10;
trainingParams.numIters = 1;
trainingParams.eps = 1e-2;
trainingParams.levelSet = 0;
trainingParams.surfaceThresh = 0.1;

scale = 2;

%%
[shapeParams, shapeImage] = create_tsdf(filename, dataDir, dim, varParams);

%%
[gpModel, predGrid, tsdfReconError, normalError, surfaceImage, selectionTime] = ...
    create_sparse_gpis(shapeParams, trainingParams, scale);

%% find antipodal points
% Set parameters of the optimizer
cfg = struct();
cfg.max_iter = 10;
cfg.max_merit_coeff_increases = 5;
cfg.merit_coeff_increase_ratio = 10;
cfg.initial_penalty_coeff = 1.0;
cfg.initial_trust_box_size = 5;
cfg.trust_shrink_ratio = .75;
cfg.trust_expand_ratio = 2.0;
cfg.min_approx_improve = 1e-8;
cfg.min_trust_box_size = 1e-5;
cfg.callback = @plot_surface_grasp_points;
cfg.full_hessian = true;
cfg.surfaceImage = surfaceImage;
cfg.scale = scale;

%% get grasp points
[x_surface_i, x_surface_j] = ...
    find(abs(reshape(predGrid.tsdf, predGrid.gridDim, predGrid.gridDim)) < trainingParams.surfaceThresh);
n_surf = size(x_surface_i, 1);
ind1 = uint16(rand * n_surf);
ind2 = uint16(rand * n_surf);
x_init = [x_surface_j(ind1); x_surface_i(ind1);...
          x_surface_j(ind2); x_surface_i(ind2)];
      
%%
[x_grasp, x_all_iters] = find_antipodal_grasp_points(x_init, gpModel, ...
    cfg, predGrid.gridDim);

%% Plot the grasp points
figure(2);
d = 2;
xs1 = x_init(1:d,1);
xs2 = x_init(d+1:2*d,:);

subplot(1,2,1);
imshow(surfaceImage); 
title('Initial Grasp', 'FontSize', 15);
hold on;
plot(scale*xs1(1,:), scale*xs1(2,:), 'rx-', 'MarkerSize', 20, 'LineWidth', 1.5);
plot(scale*xs2(1,:), scale*xs2(2,:), 'gx-', 'MarkerSize', 20, 'LineWidth', 1.5);
hold off;

xs1 = x_grasp(1:d,1);
xs2 = x_grasp(d+1:2*d,:);

subplot(1,2,2);
imshow(surfaceImage);
title('Final Grasp', 'FontSize', 15);
hold on;
plot(scale*xs1(1,:), scale*xs1(2,:), 'rx-', 'MarkerSize', 20, 'LineWidth', 1.5);
plot(scale*xs2(1,:), scale*xs2(2,:), 'gx-', 'MarkerSize', 20, 'LineWidth', 1.5);
hold off;

%% Plot surface
figure;
surfaceInd = find(allTsdf < 0.1);
surfaceTsdf = allTsdf(surfaceInd);
surfacePoints = allTsdf(surfaceInd);
imshow(abs(allTsdfGrid) < 0.1);

%% Plot gradients
subplot(1,3,1);
imshow(trueTsdfGrid);
subplot(1,3,2);
imshow(abs(Gx));
subplot(1,3,3);
imshow(abs(Gy));

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
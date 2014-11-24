% Random things I like to do at the end of gpis_2d but they're not
% necessary

%% try out these Brown shapes
close all;

shape1 = imread('data/brown_dataset/Bone01.pgm');
shape2 = imread('data/brown_dataset/Bone02.pgm');

% shape1 = imread('data/pgms/bottle16.pgm');
% shape2 = imread('data/pgms/glas01.pgm');
padding = 20;
D = 0.25;
grid_dim = max([size(shape1) size(shape2)]) + 2*padding;

M1 = 255*ones(grid_dim);
M1(padding+1:padding+size(shape1,1), padding+1:padding+size(shape1,2)) = shape1;
shape1 = M1;

M2 = 255*ones(grid_dim);
M2(padding+1:padding+size(shape2,1), padding+1:padding+size(shape2,2)) = shape2;
shape2 = M2;

tsdf1 = trunc_signed_distance(255-shape1, 25);
tsdf2 = trunc_signed_distance(255-shape2, 25);

penalties = struct();
penalties.Q = 10.0;
stop_criteria = struct();
stop_criteria.T = 1.0;
stop_criteria.eps = 1e-2;
stop_criteria.max_iter = 20;
update_params = struct();
update_params.shrink_temp = 0.1;

tsdf1_down = imresize(tsdf1, D);
tsdf2_down = imresize(tsdf2, D);

figure(2);
subplot(1,2,1);
sdf_surface(tsdf1, 0.5);
subplot(1,2,2);
sdf_surface(tsdf2, 0.5);
% 
% figure(2);
% subplot(1,2,1);
% imagesc(tsdf1);
% subplot(1,2,2);
% imagesc(tsdf2);

%%
interpolate_sdfs(tsdf1, tsdf2, grid_dim, 0.5, 2);

%% matlab registration
[opt, met] = imregconfig('monomodal');
tsdf1_padded = thresh * ones(3*grid_dim); % pad image to remove warp fill-in
tsdf1_padded(grid_dim+1:2*grid_dim, grid_dim+1:2*grid_dim) = tsdf1;
tsdf1_reg = imregister(tsdf1_padded, tsdf2, 'similarity', opt, met);

figure(3);
subplot(1,3,1);
imshow(tsdf1);
title('Original Object', 'FontSize', 15);
subplot(1,3,2);
imshow(tsdf2);
title('Target Object', 'FontSize', 15);
subplot(1,3,3);
imshow(tsdf1_reg);
title('Registered Object', 'FontSize', 15);
%%
interpolate_sdfs(tsdf1_reg, tsdf2, grid_dim, 0.5, 2);

%% std pose
tsdf1_std = standardize_tsdf(tsdf1);
tsdf2_std = standardize_tsdf(tsdf2);

figure(14);
subplot(1,2,1);
imshow(tsdf1_std);
subplot(1,2,2);
imshow(tsdf2_std);

%% KNN
X = [tsdf1(:)'; tsdf2(:)'];
NS = KDTreeSearcher(X, 'BucketSize', 2);

%%
a = zeros(1,grid_dim * grid_dim);
e = knnsearch(NS, a);

%%
registration = register_2d_rigid_unknown_corrs(tsdf1_down, tsdf2_down, penalties,...
    stop_criteria, update_params);
registration.t = (1.0 / D) * registration.t;
registration.s = 1.0;

%%
interpolate_sdfs(tsdf1, tsdf2, grid_dim, 0.5, 2)

%% create registered tsdfs
tsdf1_reg = warp_grid(registration, tsdf1);
%%

A = [registration.R', zeros(2,1); (1.0 / D) * registration.t', 1];
tf = affine2d(A);
tf = maketform('affine', A);
[tsdf1_reg, xdata, ydata] = imtransform(tsdf1, tf, 'Size', [grid_dim, grid_dim]);
%%
interpolate_sdfs(tsdf1_reg, tsdf2, grid_dim, 0.5, 2)
%%
% figure(1);
% subplot(1,2,1);
% imagesc(abs(tsdf1 - tsdf1_reg));
% subplot(1,2,2);
% imagesc(abs(tsdf1_reg - tsdf2));

figure(2);
subplot(1,3,1);
sdf_surface(tsdf1, 0.5);
subplot(1,3,2);
sdf_surface(tsdf2, 0.5);
subplot(1,3,3);
sdf_surface(tsdf1_reg, 0.5);


%%
close all;

predGrid = shapeSamples{15};
tsdf1 = predGrid.tsdf;
predGrid = shapeSamples{50};
tsdf2 = predGrid.tsdf;

interpolate_sdfs(tsdf1, tsdf2, 25, 2.0, 4.0);

%% random shape generator?
grid_dim = 100;
minPoints = 3;
maxPoints = 10;
numPoints = minPoints + uint8((maxPoints-minPoints)*rand());
randomPoints = ones(numPoints,2) + (grid_dim-1)*rand(numPoints, 2);
DT = delaunayTriangulation(randomPoints);

figure(4);
triplot(DT);

[X, Y] = meshgrid(1:grid_dim, 1:grid_dim);
testPoints = [X(:), Y(:)];
insideMask = inhull(testPoints, randomPoints);
insideMask = reshape(insideMask, [grid_dim grid_dim]);
figure(5);
imshow(insideMask);

truncSignedDist = trunc_signed_distance(insideMask, 10);

figure(6);
sdf_surface(truncSignedDist(:), grid_dim, 2);

%% some random ass tsdf stuff
predGrid = experimentResults.constructionResults.predGrid;
tsdfGrid = reshape(predGrid.tsdf, [25, 25]);
tsdfGridBig = high_res_tsdf(tsdfGrid, 2);

figure(1);
tsdfColors = zeros(size(tsdfGridBig,1), size(tsdfGridBig,2), 3);
tsdfColors(:,:,1) = ones(size(tsdfGridBig));
tsdfColors(:,:,2) = ones(size(tsdfGridBig));
surf(tsdfGridBig, tsdfColors);%, 'LineStyle', 'none');

% set(gca, 'XTick', []);
% set(gca, 'YTick', []);
hold on;

zeroCrossing = zeros(size(tsdfGridBig));
%colormap([0,0,1]);
zcColors = zeros(size(tsdfGridBig,1), size(tsdfGridBig,2), 3);
zcColors(:,:,3) = ones(size(tsdfGridBig));
surf(zeroCrossing, zcColors);
%colorbar

tsdfThresh = tsdfGridBig > 0;
SE = strel('square', 3);
I_d = imdilate(tsdfThresh, SE);

% create border masks
insideMaskOrig = (tsdfThresh == 0);
outsideMaskDi = (I_d == 1);
tsdfSurface = double(~(outsideMaskDi & insideMaskOrig));
[interiorI, interiorJ] = find(insideMaskOrig == 1);
%plot3(interiorJ, interiorI, -2*ones(size(interiorI,1),1), ...
%    'LineWidth', 1, 'Color', [1,0,0]);

%%
offset = 1.17; % num seconds to sample shape

for i = 1:8
    maxP = bestPredGrasps{1,i}.sampleTimes.maxP;
    times = bestPredGrasps{1,i}.sampleTimes.sampleTimes;
    adjustedTimes = offset + times;
    adjustedTimesCum = cumsum(adjustedTimes);

    figure(1);
    plot(log(adjustedTimesCum), maxP, 'LineWidth', 2);
    hold on;
end


%% ANNEALING (TODO: try it out)
%     f = @(x) soft_constraint_energy(x, gpModel, optimizationParams, shapeParams.gridDim, shapeParams.com, ...
%             predGrid, coneAngle, experimentConfig.numBadContacts, ...
%             plateWidth, gripWidth, graspSigma);
%     lb = zeros(4,1);
%     ub = shapeParams.gridDim * ones(4,1);
%     
%     info = struct();
%     info.cfg = optimizationParams;
%     plotfn = @(opts,optimvals,flag) plot_surface_grasp_points(optimvals.x, info);
%     options = saoptimset('Display', 'iter', 'DisplayInterval', 25);
%     options = saoptimset(options, 'InitialTemperature', 5);
%     options = saoptimset(options, 'TolFun', 1e-6);
%     options = saoptimset(options, 'MaxIter', 2000);
%     options = saoptimset(options, 'PlotFcns', plotfn, 'PlotInterval', 25);
%     [bestGrasp, opt_p_fc_approx, exitFlat, output] = ...
%         simulannealbnd(f, initGrasp, lb, ub, options);
     

%%
figure(1);
bar([experimentResults.initialGraspResults.meanQ, ...
     experimentResults.randomFcGraspResults.meanQ, ...
     experimentResults.randomSampleFcGraspResults.meanQ, ... 
     experimentResults.antipodalGraspResults.meanQ, ...
     experimentResults.ucMeanGraspResults.meanQ  
    ]);
xlabel('Trial');
ylabel('Expected FC');
title('Comparison of FC Quality');
legend('Initial', 'Sampling (Mean Eval)', 'Sampling (Sample Eval)', 'Antipodal', 'FC Opt', ...
    'Location', 'Best');
% bar(experimentResults.randomSampleFcGraspResults.meanQ);
% bar(experimentResults.antipodalGraspResults.meanQ);
% bar(experimentResults.ucMeanGraspResults.meanQ);

%%
figure(13);
disp('New grasp');
imshow(experimentResults.constructionResults.surfaceImage);
hold on;
initGrasps(:,1) = ...
    get_initial_antipodal_grasp(experimentResults.constructionResults.predGrid, false);
hold off;

%%
rng(100);
%%
figure;
imshow(shapeSurfaceImage);
hold on;
init_grasp = get_initial_antipodal_grasp(experimentResults.constructionResults.predGrid, false);

x1 = init_grasp(1:2);
x2 = init_grasp(3:4);

plot(scale*x1(1,:), scale*x1(2,:), 'rx-', 'MarkerSize', 20, 'LineWidth', 1.5);
plot(scale*x2(1,:), scale*x2(2,:), 'gx-', 'MarkerSize', 20, 'LineWidth', 1.5);
hold off;

%%
best_max_mn_q = 0;
best_mean_mn_q = 0;
best_max_mn_i = 0;
best_mean_mn_i = 0;
means = zeros(144,1);
maxs = zeros(144,1);
for i = 1:144
    max_mn_q = hyperResults{1,i}.max_mn_q;
    mean_mn_q = hyperResults{1,i}.mean_mn_q;
    means(i) = mean_mn_q;
    maxs(i) = max_mn_q;
    %fprintf('Mean: %f\n', mean_mn_q);
    if max_mn_q > best_max_mn_q
        best_max_mn_q = max_mn_q;
        best_max_mn_i = i;
    end
    if mean_mn_q > best_mean_mn_q
        best_mean_mn_q = mean_mn_q;
        best_mean_mn_i = i;
    end
end

%%
 [bestGrasp, bestQ, bestV] = find_grasp_sampling( experimentResults.constructionResults.predGrid, experimentConfig, shapeParams, shapeSamples, experimentResults.constructionResults.surfaceImage, 2, 100);

%% compare different visualizations
shapeSurfaceImage = ...
    create_tsdf_image_sampled(shapeParams, shapeSamples, scale, 1.0);

%%
[shapeImage, surfaceImage] = ...
    create_tsdf_image_blurred(experimentResults.constructionResults.predGrid, scale);

%%
tsdfGrid = reshape(experimentResults.constructionResults.predGrid.tsdf, 25, 25);
tsdfGrid = imresize(tsdfGrid, 2);
meanSurface = zeros(50, 50, 3);
surface = zeros(50,50);
alphaMask = abs(tsdfGrid) < shapeParams.surfaceThresh;
surface(abs(tsdfGrid) < shapeParams.surfaceThresh) = 1;
meanSurface(:,:,3) = surface;
%meanSurface(abs(tsdfGrid) < 0.2, 1) = 0;

figure(4);
subplot(1,3,1);
imshow(experimentResults.constructionResults.surfaceImage);
title('Old version', 'FontSize', 15);
% subplot(1,5,2);
% imshow(shapeImageSampled);
% title('Alpha Blended Tsdf Samples');
subplot(1,3,2);
imshow(shapeSurfaceImage);
hold on;
ms = image(meanSurface);
set(ms,'AlphaData',alphaMask);
hold off;
title('Blended Surface Samples', 'FontSize', 15);
% subplot(1,5,4);
% imshow(shapeImage);
% title('Blurred Tsdf');
subplot(1,3,3);
imshow(surfaceImage);
hold on;
ms = image(meanSurface);
set(ms,'AlphaData',alphaMask);
hold off;
title('Variance-Weighted Gaussian Blurring', 'FontSize', 15);

%%
x1 = [6; 12];
x2 = [16; 12];
grad1 = [1; 0];
grad2 = [-1; 0];
figure(20);
subplot(1,4,1);
plot_grasp_points(shapeSurfaceImage, x1, x2, grad1, grad2, scale, 3);
hold on;
ms = image(meanSurface);
set(ms,'AlphaData',alphaMask);
hold off;
title('Grasp Points', 'FontSize', 20);
subplot(1,4,2);
plot_grasp_arrows(shapeSurfaceImage, x1, x2, grad1, grad2, scale, 4);
hold on;
ms = image(meanSurface);
set(ms,'AlphaData',alphaMask);
hold off;
title('Grasp Arrows', 'FontSize', 20);
subplot(1,4,3);
plot_grasp_lines(shapeSurfaceImage, x1, x2, grad1, grad2, scale, 4);
hold on;
ms = image(meanSurface);
set(ms,'AlphaData',alphaMask);
hold off;
title('Grasp Lines', 'FontSize', 20);
subplot(1,4,4);
plot_grasp_parallel_plate(shapeSurfaceImage, x1, x2, grad1, grad2, scale, 4);
hold on;
ms = image(meanSurface);
set(ms,'AlphaData',alphaMask);
hold off;
title('Grasp Parallel Plates', 'FontSize', 20);

%% grasp plots (quiver version)
x1 = [10; 12];
x2 = [15; 12];
x_grasp = [x1; x2];

figure(40);
imshow(shapeImageSampled);
hold on;
grad1 = [2; 0];
grad2 = [-2; 0];
start1 = x1 - grad1;
start2 = x2 - grad2;

% quiver(scale*start1(1,:), scale*start1(2,:), scale*grad1(1,:), scale*grad1(2,:), 'r', 'LineWidth', 2);
% quiver(scale*start2(1,:), scale*start2(2,:), scale*grad2(1,:), scale*grad2(2,:), 'r', 'LineWidth', 2);

arrow(scale*start1, scale*x1, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 1, 'Width', 1, 'TipAngle', 60);
arrow(scale*start2, scale*x2, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 1, 'Width', 1, 'TipAngle', 60);

%annotation('arrow', [x1(1,:), 1] / (25), [x1(2,:), 1] / (25));
% 
% plot(scale*x1(1,:), scale*x1(2,:), 'rx-', 'MarkerSize', 20, 'LineWidth', 1.5);
% plot(scale*x2(1,:), scale*x2(2,:), 'gx-', 'MarkerSize', 20, 'LineWidth', 1.5);
hold off;

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
varParams.noiseScale = 0.01;
varParams.specularNoise = false;
varParams.sparsityRate = 0.0;

varParams.noiseGradMode = 'None';
varParams.horizScale = 1;
varParams.vertScale = 1;

%%
trainingParams = struct();
trainingParams.activeSetMethod = 'LevelSet';
trainingParams.activeSetSize = 100;
trainingParams.beta = 10;
trainingParams.numIters = 1;
trainingParams.eps = 1e-2;
trainingParams.levelSet = 0;
trainingParams.surfaceThresh = 0.1;
trainingParams.scale = 2;

%%
[shapeParams, shapeImage] = create_tsdf(filename, dataDir, dim, varParams);

%%
[gpModel, predGrid, tsdfReconError, normalError, surfaceImage, selectionTime] = ...
    create_sparse_gpis(shapeParams, trainingParams, trainingParams.scale);

%% find antipodal points
% Set parameters of the optimizer
cfg = struct();
cfg.max_iter = 10;
cfg.max_merit_coeff_increases = 5;
cfg.merit_coeff_increase_ratio = 10;
cfg.initial_penalty_coeff = 0.5;
cfg.initial_trust_box_size = 5;
cfg.trust_shrink_ratio = .75;
cfg.trust_expand_ratio = 2.0;
cfg.min_approx_improve = 1e-8;
cfg.min_trust_box_size = 1e-2;
cfg.callback = @plot_surface_grasp_points;
cfg.full_hessian = true;
cfg.scale = scale;
cfg.surfaceImage = surfaceImage;
cfg.lambda = 0.0;
cfg.nu = 1;
cfg.min_init_dist = 5;

%% get grasp points
[x_surface_i, x_surface_j] = ...
    find(abs(reshape(predGrid.tsdf, predGrid.gridDim, predGrid.gridDim)) < trainingParams.surfaceThresh);
n_surf = size(x_surface_i, 1);
ind1 = uint16(rand * n_surf);
x1 = [x_surface_j(ind1); x_surface_i(ind1)];
found_ind2 = false;
while ~found_ind2
    ind2 = uint16(rand * n_surf);
    x2 = [x_surface_j(ind2); x_surface_i(ind2)];
    if norm(x1 - x2) > cfg.min_init_dist
        found_ind2 = true;
    end
end
x_init = [x1; x2];      
%%
[x_grasp, x_all_iters] = find_antipodal_grasp_points(x_init, gpModel, ...
    cfg, predGrid.gridDim, com);

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

%% Eval antipodal grasp
num_contacts = 2;
fric_coef = 0.5;
cone_angle = atan(fric_coef);
x_start_1 = xs2 + 1.4*(xs1 - xs2);
x_start_2 = xs1 + 1.4*(xs2 - xs1);

cp1 = [x_start_1'; xs2'];
cp2 = [x_start_2'; xs1'];
cp = [cp1; cp2];
[mq, vq, success] = MC_sample(gpModel, predGrid.points, cone_angle, cp, num_contacts, com);
 
%% Eval random grasp
num_contacts = 2;
fric_coef = 0.5;
cone_angle = atan(fric_coef);
[cp1,cp2] = get_random_grasp(predGrid.gridDim);
cp = [cp1; cp2]

[mq, vq, success] = MC_sample(gpModel, predGrid.points, cone_angle, cp, num_contacts, com);
 

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

%% old experiment code


    % get initial grasp point
    [x_surface_i, x_surface_j] = ...
        find(abs(reshape(predGrid.tsdf, predGrid.gridDim, predGrid.gridDim)) < trainingParams.surfaceThresh);
    n_surf = size(x_surface_i, 1);
    
    ind1 = uint16(rand * n_surf + 1);
    x1 = [x_surface_j(ind1); x_surface_i(ind1)];
    found_ind2 = false;
    while ~found_ind2
        ind2 = uint16(rand * n_surf + 1);
        x2 = [x_surface_j(ind2); x_surface_i(ind2)];
        if norm(x1 - x2) > optimizationParams.min_init_dist
            found_ind2 = true;
        end
    end
    x_init = [x1; x2];
    x_start_1 = com' + 2.0*(x1 - com');
    x_start_2 = com' + 2.0*(x2 - com');
    
    % evaluate initial grasp
    cp1 = [x_start_1'; com];
    cp2 = [x_start_2'; com];
    cp = [cp1; cp2];
    
    startTime = tic;
    [mn_q, v_q, success] = mc_sample_fast(predGrid.points, ...
       cone_angle, cp, num_contacts, shapeSamples, dim, ...
       trainingParams.surfaceThresh, 10, false);
    evalTime = toc(startTime);
    fprintf('Evalutated quality on %d samples in %f sec\n', ...
        experimentConfig.numSamples, evalTime);
    
    initialLOA{i} = cp;
    initialMeanQ(i,1) = mn_q;
    initialVarQ(i,1) = v_q;
    
    optimizationParams.surfaceImage = surfaceImage;
    
    % optimize grasp points w/o uncertainty
    disp('Optimizing grasp points w/o uncertainty...');
    startTime = tic;
    [x_grasp, x_all_iters, opt_success] = find_antipodal_grasp_points(x_init, gpModel, ...
        optimizationParams, predGrid.gridDim, com, 0);
    ucTimes(i,:) = toc(startTime);
    ucGrasps(i,:) = x_grasp';
    ucSuccesses(i,:) = opt_success;

    % plot the grasp points
    h = figure(2);
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

    figureName = sprintf('%s\%s_uc_%d.eps', outputDir, filename, i);
    print(h,'-deps', figureName);
    
    % evaluate grasp quality
    disp('Evaluating unconstrained grasp quality');
    x_start_1 = xs2 + 1.4*(xs1 - xs2);
    x_start_2 = xs1 + 1.4*(xs2 - xs1);

    cp1 = [x_start_1'; xs2'];
    cp2 = [x_start_2'; xs1'];
    cp = [cp1; cp2];
    [mn_q, v_q, success] = MC_sample(gpModel, predGrid.points, cone_angle, ...
        cp, num_contacts, com, trainingParams.numSamples, ...
        trainingParams.surfaceThresh);
    ucMeanQ(i,1) = mn_q;
    ucVarQ(i,1) = v_q;
    
    % optimize grasp points with uncertainty
    disp('Optimizing grasp points with uncertainty...');
    startTime = tic;
    [x_grasp, x_all_iters, opt_success] = find_antipodal_grasp_points(x_init, gpModel, ...
        optimizationParams, predGrid.gridDim, com);
    antipodalTimes(i,:) = toc(startTime);
    antipodalGrasps(i,:) = x_grasp';
    antipodalSuccesses(i,:) = opt_success;

    % plot the grasp points
    h = figure(2);
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

    figureName = sprintf('%s\%s_antipodal_%d.eps', outputDir, filename, i);
    print(h,'-deps', figureName)

    % evaluate the quality of the antipodal grasp with uncertainty
    disp('Evaluating grasp quality');
    x_start_1 = xs2 + 1.4*(xs1 - xs2);
    x_start_2 = xs1 + 1.4*(xs2 - xs1);

    cp1 = [x_start_1'; xs2'];
    cp2 = [x_start_2'; xs1'];
    cp = [cp1; cp2];
    [mn_q, v_q, success] = MC_sample(gpModel, predGrid.points, cone_angle, ...
        cp, num_contacts, com, trainingParams.numSamples, ...
        trainingParams.surfaceThresh);
    
    if ~success
        disp('Bad contacts. Retrying...');
        continue;
    end

    antipodalMeanQ(i,1) = mn_q;
    antipodalVarQ(i,1) = v_q;
    i = i+1;
end

% evaluate random grasps
randomLOA  = cell(experimentConfig.graspIters, 1);
randomMeanQ = zeros(experimentConfig.graspIters, 1);
randomVarQ = zeros(experimentConfig.graspIters, 1);

% i = 1;
% while i <= experimentConfig.graspIters
%     fprintf('Selecting random pair %d\n', i);
% 
%     [cp1,cp2] = get_random_grasp(predGrid.gridDim);
%     cp = [cp1; cp2];
%     randomLOA{i} = [cp1, cp2];
% 
%     [mn_q, v_q, success] = MC_sample(gpModel, predGrid.points, cone_angle, ...
%         cp, num_contacts, com, trainingParams.numSamples, ...
%         trainingParams.surfaceThresh);
%     
%     if ~success
%         disp('Bad contacts. Retrying...');
%         continue;
%     end
% 
%     randomMeanQ(i,1) = mn_q;
%     randomVarQ(i,1) = v_q;
%     i = i+1;
% end
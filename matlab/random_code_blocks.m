% Random things I like to do at the end of gpid_2d but they're not
% necessary

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
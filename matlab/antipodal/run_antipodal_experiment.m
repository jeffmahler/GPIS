function [experimentResults, gpModel, shapeParams] = ...
    run_antipodal_experiment(dim, filename, dataDir, outputDir, newShape, ...
                             experimentConfig, varParams, trainingParams, ...
                             optimizationParams)
%RUN_ANTIPODAL_EXPERIMENT Summary of this function goes here
%   Detailed explanation goes here

% create a new shape
if newShape
    [points, com] = new_shape(filename, dataDir, dim);
end

% create the tsdf
[shapeParams, shapeImage, points, com] =...
    create_tsdf(filename, dataDir, dim, varParams);

% construct a gpis either from the full matrix or otherwise
if strcmp(trainingParams.activeSetMethod, 'Full') == 1

    startTime = tic;
    gpModel = create_gpis(shapeParams.points, shapeParams.tsdf, ...
        shapeParams.normals, shapeParams.noise, true, ...
        trainingParams.hyp, trainingParams.numIters);
    constructionTime = toc(startTime);
   
    [predGrid, predSurface] = predict_2d_grid(gpModel, shapeParams.gridDim,...
        trainingParams.surfaceThresh);

    tsdfReconError = ...
        evaluate_errors(predGrid.tsdf, shapeParams.fullTsdf, activeSetSize);
    normalError = ...
        evaluate_errors(predGrid.normals, shapeParams.fullNormals, activeSetSize);

else
    [gpModel, predGrid, tsdfReconError, normalError, surfaceImage, constructionTime] = ...
        create_sparse_gpis(shapeParams, trainingParams, trainingParams.scale);
end
fprintf('Construction took %d seconds\n', constructionTime);


% optimize for antipodal grasp points many times and evaluate quality
antipodalGrasps = zeros(experimentConfig.graspIters, 4);
antipodalMeanQ = zeros(experimentConfig.graspIters, 1);
antipodalVarQ = zeros(experimentConfig.graspIters, 1);
antipodalTimes = zeros(experimentConfig.graspIters, 1);

num_contacts = 2;
d = 2; % dimension of input
cone_angle = atan(experimentConfig.frictionCoef);
scale = optimizationParams.scale;  

i = 1;
while i <= experimentConfig.graspIters
    fprintf('Selecting antipodal pair %d\n', i);

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
    optimizationParams.surfaceImage = surfaceImage;

    % optimize grasp points
    disp('Optimizing grasp points...');
    startTime = tic;
    [x_grasp, x_all_iters] = find_antipodal_grasp_points(x_init, gpModel, ...
        optimizationParams, predGrid.gridDim, com);
    antipodalTimes(i,:) = toc(startTime);
    antipodalGrasps(i,:) = x_grasp';

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

    % evaluate the quality of the antipodal grasp
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

i = 1;
while i <= experimentConfig.graspIters
    fprintf('Selecting random pair %d\n', i);

    [cp1,cp2] = get_random_grasp(predGrid.gridDim);
    cp = [cp1; cp2];
    randomLOA{i} = [cp1, cp2];

    [mn_q, v_q, success] = MC_sample(gpModel, predGrid.points, cone_angle, ...
        cp, num_contacts, com, trainingParams.numSamples, ...
        trainingParams.surfaceThresh);
    
    if ~success
        disp('Bad contacts. Retrying...');
        continue;
    end

    randomMeanQ(i,1) = mn_q;
    randomVarQ(i,1) = v_q;
    i = i+1;
end

% save results and return
experimentResults = struct();

experimentResults.tsdfReconError = tsdfReconError;
experimentResults.normalError = normalError;
experimentResults.constructionTime = constructionTime;

experimentResults.antipodalGrasps = antipodalGrasps;
experimentResults.antipodalMeanQ = antipodalMeanQ;
experimentResults.antipodalVarQ = antipodalVarQ;
experimentResults.antipodalTimes = antipodalTimes;

experimentResults.randomLOA = randomLOA;
experimentResults.randomMeanQ = randomMeanQ;
experimentResults.randomVarQ = randomVarQ;

function [experimentResults, gpModel, shapeParams, shapeSamples] = ...
    run_antipodal_experiment(dim, filename, dataDir, outputDir, newShape, ...
                             experimentConfig, varParams, trainingParams, ...
                             optimizationParams, createGpis)
%RUN_ANTIPODAL_EXPERIMENT runs experiments for antipodal grasping with GPIS

if createGpis
    % create a new gpis experiment model and save
    [gpModel, shapeParams, shapeSamples, constructionResults] = ...
        create_experiment_object(dim, filename, dataDir, newShape, ...
                                 experimentConfig, varParams, trainingParams);
                             
    fprintf('Construction took %f seconds\n', constructionResults.constructionTime);
else
   % load existing gpis experiment model
   [gpModel, shapeParams, shapeSamples, constructionResults] = ...
        load_experiment_object(filename, dataDir);
end

% optimize for antipodal grasp points many times and evaluate quality
initialLOA  = cell(experimentConfig.graspIters, 1);
initialMeanQ = zeros(experimentConfig.graspIters, 1);
initialVarQ = zeros(experimentConfig.graspIters, 1);

ucGrasps = zeros(experimentConfig.graspIters, 4);
ucMeanQ = zeros(experimentConfig.graspIters, 1);
ucVarQ = zeros(experimentConfig.graspIters, 1);
ucTimes = zeros(experimentConfig.graspIters, 1);
ucSuccesses = zeros(experimentConfig.graspIters, 1);

antipodalGrasps = zeros(experimentConfig.graspIters, 4);
antipodalMeanQ = zeros(experimentConfig.graspIters, 1);
antipodalVarQ = zeros(experimentConfig.graspIters, 1);
antipodalTimes = zeros(experimentConfig.graspIters, 1);
antipodalSuccesses = zeros(experimentConfig.graspIters, 1);

num_contacts = 2;
d = 2; % dimension of input
cone_angle = atan(experimentConfig.frictionCoef);
scale = optimizationParams.scale;  
predGrid = constructionResults.predGrid;
com = shapeParams.com;

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

% save results and return
experimentResults = struct();

experimentResults.constructionResults = constructionResults;

experimentResults.initialLOA = initialLOA;
experimentResults.initialMeanQ = initialMeanQ;
experimentResults.initialVarQ = initialVarQ;

experimentResults.ucGrasps = ucGrasps;
experimentResults.ucMeanQ = ucMeanQ;
experimentResults.ucVarQ = ucVarQ;
experimentResults.ucTimes = ucTimes;
experimentResults.ucSuccesses = ucSuccesses;

experimentResults.antipodalGrasps = antipodalGrasps;
experimentResults.antipodalMeanQ = antipodalMeanQ;
experimentResults.antipodalVarQ = antipodalVarQ;
experimentResults.antipodalTimes = antipodalTimes;
experimentResults.antipodalSuccesses = antipodalSuccesses;

experimentResults.randomLOA = randomLOA;
experimentResults.randomMeanQ = randomMeanQ;
experimentResults.randomVarQ = randomVarQ;

function [experimentResults, gpModel, shapeParams, shapeSamples] = ...
    run_antipodal_experiment(dim, filename, dataDir, outputDir, newShape, ...
                             experimentConfig, varParams, trainingParams, ...
                             optimizationParams, createGpis)
%RUN_ANTIPODAL_EXPERIMENT runs experiments for antipodal grasping with GPIS

if createGpis
    % create a new gpis experiment model and save
    [gpModel, shapeParams, shapeSamples, constructionResults] = ...
        create_experiment_object(dim, filename, dataDir, newShape, ...
                                 experimentConfig, varParams, trainingParams, ...
                                 optimizationParams.scale);
                             
    fprintf('Construction took %f seconds\n', constructionResults.constructionTime);
else
   % load existing gpis experiment model
   [gpModel, shapeParams, shapeSamples, constructionResults] = ...
        load_experiment_object(filename, dataDir, optimizationParams.scale);
end

% optimize for antipodal grasp points many times and evaluate quality
initialGrasps  = zeros(experimentConfig.graspIters, 4);
initialMeanQ = zeros(experimentConfig.graspIters, 1);
initialVarQ = zeros(experimentConfig.graspIters, 1);
initialSuccesses = zeros(experimentConfig.graspIters, 1);
initialPfc = zeros(experimentConfig.graspIters, 1);

ucGrasps = zeros(experimentConfig.graspIters, 4);
ucOpt = zeros(experimentConfig.graspIters, 1);
ucMeanQ = zeros(experimentConfig.graspIters, 1);
ucVarQ = zeros(experimentConfig.graspIters, 1);
ucTimes = zeros(experimentConfig.graspIters, 1);
ucSatisfy = zeros(experimentConfig.graspIters, 1);
ucSuccesses = zeros(experimentConfig.graspIters, 1);
ucPfc = zeros(experimentConfig.graspIters, 1);

antipodalGrasps = zeros(experimentConfig.graspIters, 4);
antipodalOpt = zeros(experimentConfig.graspIters, 1);
antipodalMeanQ = zeros(experimentConfig.graspIters, 1);
antipodalVarQ = zeros(experimentConfig.graspIters, 1);
antipodalTimes = zeros(experimentConfig.graspIters, 1);
antipodalSatisfy= zeros(experimentConfig.graspIters, 1);
antipodalSuccesses = zeros(experimentConfig.graspIters, 1);
antipodalPfc = zeros(experimentConfig.graspIters, 1);

detMeanGrasps = zeros(experimentConfig.graspIters, 4);
detMeanOpt = zeros(experimentConfig.graspIters, 1);
detMeanMeanQ = zeros(experimentConfig.graspIters, 1);
detMeanVarQ = zeros(experimentConfig.graspIters, 1);
detMeanTimes = zeros(experimentConfig.graspIters, 1);
detMeanSatisfy= zeros(experimentConfig.graspIters, 1);
detMeanSuccesses = zeros(experimentConfig.graspIters, 1);
detMeanPfc = zeros(experimentConfig.graspIters, 1);

ucMeanGrasps = zeros(experimentConfig.graspIters, 4);
ucMeanOpt = zeros(experimentConfig.graspIters, 1);
ucMeanMeanQ = zeros(experimentConfig.graspIters, 1);
ucMeanVarQ = zeros(experimentConfig.graspIters, 1);
ucMeanTimes = zeros(experimentConfig.graspIters, 1);
ucMeanSatisfy= zeros(experimentConfig.graspIters, 1);
ucMeanSuccesses = zeros(experimentConfig.graspIters, 1);
ucMeanPfc = zeros(experimentConfig.graspIters, 1);

detMeanUcGrasps = zeros(experimentConfig.graspIters, 4);
detMeanUcOpt = zeros(experimentConfig.graspIters, 1);
detMeanUcMeanQ = zeros(experimentConfig.graspIters, 1);
detMeanUcVarQ = zeros(experimentConfig.graspIters, 1);
detMeanUcTimes = zeros(experimentConfig.graspIters, 1);
detMeanUcSatisfy= zeros(experimentConfig.graspIters, 1);
detMeanUcSuccesses = zeros(experimentConfig.graspIters, 1);
detMeanUcPfc = zeros(experimentConfig.graspIters, 1);

ucMeanUcGrasps = zeros(experimentConfig.graspIters, 4);
ucMeanUcOpt = zeros(experimentConfig.graspIters, 1);
ucMeanUcMeanQ = zeros(experimentConfig.graspIters, 1);
ucMeanUcVarQ = zeros(experimentConfig.graspIters, 1);
ucMeanUcTimes = zeros(experimentConfig.graspIters, 1);
ucMeanUcSatisfy= zeros(experimentConfig.graspIters, 1);
ucMeanUcSuccesses = zeros(experimentConfig.graspIters, 1);
ucMeanUcPfc = zeros(experimentConfig.graspIters, 1);

randomFcGrasps = zeros(experimentConfig.graspIters, 4);
randomFcMeanQ = zeros(experimentConfig.graspIters, 1);
randomFcVarQ = zeros(experimentConfig.graspIters, 1);
randomFcSuccesses = zeros(experimentConfig.graspIters, 1);
randomFcEstQ = zeros(experimentConfig.graspIters, 1);
randomFcNumAttempts = zeros(experimentConfig.graspIters, 1);
randomFcPfc = zeros(experimentConfig.graspIters, 1);

randomSampleFcGrasps = zeros(experimentConfig.graspIters, 4);
randomSampleFcMeanQ = zeros(experimentConfig.graspIters, 1);
randomSampleFcVarQ = zeros(experimentConfig.graspIters, 1);
randomSampleFcSuccesses = zeros(experimentConfig.graspIters, 1);
randomSampleFcNumAttempts = zeros(experimentConfig.graspIters, 1);
randomSampleFcPfc = zeros(experimentConfig.graspIters, 1);

numContacts = 2;
coneAngle = atan(experimentConfig.frictionCoef);
length = experimentConfig.arrowLength;
surfaceImage = constructionResults.newSurfaceImage;
scale = optimizationParams.scale;  
predGrid = constructionResults.predGrid;
useNormsForInit = false;
numSamples = size(shapeSamples, 2);
com = shapeParams.com;

if experimentConfig.visOptimization
    optimizationParams.surfaceImage = surfaceImage;
end

i = 1;
while i <= experimentConfig.graspIters
    fprintf('Selecting antipodal pair %d\n', i);

    disp('Evaluating initial grasp without uncertainty');
    initGrasp = get_initial_antipodal_grasp(predGrid, useNormsForInit);
    loa = create_ap_loa(initGrasp, experimentConfig.gripWidth);
    graspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);
    
    % evaluate success of initial grasp for reference
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);
    initialGrasps(i,:) = initGrasp';
    initialMeanQ(i) = mn_q;
    initialVarQ(i) = v_q;
    initialSuccesses(i) = success;
    initialPfc(i) = p_fc;

    % evaluate unconstrained grasp (if specified)
    % this means it will solve an antipodal feasibility optimization
    % problem without a measure of uncertainty
    if experimentConfig.evalUcGrasps
        disp('Evaluating antipodal grasp without uncertainty');
        % optimization times
        startTime = tic;
        useUncertainty = 0;
        [optGrasp, optVal, allIters, constSatisfaction] = ...
            find_antipodal_grasp_points(initGrasp, gpModel, ...
                optimizationParams, shapeParams.gridDim, shapeParams.com, ...
                useUncertainty);
        optimizationTime = toc(startTime);

        ucTimes(i) = optimizationTime;
        ucSatisfy(i) = constSatisfaction;
        ucOpt(i) = optVal;

        % evaluate quality with MC sampling
        [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);
        ucGrasps(i,:) = optGrasp';
        ucMeanQ(i) = mn_q;
        ucVarQ(i) = v_q;
        ucSuccesses(i) = success;
        ucPfc(i) = p_fc;
    end

    % evaluate antipodal optimization w/ uncertainty
    % optimize the antipodal grasps
    disp('Evaluating antipodal optimized grasp w/ uncertainty');
    startTime = tic;
    [optGrasp, optVal,  allIters, constSatisfaction] = ...
        find_antipodal_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com);
    optimizationTime = toc(startTime);

    antipodalTimes(i) = optimizationTime;
    antipodalSatisfy(i) = constSatisfaction;
    antipodalOpt(i) = optVal;

    % evaluate quality with MC sampling
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);
    if ~success && experimentConfig.rejectUnsuccessfulGrasps
        disp('Bad contacts on antipodal grasps. Retrying...');
        continue;
    end

    antipodalGrasps(i,:) = optGrasp';
    antipodalMeanQ(i) = mn_q;
    antipodalVarQ(i) = v_q;
    antipodalSuccesses(i) = success;
    antipodalPfc(i) = p_fc;
    
    % evaluate antipodal optimization of just the FC metric
    disp('Evaluating FC optimized grasp');
    useUncertainty = 0;
    startTime = tic;
    [optGrasp, optVal, allIters, constSatisfaction] = ...
        find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com, ...
            predGrid, coneAngle, experimentConfig.numBadContacts, useUncertainty);
    optimizationTime = toc(startTime);

    detMeanTimes(i) = optimizationTime;
    detMeanSatisfy(i) = constSatisfaction;
    detMeanOpt(i) = optVal;

    % evaluate quality with MC sampling
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);

    detMeanGrasps(i,:) = optGrasp';
    detMeanMeanQ(i) = mn_q;
    detMeanVarQ(i) = v_q;
    detMeanSuccesses(i) = success;
    detMeanPfc(i) = p_fc;
    
    % evaluate antipodal optimization w/ uncertainty and FC metric
    disp('Evaluating FC optimized grasp w/ uncertainty');
    startTime = tic;
    [optGrasp, optVal, allIters, constSatisfaction] = ...
        find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com, ...
            predGrid, coneAngle, experimentConfig.numBadContacts);
    optimizationTime = toc(startTime);

    ucMeanTimes(i) = optimizationTime;
    ucMeanSatisfy(i) = constSatisfaction;
    ucMeanOpt(i) = optVal;

    % evaluate quality with MC sampling
    bestLoa = create_ap_loa(initGrasp, experimentConfig.gripWidth);
    bestGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, bestGraspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);

    ucMeanGrasps(i,:) = optGrasp';
    ucMeanMeanQ(i) = mn_q;
    ucMeanVarQ(i) = v_q;
    ucMeanSuccesses(i) = success;
    ucMeanPfc(i) = p_fc;
    
    % evaluate antipodal optimization of just the FC metric
    disp('Evaluating FC optimized grasp without antipodal constraint');
    useUncertainty = 0;
    forceAp = false;
    startTime = tic;
    [optGrasp, optVal, allIters, constSatisfaction] = ...
        find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com, ...
            predGrid, coneAngle, experimentConfig.numBadContacts, useUncertainty, forceAp);
    optimizationTime = toc(startTime);

    detMeanUcTimes(i) = optimizationTime;
    detMeanUcSatisfy(i) = constSatisfaction;
    detMeanUcOpt(i) = optVal;

    % evaluate quality with MC sampling
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);

    detMeanUcGrasps(i,:) = optGrasp';
    detMeanUcMeanQ(i) = mn_q;
    detMeanUcVarQ(i) = v_q;
    detMeanUcSuccesses(i) = success;
    detMeanUcPfc(i) = p_fc;

    % evaluate antipodal optimization of just the FC metric
    disp('Evaluating FC optimized grasp with uncertainty without antipodal constraint');
    useUncertainty = 1;
    forceAp = false;
    startTime = tic;
    [optGrasp, optVal, allIters, constSatisfaction] = ...
        find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com, ...
            predGrid, coneAngle, experimentConfig.numBadContacts, useUncertainty, forceAp);
    optimizationTime = toc(startTime);

    ucMeanUcTimes(i) = optimizationTime;
    ucMeanUcSatisfy(i) = constSatisfaction;
    ucMeanUcOpt(i) = optVal;

    % evaluate quality with MC sampling
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);

    ucMeanUcGrasps(i,:) = optGrasp';
    ucMeanUcMeanQ(i) = mn_q;
    ucMeanUcVarQ(i) = v_q;
    ucMeanUcSuccesses(i) = success;
    ucMeanUcPfc(i) = p_fc;
    
    % evaluate the quality of randomly selected grasps on the mean shape 
    disp('Evaluating random grasps on FC for mean shape');
    attemptedGrasps = [];
    fcQ = [];
    k = 0;
    startTime = tic;
    duration = 0;

    while duration < ucMeanTimes(i) || k < 1
        % get random grasp
        randGrasp = get_initial_antipodal_grasp(predGrid, useNormsForInit);
        randLoa = create_ap_loa(randGrasp, experimentConfig.gripWidth);
            
        % evaluate FC on mean shape
        meanSample = {predGrid};
        graspSample = {randLoa};
        [mn_q, v_q, success, p_fc] = ...
            mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSample, numContacts, ...
                                    meanSample, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);

        % add quality to running count
        if success
            attemptedGrasps = [attemptedGrasps; randGrasp'];
            fcQ = [fcQ; mn_q];
            k = k+1;
        end

        duration = toc(startTime);
    end

    % choose the grasp with maximum FC quality and evaluate
    bestGraspIndices = find(fcQ == max(fcQ));
    randomFcGrasps(i,:) = attemptedGrasps(bestGraspIndices(1), :);
    randomFcEstQ(i) = fcQ(bestGraspIndices(1));
    randomFcNumAttempts(i) = k;

    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);
    randomFcMeanQ(i) = mn_q;
    randomFcVarQ(i) = v_q;
    randomFcSuccesses(i) = success;
    randomFcPfc(i) = p_fc;

    % evaluate the quality of randomly selected grasps on the samples
    if experimentConfig.evalRandSampleFcGrasps
        disp('Evaluating random grasps on sampled FC');
        startTime = tic;
        duration = 0;
        attemptedGrasps = [];
        fcQ = [];
        fcVarQ = [];
        fcP = [];
        k = 0;

        while duration < ucMeanTimes(i) || k < 1
            % get random grasp
            randGrasp = get_initial_antipodal_grasp(predGrid, useNormsForInit);
            randLoa = create_ap_loa(randGrasp, experimentConfig.gripWidth);
            randGraspSamples = sample_grasps(randLoa, experimentConfig.graspSigma, numSamples);
            
            % evaluate FC on mean shape
            [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, randGraspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    experimentConfig.plateWidth, ...
                                    experimentConfig.visSampling);

            % add quality to running count
            if success
                attemptedGrasps = [attemptedGrasps; randGrasp'];
                fcQ = [fcQ; mn_q];
                fcVarQ = [fcVarQ; v_q];
                fcP = [fcP; p_fc];
                k = k+1;
            end

            duration = toc(startTime);
        end
        
        % choose the grasp with maximum FC quality and evaluate
        bestGraspIndices = find(fcQ == max(fcQ));
        randomSampleFcGrasps(i,:) = attemptedGrasps(bestGraspIndices(1), :);
        randomSampleFcMeanQ(i) = fcQ(bestGraspIndices(1));
        randomSampleFcVarQ(i) = fcVarQ(bestGraspIndices(1));
        randomSampleFcPfc(i) = fcP(bestGraspIndices(1));
        randomSampleFcSuccesses(i) = true; % technically they're successful if saved
        randomSampleFcNumAttempts(i) = k;

        % TODO: remove this (its for vis only)
        loa = create_ap_loa(randomSampleFcGrasps(i,:)', experimentConfig.loaScale);
        [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                        coneAngle, loa, numContacts, ...
                                        shapeSamples, shapeParams.gridDim, ...
                                        shapeParams.surfaceThresh, ...
                                        experimentConfig.numBadContacts, ...
                                        experimentConfig.visSampling);
    end

    % alternate visualization of all grasps
    if experimentConfig.visGrasps && experimentConfig.evalRandSampleFcGrasps && ...
            experimentConfig.evalUcGrasps && false % don't run this anymore...
        figure(31);
        subplot(1,5,1);
        visualize_grasp(initGrasp, predGrid, surfaceImage, scale, length);
        title('Initial Grasp', 'FontSize', 15);
        xlabel(sprintf('Q = %f', initialMeanQ(i)), 'FontSize', 15);
        subplot(1,5,2);
        visualize_grasp(randomFcGrasps(i,:)', predGrid, surfaceImage, scale, length);
        title('Sampled Best Grasp on Mean Shape', 'FontSize', 15);
        xlabel(sprintf('Q = %f', randomFcMeanQ(i)), 'FontSize', 15);
        subplot(1,5,3);
        visualize_grasp(randomSampleFcGrasps(i,:)', predGrid, surfaceImage, scale, length);
        title('Sampled Best Grasp on Samples', 'FontSize', 15);
        xlabel(sprintf('Q = %f', randomSampleFcMeanQ(i)), 'FontSize', 15);
        subplot(1,5,4);
        visualize_grasp(ucGrasps(i,:)', predGrid, surfaceImage, scale, length);
        title('Antipodal Feasible Grasp', 'FontSize', 15);
        xlabel(sprintf('Q = %f', ucMeanQ(i)), 'FontSize', 15);
        subplot(1,5,5);
        visualize_grasp(antipodalGrasps(i,:)', predGrid, surfaceImage, scale, length);
        title('Optimized Grasp', 'FontSize', 15);
        xlabel(sprintf('Q = %f', antipodalMeanQ(i)), 'FontSize', 15);
    elseif experimentConfig.visGrasps
        % visualize base config
        figure(31);
        subplot(1,4,1);
        visualize_grasp(initGrasp, predGrid, surfaceImage, scale, length);
        title('Initial Grasp', 'FontSize', 15);
        xlabel(sprintf('Q = %f', initialMeanQ(i)), 'FontSize', 15);
        subplot(1,4,2);
        visualize_grasp(randomFcGrasps(i,:)', predGrid, surfaceImage, scale, length);
        title('Sampled Best Grasp on Mean Shape', 'FontSize', 15);
        xlabel(sprintf('Q = %f', randomFcMeanQ(i)), 'FontSize', 15);
        subplot(1,4,3);
        visualize_grasp(antipodalGrasps(i,:)', predGrid, surfaceImage, scale, length);
        title('Antipodal Optimized Grasp', 'FontSize', 15);
        xlabel(sprintf('Q = %f', antipodalMeanQ(i)), 'FontSize', 15);
        subplot(1,4,4);
        visualize_grasp(ucMeanGrasps(i,:)', predGrid, surfaceImage, scale, length);
        title('FC Mean Optimized Grasp', 'FontSize', 15);
        xlabel(sprintf('Q = %f', ucMeanMeanQ(i)), 'FontSize', 15);
    end

    i = i+1;
end

% create results structure
experimentResults = struct();

experimentResults.constructionResults = constructionResults;

experimentResults.initialGraspResults = ...
    create_grasp_results_struct(initialGrasps, initialMeanQ, initialVarQ, ...
                                initialSuccesses, [], [], [], [], [], initialPfc);
experimentResults.ucGraspResults = ...
    create_grasp_results_struct(ucGrasps, ucMeanQ, ucVarQ, ...
                                ucSuccesses, ucSatisfy, ucTimes, [], ...
                                ucOpt, [], ucPfc);

experimentResults.antipodalGraspResults = ...
    create_grasp_results_struct(antipodalGrasps, antipodalMeanQ, antipodalVarQ, ...
                                antipodalSuccesses, antipodalSatisfy, antipodalTimes, ...
                                [], antipodalOpt, [], antipodalPfc);


experimentResults.detMeanGraspResults = ...
    create_grasp_results_struct(detMeanGrasps, detMeanMeanQ, detMeanVarQ, ...
                                detMeanSuccesses, detMeanSatisfy, detMeanTimes, ...
                                [], detMeanOpt, [], detMeanPfc);

experimentResults.ucMeanGraspResults = ...
    create_grasp_results_struct(ucMeanGrasps, ucMeanMeanQ, ucMeanVarQ, ...
                                ucMeanSuccesses, ucMeanSatisfy, ucMeanTimes, ...
                                [], ucMeanOpt, [], ucMeanPfc);

experimentResults.detMeanUcGraspResults = ...
    create_grasp_results_struct(detMeanUcGrasps, detMeanUcMeanQ, detMeanUcVarQ, ...
                                detMeanUcSuccesses, detMeanUcSatisfy, detMeanUcTimes, ...
                                [], detMeanUcOpt, [], detMeanPfc);

experimentResults.ucMeanUcGraspResults = ...
    create_grasp_results_struct(ucMeanUcGrasps, ucMeanUcMeanQ, ucMeanUcVarQ, ...
                                ucMeanUcSuccesses, ucMeanUcSatisfy, ucMeanUcTimes, ...
                                [], ucMeanUcOpt, [], ucMeanPfc);                        
                            
experimentResults.randomFcGraspResults = ...
    create_grasp_results_struct(randomFcGrasps, randomFcMeanQ, randomFcVarQ, ...
                                randomFcSuccesses, [], [], randomFcNumAttempts, ...
                                [], randomFcEstQ, randomFcPfc);

experimentResults.randomSampleFcGraspResults = ...
    create_grasp_results_struct(randomSampleFcGrasps, randomSampleFcMeanQ, randomSampleFcVarQ, ...
                                randomSampleFcSuccesses, [], [], randomSampleFcNumAttempts, ...
                                [], [], randomSampleFcPfc);

% save with hash
experimentHash = uint32(1e6 * rand());
experimentFilename = sprintf('%s/experiment_%d.mat', outputDir, experimentHash);
experimentResults.id = experimentHash;
if experimentConfig.graspIters > 0
    save(experimentFilename, 'experimentResults');
end

fprintf('Experiment %d complete\n', experimentHash);

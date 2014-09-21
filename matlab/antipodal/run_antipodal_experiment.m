function [experimentResults, gpModel, shapeParams, shapeSamples] = ...
    run_antipodal_experiment(dim, filename, gripScale, dataDir, outputDir, meanCompDir, ...
                             newShape, experimentConfig, varParams, trainingParams, ...
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
initialNomQ = zeros(experimentConfig.graspIters, 1);
initialNomP = zeros(experimentConfig.graspIters, 1);

detFcOptGrasps = zeros(experimentConfig.graspIters, 4);
detFcOptVals = zeros(experimentConfig.graspIters, 1);
detFcOptMeanQ = zeros(experimentConfig.graspIters, 1);
detFcOptVarQ = zeros(experimentConfig.graspIters, 1);
detFcOptTimes = zeros(experimentConfig.graspIters, 1);
detFcOptSatisfy= zeros(experimentConfig.graspIters, 1);
detFcOptSuccesses = zeros(experimentConfig.graspIters, 1);
detFcOptPfc = zeros(experimentConfig.graspIters, 1);
detFcOptNomQ = zeros(experimentConfig.graspIters, 1);
detFcOptNomP = zeros(experimentConfig.graspIters, 1);

ucFcOptGrasps = zeros(experimentConfig.graspIters, 4);
ucFcOptVals = zeros(experimentConfig.graspIters, 1);
ucFcOptMeanQ = zeros(experimentConfig.graspIters, 1);
ucFcOptVarQ = zeros(experimentConfig.graspIters, 1);
ucFcOptTimes = zeros(experimentConfig.graspIters, 1);
ucFcOptSatisfy= zeros(experimentConfig.graspIters, 1);
ucFcOptSuccesses = zeros(experimentConfig.graspIters, 1);
ucFcOptPfc = zeros(experimentConfig.graspIters, 1);
ucFcOptNomQ = zeros(experimentConfig.graspIters, 1);
ucFcOptNomP = zeros(experimentConfig.graspIters, 1);

% cf = constraint free
cfDetFcOptGrasps = zeros(experimentConfig.graspIters, 4);
cfDetFcOptVals = zeros(experimentConfig.graspIters, 1);
cfDetFcOptMeanQ = zeros(experimentConfig.graspIters, 1);
cfDetFcOptVarQ = zeros(experimentConfig.graspIters, 1);
cfDetFcOptTimes = zeros(experimentConfig.graspIters, 1);
cfDetFcOptSatisfy= zeros(experimentConfig.graspIters, 1);
cfDetFcOptSuccesses = zeros(experimentConfig.graspIters, 1);
cfDetFcOptPfc = zeros(experimentConfig.graspIters, 1);
cfDetFcOptNomQ = zeros(experimentConfig.graspIters, 1);
cfDetFcOptNomP = zeros(experimentConfig.graspIters, 1);

cfUcFcOptGrasps = zeros(experimentConfig.graspIters, 4);
cfUcFcOptVals = zeros(experimentConfig.graspIters, 1);
cfUcFcOptMeanQ = zeros(experimentConfig.graspIters, 1);
cfUcFcOptVarQ = zeros(experimentConfig.graspIters, 1);
cfUcFcOptTimes = zeros(experimentConfig.graspIters, 1);
cfUcFcOptSatisfy= zeros(experimentConfig.graspIters, 1);
cfUcFcOptSuccesses = zeros(experimentConfig.graspIters, 1);
cfUcFcOptPfc = zeros(experimentConfig.graspIters, 1);
cfUcFcOptNomQ = zeros(experimentConfig.graspIters, 1);
cfUcFcOptNomP = zeros(experimentConfig.graspIters, 1);

numContacts = 2;
coneAngle = atan(experimentConfig.frictionCoef);
length = experimentConfig.arrowLength;
surfaceImage = constructionResults.newSurfaceImage;
scale = optimizationParams.scale;
win = experimentConfig.smoothWin;
sig = experimentConfig.smoothSig;
predGrid = constructionResults.predGrid;
useNormsForInit = false;
numSamples = size(shapeSamples, 2);
com = shapeParams.com;

% create struct for nominal shape
nominalShape = struct();
nominalShape.tsdf = shapeParams.fullTsdf;
nominalShape.normals = shapeParams.fullNormals;
nominalShape.points = shapeParams.all_points;
nominalShape.noise = zeros(size(nominalShape.tsdf,1), 1);
nominalShape.gridDim = shapeParams.gridDim;
nominalShape.surfaceThresh = shapeParams.surfaceThresh;
nominalShape.com = shapeParams.com;

tsdfGrid = reshape(nominalShape.tsdf, [nominalShape.gridDim, nominalShape.gridDim,]);
nominalShape.shapeImage = high_res_tsdf(tsdfGrid, scale, win, sig);

% figure;
% imshow(nominalShape.shapeImage);

graspSigma = experimentConfig.graspSigma;
gripWidth = gripScale * experimentConfig.objScale * dim;
plateWidth = gripWidth * experimentConfig.plateScale;
plateWidth = uint16(round(plateWidth));
optimizationParams.grip_width = gripWidth;

nomShapeSamples = cell(1, numSamples);
for k = 1:numSamples
    nomShapeSamples{k} = nominalShape;
end

if experimentConfig.visOptimization
    optimizationParams.surfaceImage = surfaceImage;
end

if experimentConfig.graspIters < 1
    experimentResults = struct();
    experimentResults.constructionResults = constructionResults;
    return;
end

i = 1;
while i <= experimentConfig.graspIters
    fprintf('Selecting antipodal pair %d\n', i);

    disp('Evaluating initial grasp without uncertainty');
    initGrasp = get_initial_antipodal_grasp(predGrid, useNormsForInit);
    
    % evaluate success of initial grasp for reference
    loa = create_ap_loa(initGrasp, experimentConfig.gripWidth);
    graspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);
    initialGrasps(i,:) = initGrasp';
    initialMeanQ(i) = mn_q;
    initialVarQ(i) = v_q;
    initialSuccesses(i) = success;
    initialPfc(i) = p_fc;

    % evaluate quality on nominal shape
    nomGraspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, nomGraspSamples, numContacts, ...
                                    nomShapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);

    initialNomQ(i) = mn_q';
    initialNomP(i) = p_fc;
    
    % evaluate antipodal optimization of just the FC metric
    disp('Evaluating FC optimized grasp');
    useUncertainty = false;
    startTime = tic;
    [optGrasp, optVal, allIters, constSatisfaction] = ...
        find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com, ...
            predGrid, coneAngle, experimentConfig.numBadContacts, ...
            plateWidth, gripWidth, graspSigma, useUncertainty);
    optimizationTime = toc(startTime);

    detFcOptTimes(i) = optimizationTime;
    detFcOptSatisfy(i) = constSatisfaction;
    detFcOptVals(i) = optVal;

    % evaluate quality with MC sampling
    bestLoa = create_ap_loa(optGrasp, experimentConfig.gripWidth);
    bestGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, bestGraspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);

    detFcOptGrasps(i,:) = optGrasp';
    detFcOptMeanQ(i) = mn_q;
    detFcOptVarQ(i) = v_q;
    detFcOptSuccesses(i) = success;
    detFcOptPfc(i) = p_fc;
    
    % evaluate quality on nominal shape
    nomGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, nomGraspSamples, numContacts, ...
                                    nomShapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);

    detFcOptNomQ(i) = mn_q';
    detFcOptNomP(i) = p_fc;
    
    % evaluate antipodal optimization w/ uncertainty and FC metric
    disp('Evaluating FC optimized grasp w/ uncertainty');
    startTime = tic;
    [optGrasp, optVal, allIters, constSatisfaction] = ...
        find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com, ...
            predGrid, coneAngle, experimentConfig.numBadContacts, ...
            plateWidth, gripWidth, graspSigma);
    ucFcOptTimes(i) = optimizationTime;
    ucFcOptSatisfy(i) = constSatisfaction;
    ucFcOptVals(i) = optVal;

    % evaluate quality with MC sampling
    bestLoa = create_ap_loa(optGrasp, experimentConfig.gripWidth);
    bestGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, bestGraspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);

    ucFcOptGrasps(i,:) = optGrasp';
    ucFcOptMeanQ(i) = mn_q;
    ucFcOptVarQ(i) = v_q;
    ucFcOptSuccesses(i) = success;
    ucFcOptPfc(i) = p_fc;
    
    % evaluate quality on nominal shape
    nomGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, nomGraspSamples, numContacts, ...
                                    nomShapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);

    ucFcOptNomQ(i) = mn_q';
    ucFcOptNomP(i) = p_fc;
    
    % evaluate antipodal optimization of just the FC metric
    disp('Evaluating FC optimized grasp without antipodal constraint');
    useUncertainty = false;
    forceAp = false;
    startTime = tic;
    [optGrasp, optVal, allIters, constSatisfaction] = ...
        find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com, ...
            predGrid, coneAngle, experimentConfig.numBadContacts, ...
            plateWidth, gripWidth, graspSigma, ...
            useUncertainty, forceAp);
    optimizationTime = toc(startTime);

    cfDetFcOptTimes(i) = optimizationTime;
    cfDetFcOptSatisfy(i) = constSatisfaction;
    cfDetFcOptVals(i) = optVal;

    % evaluate quality with MC sampling
    bestLoa = create_ap_loa(optGrasp, experimentConfig.gripWidth);
    bestGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, bestGraspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);

    cfDetFcOptGrasps(i,:) = optGrasp';
    cfDetFcOptMeanQ(i) = mn_q;
    cfDetFcOptVarQ(i) = v_q;
    cfDetFcOptSuccesses(i) = success;
    cfDetFcOptPfc(i) = p_fc;

    % evaluate quality on nominal shape
    nomGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, nomGraspSamples, numContacts, ...
                                    nomShapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);

    cfDetFcOptNomQ(i) = mn_q';
    cfDetFcOptNomP(i) = p_fc;
    
    % evaluate antipodal optimization of the FC metric with uncertainty
    % penalty
    disp('Evaluating FC optimized grasp with uncertainty without antipodal constraint');
    useUncertainty = true;
    forceAp = false;
    startTime = tic;
    [optGrasp, optVal, allIters, constSatisfaction] = ...
        find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com, ...
            predGrid, coneAngle, experimentConfig.numBadContacts, ...
            plateWidth, gripWidth, graspSigma, ...
            useUncertainty, forceAp);
    optimizationTime = toc(startTime);

    cfUcFcOptTimes(i) = optimizationTime;
    cfUcFcOptSatisfy(i) = constSatisfaction;
    cfUcFcOptVals(i) = optVal;

    % evaluate quality with MC sampling
    bestLoa = create_ap_loa(optGrasp, experimentConfig.gripWidth);
    bestGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, bestGraspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);

    cfUcFcOptGrasps(i,:) = optGrasp';
    cfUcFcOptMeanQ(i) = mn_q;
    cfUcFcOptVarQ(i) = v_q;
    cfUcFcOptSuccesses(i) = success;
    cfUcFcOptPfc(i) = p_fc;
    
    % evaluate quality on nominal shape
    nomGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, nomGraspSamples, numContacts, ...
                                    nomShapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, ...
                                    experimentConfig.visSampling);

    cfUcFcOptNomQ(i) = mn_q';
    cfUcFcOptNomP(i) = p_fc;
    
    % go on to the next initialization
    i = i+1;
end

% choose the best grasp from all initializations from each optimization
initialQIndex = find(initialMeanQ == max(initialMeanQ));
initialPIndex = find(initialPfc == max(initialPfc));
detFcOptIndex = find(detFcOptVals == min(detFcOptVals));
ucFcOptIndex = find(ucFcOptVals == min(ucFcOptVals));
cfDetFcOptIndex = find(cfDetFcOptVals == min(cfDetFcOptVals));
cfUcFcOptIndex = find(cfUcFcOptVals == min(cfUcFcOptVals));

% only take the first index for each
initialQIndex = initialQIndex(1);
initialPIndex = initialPIndex(1);
detFcOptIndex = detFcOptIndex(1);
ucFcOptIndex = ucFcOptIndex(1);
cfDetFcOptIndex = cfDetFcOptIndex(1);
cfUcFcOptIndex = cfUcFcOptIndex(1);


% compute the histograms for each of the best grasps
dispHist = true;
loa = create_ap_loa(detFcOptGrasps(detFcOptIndex,:)', experimentConfig.gripWidth);
graspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);    
[exp_mn_q, exp_v_q, exp_success, exp_p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, plateWidth, ...
                                    experimentConfig.visSampling, dispHist, ...
                                    experimentConfig.qScale);
h = figure(99);
title('Histogram of Quality for Det AP Constrained Opt', 'FontSize', 10);
print(h, '-depsc', sprintf('%s/%s_hist_det_ap.eps', outputDir, filename));

loa = create_ap_loa(ucFcOptGrasps(ucFcOptIndex,:)', experimentConfig.gripWidth);
graspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);    
[exp_mn_q, exp_v_q, exp_success, exp_p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, plateWidth, ...
                                    experimentConfig.visSampling, dispHist, ...
                                    experimentConfig.qScale);
h = figure(99);
title('Histogram of Quality for Uc AP Constrained Opt', 'FontSize', 10);
print(h, '-depsc', sprintf('%s/%s_hist_uc_ap.eps', outputDir, filename));

loa = create_ap_loa(cfDetFcOptGrasps(cfDetFcOptIndex,:)', experimentConfig.gripWidth);
graspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);    
[exp_mn_q, exp_v_q, exp_success, exp_p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, plateWidth, ...
                                    experimentConfig.visSampling, dispHist, ...
                                    experimentConfig.qScale);
h = figure(99);
title('Histogram of Quality for Det Opt', 'FontSize', 10);
print(h, '-depsc', sprintf('%s/%s_hist_det.eps', outputDir, filename));

loa = create_ap_loa(cfUcFcOptGrasps(cfUcFcOptIndex,:)', experimentConfig.gripWidth);
graspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);    
[exp_mn_q, exp_v_q, exp_success, exp_p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, plateWidth, ...
                                    experimentConfig.visSampling, dispHist, ...
                                    experimentConfig.qScale);
h = figure(99);
title('Histogram of Quality for Uc Opt', 'FontSize', 10);
print(h, '-depsc', sprintf('%s/%s_hist_uc.eps', outputDir, filename));

% load the best grasps on the nominal, mean, and GPIS shapes
nomGraspFilename = sprintf('%s/%s_best_grasp_nom.mat', meanCompDir, filename);
meanGraspFilename = sprintf('%s/%s_best_grasp_mean.mat', meanCompDir, filename);
ucGraspFilename = sprintf('%s/%s_best_grasp_uncertainty.mat', meanCompDir, filename);

load(nomGraspFilename);  % named bestNom
load(meanGraspFilename); % named bestMean
load(ucGraspFilename);   % named bestSampling

% compare the best grasps with one another
h = figure(31);
% subplot(1,5,1);
% visualize_grasp(initGrasp, predGrid, surfaceImage, scale, length);
% title('Initial Grasp', 'FontSize', 15);
% xlabel(sprintf('Q = %f', initialMeanQ(i)), 'FontSize', 15);
subplot(1,4,1);
visualize_grasp(cfDetFcOptGrasps(cfDetFcOptIndex,:)', predGrid, surfaceImage, scale, length);
title('Quality Only', 'FontSize', 6);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*cfDetFcOptNomQ(cfDetFcOptIndex), cfDetFcOptPfc(cfDetFcOptIndex)), ...
    'FontSize', 6);  

subplot(1,4,2);
visualize_grasp(cfUcFcOptGrasps(cfUcFcOptIndex,:)', predGrid, surfaceImage, scale, length);
title('Uncertainty Penalty Only', 'FontSize', 6);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*cfUcFcOptNomQ(cfUcFcOptIndex), cfUcFcOptPfc(cfUcFcOptIndex)), ...
    'FontSize', 6);  

subplot(1,4,3);
visualize_grasp(detFcOptGrasps(detFcOptIndex,:)', predGrid, surfaceImage, scale, length);
title('Deterministic Quality w/ AP Constraint', 'FontSize', 6);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*detFcOptNomQ(detFcOptIndex), detFcOptPfc(detFcOptIndex)), ...
    'FontSize', 6);

subplot(1,4,4);
visualize_grasp(ucFcOptGrasps(ucFcOptIndex,:)', predGrid, surfaceImage, scale, length);
title('Uncertainty Penalty w/ AP Constraint', 'FontSize', 6);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*ucFcOptNomQ(ucFcOptIndex), ucFcOptPfc(ucFcOptIndex)), ...
    'FontSize', 6);  

print(h, '-depsc', sprintf('%s/%s_comp_opt.eps', outputDir, filename));

% compare opt grasp with best initial grasp
h = figure(33);
subplot(1,3,1);
visualize_grasp(initialGrasps(initialQIndex,:)', predGrid, surfaceImage, scale, length);
title('Best E[Q] Initial Grasp', 'FontSize', 8);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*initialNomQ(initialQIndex), initialPfc(initialQIndex)), ...
    'FontSize', 8);  

subplot(1,3,2);
visualize_grasp(initialGrasps(initialPIndex,:)', predGrid, surfaceImage, scale, length);
title('Best P(FC) Initial Grasp', 'FontSize', 8);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*initialNomQ(initialPIndex), initialPfc(initialPIndex)), ...
    'FontSize', 8);    

subplot(1,3,3);
visualize_grasp(ucFcOptGrasps(ucFcOptIndex,:)', predGrid, surfaceImage, scale, length);
title('Uncertainty Penalty w/ AP Constraint', 'FontSize', 8);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*ucFcOptNomQ(ucFcOptIndex), ucFcOptPfc(ucFcOptIndex)), ...
    'FontSize', 8); 

print(h, '-depsc', sprintf('%s/%s_comp_init.eps', outputDir, filename));

% compare the det and uc optimizations with the best experimental
h = figure(34);
subplot(1,4,1);
visualize_grasp(bestNom.nomQGrasp.bestGrasp', nominalShape, nominalShape.shapeImage, scale, length);
title('Best Grasp for Nominal Shape', 'FontSize', 6);
xlabel(sprintf('Q = %.03f', ...
    experimentConfig.qScale*bestNom.nomQGrasp.Q), ...
    'FontSize', 6);

subplot(1,4,2);
visualize_grasp(bestMean.bestGrasp', nominalShape, nominalShape.shapeImage, scale, length);
title('Best Grasp for Mean Shape', 'FontSize', 6);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*bestMean.nomQ, bestMean.expP), ...
    'FontSize', 6);

subplot(1,4,3);
visualize_grasp(bestSampling.expPGrasp.bestGrasp', nominalShape, nominalShape.shapeImage, scale, length);
title('Best Grasp for GPIS Using P(FC)', 'FontSize', 6);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*bestSampling.expPGrasp.nomQ, ...
    bestSampling.expPGrasp.P), ...
    'FontSize', 6);

subplot(1,4,4);
visualize_grasp(ucFcOptGrasps(ucFcOptIndex,:)', nominalShape, nominalShape.shapeImage, scale, length);
title('Best Grasp Using Optimization', 'FontSize', 6);
xlabel(sprintf('Q = %.03f\n P(FC) = %.03f', ...
    experimentConfig.qScale*ucFcOptNomQ(ucFcOptIndex), ucFcOptPfc(ucFcOptIndex)), ...
    'FontSize', 6); 

print(h, '-depsc', sprintf('%s/%s_comp4.eps', outputDir, filename));

% create results structure
experimentResults = struct();
experimentResults.constructionResults = constructionResults;

experimentResults.initialGraspResults = ...
    create_grasp_results_struct(initialGrasps, initialMeanQ, initialVarQ, ...
                                initialSuccesses, [], [], [], [], [], initialPfc);
experimentResults.initialGraspResults.bestQIndex = initialQIndex;
experimentResults.initialGraspResults.bestPIndex = initialPIndex;

experimentResults.detFcOptGraspResults = ...
    create_grasp_results_struct(detFcOptGrasps, detFcOptMeanQ, detFcOptVarQ, ...
                                detFcOptSuccesses, detFcOptSatisfy, detFcOptTimes, ...
                                [], detFcOptVals, [], detFcOptPfc);
experimentResults.detFcOptGraspResults.bestIndex = detFcOptIndex;

experimentResults.ucFcOptGraspResults = ...
    create_grasp_results_struct(ucFcOptGrasps, ucFcOptMeanQ, ucFcOptVarQ, ...
                                ucFcOptSuccesses, ucFcOptSatisfy, ucFcOptTimes, ...
                                [], ucFcOptVals, [], ucFcOptPfc);
experimentResults.ucFcOptGraspResults.bestIndex = ucFcOptIndex;

experimentResults.cfDetFcOptGraspResults = ...
    create_grasp_results_struct(cfDetFcOptGrasps, cfDetFcOptMeanQ, cfDetFcOptVarQ, ...
                                cfDetFcOptSuccesses, cfDetFcOptSatisfy, cfDetFcOptTimes, ...
                                [], cfDetFcOptVals, [], detFcOptPfc);
experimentResults.cfDetFcOptGraspResults.bestIndex = cfDetFcOptIndex;

experimentResults.cfUcFcOptGraspResults = ...
    create_grasp_results_struct(cfUcFcOptGrasps, cfUcFcOptMeanQ, cfUcFcOptVarQ, ...
                                cfUcFcOptSuccesses, cfUcFcOptSatisfy, cfUcFcOptTimes, ...
                                [], cfUcFcOptVals, [], ucFcOptPfc);                        
experimentResults.cfUcFcOptGraspResults.bestIndex = cfUcFcOptIndex;

% save with hash
experimentHash = uint32(1e6 * rand());
experimentFilename = sprintf('%s/experiment_%d.mat', outputDir, experimentHash);
experimentResults.id = experimentHash;
if experimentConfig.graspIters > 0
    save(experimentFilename, 'experimentResults');
end

fprintf('Experiment %d complete\n', experimentHash);

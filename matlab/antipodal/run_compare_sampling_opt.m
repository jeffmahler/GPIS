function [experimentResults, gpModel, shapeParams, shapeSamples] = ...
    run_compare_sampling_opt(dim, filename, gripScale, dataDir, outputDir, meanCompDir, ...
                             sampleIters, newShape, experimentConfig, varParams, trainingParams, ...
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
samplingGrasps  = zeros(experimentConfig.graspIters, 4);
samplingMeanQ = zeros(experimentConfig.graspIters, 1);
samplingVarQ = zeros(experimentConfig.graspIters, 1);
samplingSuccesses = zeros(experimentConfig.graspIters, 1);
samplingPfc = zeros(experimentConfig.graspIters, 1);
samplingNomQ = zeros(experimentConfig.graspIters, 1);
samplingNomP = zeros(experimentConfig.graspIters, 1);

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

numContacts = 2;
coneAngle = atan(experimentConfig.frictionCoef);
length = experimentConfig.arrowLength;
surfaceImage = constructionResults.newSurfaceImage;
meanSurfaceImage = constructionResults.meanSurfaceImage;
scale = optimizationParams.scale;
win = experimentConfig.smoothWin;
sig = experimentConfig.smoothSig;
predGrid = constructionResults.predGrid;
useNormsForInit = false;
numSamples = size(shapeSamples, 2);
com = shapeParams.com;
sampleTimeOffset = experimentConfig.shapeSampleTime;

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

graspSigma = experimentConfig.graspSigma;
gripWidth = gripScale * experimentConfig.objScale * dim;
plateWidth = gripWidth * experimentConfig.plateScale;
plateWidth = uint16(round(plateWidth));
if plateWidth == 0
    plateWidth = 1;
end

optimizationParams.grip_width = gripWidth;

nomShapeSamples = cell(1, numSamples);
for k = 1:numSamples
    nomShapeSamples{k} = nominalShape;
end

if experimentConfig.visOptimization
    J = constructionResults.newSurfaceImage;
    surfaceImage = J;
    optimizationParams.surfaceImage = J;%constructionResults.newSurfaceImage;
end

if experimentConfig.graspIters < 1
    experimentResults = struct();
    experimentResults.constructionResults = constructionResults;
    return;
end

allSampleTimes = zeros(experimentConfig.trials, sampleIters);
allSampleMaxP = zeros(experimentConfig.trials, sampleIters);
allSampleConvTimes = zeros(experimentConfig.trials, 1);
allOptTimes = zeros(experimentConfig.trials, experimentConfig.graspIters);
allOptMaxP = zeros(experimentConfig.trials, experimentConfig.graspIters+1);
allOptConvTimes = zeros(experimentConfig.trials, 1);

for a = 1:experimentConfig.trials

% evaluate prob force closure MC sampling
[bestQGrasp, bestPGrasp, sampleTimes] = ...
    find_grasp_sampling(predGrid, ...
                        experimentConfig, shapeParams, shapeSamples, ...
                        surfaceImage, optimizationParams.scale, sampleIters, ...
                        nominalShape, gripWidth, plateWidth);
                    

i = 1;
while i <= experimentConfig.graspIters
    fprintf('Selecting antipodal pair %d\n', i);

    disp('Evaluating initial grasp without uncertainty');
    initGrasp = [0;0;realmax;realmax];
    while norm(initGrasp(1:2) - initGrasp(3:4)) > gripWidth
        initGrasp = get_initial_antipodal_grasp(predGrid, useNormsForInit);
    end
    
    % evaluate antipodal optimization w/ uncertainty and FC metric
    disp('Evaluating FC optimized grasp w/ uncertainty');
    startTime = tic;
    [optGrasp, optVal, allIters, constSatisfaction] = ...
        find_uc_mean_q_grasp_points(initGrasp, gpModel, ...
            optimizationParams, shapeParams.gridDim, shapeParams.com, ...
            predGrid, coneAngle, experimentConfig.numBadContacts, ...
            plateWidth, gripWidth, graspSigma);
    optimizationTime = toc(startTime);
    ucFcOptTimes(i) = optimizationTime;
    ucFcOptSatisfy(i) = constSatisfaction;
    ucFcOptVals(i) = optVal;
    
    % evaluate quality with MC sampling
    bestLoa = create_ap_loa(optGrasp, experimentConfig.gripWidth);
%     graspSamples = {bestLoa};
%     meanSamples = {predGrid};
%     [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
%                                     coneAngle, graspSamples, numContacts, ...
%                                     meanSamples, shapeParams.gridDim, ...
%                                     shapeParams.surfaceThresh, ...
%                                     experimentConfig.numBadContacts, ...
%                                     plateWidth, gripWidth, ...
%                                     true, false, 1, true);
    
    bestGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, bestGraspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, gripWidth, ...
                                    false, false, 1, false);
                                    %experimentConfig.visSampling);

    ucFcOptGrasps(i,:) = optGrasp';
    ucFcOptMeanQ(i) = mn_q;
    ucFcOptVarQ(i) = v_q;
    ucFcOptSuccesses(i) = success;
    ucFcOptPfc(i) = p_fc;
    
    figure(30+a);
    clf;
    visualize_grasp(optGrasp, predGrid, surfaceImage, ...
        scale, length, plateWidth, gripWidth);
    title('GP-GPIS-OPT', 'FontSize', 15);
    xlabel(sprintf('PFC = %.3f\n', p_fc), 'FontSize', 15);
    
    % evaluate quality on nominal shape
    nomGraspSamples = sample_grasps(bestLoa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, nomGraspSamples, numContacts, ...
                                    nomShapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, ...
                                    plateWidth, gripWidth, ...
                                    experimentConfig.visSampling);

    ucFcOptNomQ(i) = mn_q';
    ucFcOptNomP(i) = p_fc;
    
    % go on to the next initialization
    i = i+1;
end

% choose the best grasp from all initializations from each optimization
validOptInd = ucFcOptSuccesses > 0 & ucFcOptSatisfy > 0;
sampleConvInd = find(sampleTimes.maxP == max(sampleTimes.maxP));
sampleConvInd = sampleConvInd(1);
optConvInd = 1;

optTimes = cumsum(ucFcOptTimes)';
adjustedSampleTimes = sampleTimeOffset + cumsum(sampleTimes.sampleTimes);

validOptTimes = optTimes;
if sum(validOptInd) > 0
    validUcFcPfc = ucFcOptPfc(validOptInd);
    validOptTimes = optTimes(validOptInd);
    optConvInd = find(validUcFcPfc == validUcFcPfc(end));
    optConvInd = optConvInd(1);
end
sampleConvTime = adjustedSampleTimes(sampleConvInd);
optConvTime = validOptTimes(optConvInd);

numOpt = experimentConfig.graspIters;
optMaxP = zeros(1, numOpt+1);
optIndices = zeros(1, numOpt);
for j = 1:numOpt
    validOptInd = ucFcOptSuccesses(1:j) > 0 & ucFcOptSatisfy(1:j) > 0;
    if sum(validOptInd) == 0
        optIndices(j) = 1;
        optMaxP(j+1) = 0;
    else
        optInd = find(ucFcOptVals == ...
            min(ucFcOptVals(validOptInd)));
        optIndices(j) = optInd(1);
        optMaxP(j+1) = ucFcOptPfc(optInd(1));
    end
end
ucFcOptIndex = optIndices(end);

allOptTimes(a,:) = optTimes;
allOptMaxP(a,:) = optMaxP;
allOptConvTimes(a) = optConvTime;
allSampleTimes(a,:) = adjustedSampleTimes;
allSampleMaxP(a,:) = sampleTimes.maxP;
allSampleConvTimes(a) = sampleConvTime;

h = figure(30+a);
subplot(1,2,1);
visualize_grasp(bestPGrasp.bestGrasp', predGrid, surfaceImage, ...
    scale, length, plateWidth, gripWidth);
title('MC Sampling', 'FontSize', 6);
xlabel(sprintf('P(FC) = %.03f', ...
    sampleTimes.maxP(end)), ...
    'FontSize', 6);  

subplot(1,2,2);
visualize_grasp(ucFcOptGrasps(optIndices(end),:)', predGrid, surfaceImage, ...
    scale, length, plateWidth, gripWidth);
title('GP-GPIS-OPT', 'FontSize', 6);
xlabel(sprintf('P(FC) = %.03f', ...
    ucFcOptPfc(optIndices(end))), ...
    'FontSize', 6);  
print(h, '-depsc', sprintf('%s/%s_comp_opt_%d.eps', outputDir, filename, a));

end

% plot the times versus best quality
figure(1);
clf;
title('Comparison of P(FC) versus Runtime', 'FontSize', 20);
xlabel('Runtime (log(sec))', 'FontSize', 20);
ylabel('P(FC)', 'FontSize', 20);
plot([0 mean(allSampleTimes, 1)], [0 mean(allSampleMaxP, 1)], 'g', 'LineWidth', 2);
hold on;
plot([0, mean(allOptTimes, 1)], mean(allOptMaxP, 1), 'b', 'LineWidth', 2);

    
% compute the histograms for each of the best grasps
dispHist = true;

loa = create_ap_loa(ucFcOptGrasps(ucFcOptIndex,:)', experimentConfig.gripWidth);
graspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);    
[exp_mn_q, exp_v_q, exp_success, exp_p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    experimentConfig.numBadContacts, plateWidth, gripWidth, ...
                                    experimentConfig.visSampling, dispHist, ...
                                    experimentConfig.qScale);
h = figure(99);
title('Histogram of Quality for Uc AP Constrained Opt', 'FontSize', 10);
print(h, '-depsc', sprintf('%s/%s_hist_uc_ap.eps', outputDir, filename));


% compare the best grasps with one another
% h = figure(31);
% subplot(1,2,1);
% visualize_grasp(bestPGrasp(end,:)', predGrid, surfaceImage, ...
%     scale, length, plateWidth, gripWidth);
% title('MC Sampling', 'FontSize', 6);
% xlabel(sprintf('Q(N) = %.03f\n P(FC) = %.03f', ...
%     experimentConfig.qScale*cfDetFcOptNomQ(cfDetFcOptIndex), cfDetFcOptPfc(cfDetFcOptIndex)), ...
%     'FontSize', 6);  
% 
% subplot(1,2,2);
% visualize_grasp(ucFcOptGrasps(cfUcFcOptIndex,:)', predGrid, surfaceImage, ...
%     scale, length, plateWidth, gripWidth);
% title('GP-GPIS-OPT', 'FontSize', 6);
% xlabel(sprintf('Q(N) = %.03f\n P(FC) = %.03f', ...
%     experimentConfig.qScale*cfUcFcOptNomQ(cfUcFcOptIndex), cfUcFcOptPfc(cfUcFcOptIndex)), ...
%     'FontSize', 6);  
% print(h, '-depsc', sprintf('%s/%s_comp_opt.eps', outputDir, filename));


% create results structure
experimentResults = struct();
experimentResults.constructionResults = constructionResults;

experimentResults.ucFcOptGraspResults = ...
    create_grasp_results_struct(ucFcOptGrasps, ucFcOptMeanQ, ucFcOptVarQ, ...
                                ucFcOptSuccesses, ucFcOptSatisfy, ucFcOptTimes, ...
                                [], ucFcOptVals, [], ucFcOptPfc, ucFcOptNomQ);
experimentResults.ucFcOptGraspResults.bestIndex = ucFcOptIndex;
experimentResults.allSampleTimes = allSampleTimes;
experimentResults.allSampleMaxP = allSampleMaxP;
experimentResults.allSampleConvTimes = allSampleConvTimes;
experimentResults.allOptTimes = allOptTimes;
experimentResults.allOptMaxP= allOptMaxP;
experimentResults.allOptConvTimes = allOptConvTimes;


% save with hash
experimentHash = uint32(1e6 * rand());
experimentFilename = sprintf('%s/experiment_%s_%d.mat', outputDir, filename, experimentHash);
experimentResults.id = experimentHash;
if experimentConfig.graspIters > 0
    save(experimentFilename, 'experimentResults');
end

fprintf('Experiment %d complete\n', experimentHash);

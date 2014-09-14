% Test best experimental grasp versus grasp on the mean shape

predGrid = experimentResults.constructionResults.predGrid;
surfaceImage = experimentResults.constructionResults.surfaceImage;
newSurfaceImage = experimentResults.constructionResults.newSurfaceImage;

maxIters = 200;
vis = false;
showHist = false;
badContactThresh = 1000;

numContacts = 2;
coneAngle = atan(experimentConfig.frictionCoef);
length = experimentConfig.arrowLength;
useNormsForInit = true;

% create struct for nominal shape
nominalShape = struct();
nominalShape.tsdf = shapeParams.fullTsdf;
nominalShape.normals = shapeParams.fullNormals;
nominalShape.points = shapeParams.all_points;

%% evaluate best experimental grasp
[bestGrasp, bestQ, bestV, bestP, bestGraspSamples, avgSampleTime] = ...
    find_grasp_sampling(predGrid, ...
                        experimentConfig, shapeParams, shapeSamples, ...
                        surfaceImage, scale, 2);
                    
bestSampling = struct();
bestSampling.bestGrasp = bestGrasp;
bestSampling.expQ = bestQ;
bestSampling.expV = bestV;
bestSampling.expP = bestP;
bestSampling.expGS = bestGraspSamples;
bestSampling.avgSampleTime = avgSampleTime;

%% evaluate on mean shape
meanSample = {predGrid};
loa = create_ap_loa(bestSampling.bestGrasp', experimentConfig.gripWidth);
graspSample = {loa};
[mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                coneAngle, graspSample, numContacts, ...
                                meanSample, shapeParams.gridDim, ...
                                shapeParams.surfaceThresh, ...
                                badContactThresh, ...
                                experimentConfig.plateWidth, vis, showHist, ...
                                experimentConfig.qScale);

bestSampling.predQ = mn_q;
bestSampling.predV = v_q;
bestSampling.predP = p_fc;

%% evaluate on nominal shape
nominalSamples = cell(1,maxIters);
for i = 1:maxIters
    nominalSamples{i} = nominalShape;
end
loa = create_ap_loa(bestSampling.bestGrasp', experimentConfig.gripWidth);
graspSamples = sample_grasps(loa, experimentConfig.graspSigma, maxIters);
[mn_q, v_q, success, p_fc] = mc_sample_fast(nominalShape.points, ...
                                coneAngle, graspSamples, numContacts, ...
                                nominalSamples, shapeParams.gridDim, ...
                                shapeParams.surfaceThresh, ...
                                badContactThresh, ...
                                experimentConfig.plateWidth, vis, showHist, ...
                                experimentConfig.qScale);

bestSampling.nomQ = mn_q;
bestSampling.nomV = v_q;
bestSampling.nomP = p_fc;

%save(sprintf('results/google_objects/%s_best_grasp_uncertainty.mat', filename),'bestSampling');

%% evaluate best predicted grasp
sampleGrasps = false;
meanSample = {predGrid};
[bestGrasp, bestQ, bestV, bestP, bestGraspSamples, avgSampleTime] = ...
    find_grasp_sampling(predGrid, ...
                        experimentConfig, shapeParams, meanSample, ...
                        surfaceImage, scale, maxIters, false);
                    
bestMean = struct();
bestMean.bestGrasp = bestGrasp;
bestMean.predQ = bestQ;
bestMean.predV = bestV;
bestMean.predP = bestP;
bestMean.avgSampleTime = avgSampleTime;

%% evaluate grasp on samples
loa = create_ap_loa(bestMean.bestGrasp', experimentConfig.gripWidth);
meanGraspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);
[mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                coneAngle, meanGraspSamples, numContacts, ...
                                shapeSamples, shapeParams.gridDim, ...
                                shapeParams.surfaceThresh, ...
                                badContactThresh, ...
                                vis, false);
     
bestMean.expQ = mn_q;
bestMean.expV = v_q;
bestMean.expP = p_fc;
bestMean.expGS = meanGraspSamples;

%% evaluate on nominal shape
nominalSamples = {};
for i = 1:maxIters
    nominalSamples = {nominalShape};
end
loa = create_ap_loa(bestMean.bestGrasp', experimentConfig.gripWidth);
graspSamples = sample_grasps(loa, experimentConfig.graspSigma, maxIters);
[mn_q, v_q, success, p_fc] = mc_sample_fast(nominalShape.points, ...
                                coneAngle, graspSamples, numContacts, ...
                                nominalSamples, shapeParams.gridDim, ...
                                shapeParams.surfaceThresh, ...
                                badContactThresh, ...
                                experimentConfig.plateWidth, vis, showHist, ...
                                experimentConfig.qScale);

bestMean.nomQ = mn_q;
bestMean.nomV = v_q;
bestMean.nomP = p_fc;

%save(sprintf('results/google_objects/%s_best_grasp.mat', filename), 'bestMean');      

%% plot the results
figure(12);
subplot(1,2,1);
visualize_grasp(bestMean.bestGrasp', predGrid, newSurfaceImage, scale, ...
    experimentConfig.arrowLength);
title('Best Grasp for Mean Shape', 'FontSize', 15);
xlabel(sprintf('E[Q] = %.03f', experimentConfig.qScale*bestMean.expQ), 'FontSize', 15);
subplot(1,2,2);
visualize_grasp(bestSampling.bestGrasp', predGrid, newSurfaceImage, scale, ...
    experimentConfig.arrowLength);
title('Best Grasp for GPIS', 'FontSize', 15);
xlabel(sprintf('E[Q] = %.03f', experimentConfig.qScale*bestSampling.expQ), 'FontSize', 15);

%% plot the histograms
dispHist = true;
[exp_mn_q, exp_v_q, exp_success, exp_p_fc] = mc_sample_fast(predGrid.points, ...
                                coneAngle, bestSampling.expGS, numContacts, ...
                                shapeSamples, shapeParams.gridDim, ...
                                shapeParams.surfaceThresh, ...
                                badContactThresh, ...
                                vis, dispHist, experimentConfig.qScale);

[pred_mn_q, pred_v_q, pred_success, pred_p_fc] = mc_sample_fast(predGrid.points, ...
                                coneAngle, bestMean.expGS, numContacts, ...
                                shapeSamples, shapeParams.gridDim, ...
                                shapeParams.surfaceThresh, ...
                                badContactThresh, ...
                                vis, dispHist, experimentConfig.qScale);

%% load old results
load(sprintf('results/google_objects/%s_best_grasp_uncertainty.mat', filename));
load(sprintf('results/google_objects/%s_best_grasp.mat', filename));


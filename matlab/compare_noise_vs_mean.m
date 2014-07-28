% test grasps

predGrid = experimentResults.constructionResults.predGrid;
surfaceImage = experimentResults.constructionResults.surfaceImage;
newSurfaceImage = experimentResults.constructionResults.newSurfaceImage;

%%
[bestGrasp, bestQ, bestV] = ...
    find_grasp_sampling(predGrid, ...
                        experimentConfig, shapeParams, shapeSamples, ...
                        surfaceImage, ...
                        scale, 100);
                    
a = struct();
a.bestGrasp = bestGrasp;
a.bestQ = bestQ;
a.bestV = bestV;
save(sprintf('results/google_objects/%s_best_grasp_uncertainty.mat', filename),'a');
bestSampling = a;          

%%

meanSample = {predGrid};
[bestGrasp, bestQ, bestV] = ...
    find_grasp_sampling(predGrid, ...
                        experimentConfig, shapeParams, meanSample, ...
                        surfaceImage, ...
                        scale, 100);
                    
a = struct();
a.bestGrasp = bestGrasp;
a.bestQ = bestQ;
a.bestV = bestV;
save(sprintf('results/google_objects/%s_best_grasp.mat', filename),'a');
bestMean = a;       

%% evaluate grasp on samples
numContacts = 2;
coneAngle = atan(experimentConfig.frictionCoef);
length = experimentConfig.arrowLength;
useNormsForInit = true;

loa = create_ap_loa(bestMean.bestGrasp', experimentConfig.loaScale);
[mn_q, v_q, success] = mc_sample_fast(predGrid.points, ...
                                coneAngle, loa, numContacts, ...
                                shapeSamples, shapeParams.gridDim, ...
                                shapeParams.surfaceThresh, ...
                                1000, ...
                                false, false);
                            
%% plot the results
bestSample = load(sprintf('results/google_objects/%s_best_grasp_uncertainty.mat', filename));
bestMean = load(sprintf('results/google_objects/%s_best_grasp.mat', filename));

figure(12);
subplot(1,2,1);
visualize_grasp(bestMean.a.bestGrasp', predGrid, newSurfaceImage, scale, ...
    experimentConfig.arrowLength);
title('Best Grasp for Mean Shape', 'FontSize', 15);
xlabel(sprintf('Q = %f', 0.0037), 'FontSize', 15);
subplot(1,2,2);
visualize_grasp(bestSample.a.bestGrasp', predGrid, newSurfaceImage, scale, ...
    experimentConfig.arrowLength);
title('Best Grasp for GPIS', 'FontSize', 15);
xlabel(sprintf('Q = %f', bestSample.a.bestQ), 'FontSize', 15);

function [ bestSamplingGrasps, bestMeanGrasps, bestNomGrasps ] = ...
    run_mean_sampling_experiment(shapeNames, gripScales, dataDir, outputDir, maxIters, ...
                                experimentConfig, optimizationParams)
%RUN_MEAN_SAMPLING_EXPERIMENT Test best experimental grasp versus grasp on the mean shape

numShapes = size(shapeNames, 2);
bestSamplingGrasps = cell(1, numShapes);
bestMeanGrasps = cell(1, numShapes);
bestNomGrasps = cell(1, numShapes);
scale = optimizationParams.scale;
vis = experimentConfig.visSampling;

for i = 1:numShapes
    filename = shapeNames{i};
    
    [gpModel, shapeParams, shapeSamples, constructionResults] = ...
            load_experiment_object(filename, dataDir, optimizationParams.scale);

    % compute the width and scale
    dim = shapeParams.gridDim;
    gripScale = gripScales{i};
    gripWidth = gripScale * experimentConfig.objScale * dim;
    plateWidth = gripWidth * experimentConfig.plateScale;
    plateWidth = uint16(round(plateWidth));
        
    predGrid = constructionResults.predGrid;
    surfaceImage = constructionResults.surfaceImage;
    newSurfaceImage = constructionResults.newSurfaceImage;
    numSamples = size(shapeSamples, 2);
    
    vis = false;
    showHist = false;
    badContactThresh = experimentConfig.numBadContacts;

    numContacts = 2;
    coneAngle = atan(experimentConfig.frictionCoef);

    % create struct for nominal shape
    nominalShape = struct();
    nominalShape.tsdf = shapeParams.fullTsdf;
    nominalShape.normals = shapeParams.fullNormals;
    nominalShape.points = shapeParams.all_points;
    nominalShape.noise = zeros(size(nominalShape.tsdf,1), 1);
    nominalShape.gridDim = shapeParams.gridDim;
    nominalShape.surfaceThresh = shapeParams.surfaceThresh;
    nominalShape.com = shapeParams.com;
    
    %% find the best grasp experimentally
    [bestQGrasp, bestPGrasp, avgSampleTime] = ...
        find_grasp_sampling(predGrid, ...
                            experimentConfig, shapeParams, shapeSamples, ...
                            surfaceImage, optimizationParams.scale, maxIters, ...
                            gripWidth, plateWidth);

    bestSampling = struct();
    bestSampling.expQGrasp = bestQGrasp;
    bestSampling.expPGrasp = bestPGrasp;
    bestSampling.avgSampleTime = avgSampleTime;

    %% evaluate best Q grasp on mean shape
    meanSample = {predGrid};
    loa = create_ap_loa(bestSampling.expQGrasp.bestGrasp', gripWidth);
    graspSample = {loa};
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSample, numContacts, ...
                                    meanSample, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, ...
                                    plateWidth, vis, showHist, ...
                                    experimentConfig.qScale);
                                           
    bestSampling.expQGrasp.predQ = mn_q;
    bestSampling.expQGrasp.predV = v_q;
    bestSampling.expQGrasp.predP = p_fc;
    
    %% evaluate best P grasp on mean shape
    loa = create_ap_loa(bestSampling.expPGrasp.bestGrasp', gripWidth);
    graspSample = {loa};
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, graspSample, numContacts, ...
                                    meanSample, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, ...
                                    plateWidth, vis, showHist, ...
                                    experimentConfig.qScale);
                                           
    bestSampling.expPGrasp.predQ = mn_q;
    bestSampling.expPGrasp.predV = v_q;
    bestSampling.expPGrasp.predP = p_fc;

    %% evaluate best Q grasp on nominal shape
    nominalSamples = cell(1,maxIters);
    for j = 1:maxIters
        nominalSamples{j} = nominalShape;
    end
    loa = create_ap_loa(bestSampling.expQGrasp.bestGrasp', gripWidth);
    graspSamples = sample_grasps(loa, experimentConfig.graspSigma, maxIters);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(nominalShape.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    nominalSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, ...
                                    plateWidth, vis, showHist, ...
                                    experimentConfig.qScale);
                              
    bestSampling.expQGrasp.nomQ = mn_q;
    bestSampling.expQGrasp.nomV = v_q;
    bestSampling.expQGrasp.nomP = p_fc;
    
    %% evaluate best P grasp on nominal shape
    nominalSamples = cell(1,maxIters);
    for j = 1:maxIters
        nominalSamples{j} = nominalShape;
    end
    loa = create_ap_loa(bestSampling.expPGrasp.bestGrasp', gripWidth);
    graspSamples = sample_grasps(loa, experimentConfig.graspSigma, maxIters);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(nominalShape.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    nominalSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, ...
                                    plateWidth, vis, showHist, ...
                                    experimentConfig.qScale);
                              
    bestSampling.expPGrasp.nomQ = mn_q;
    bestSampling.expPGrasp.nomV = v_q;
    bestSampling.expPGrasp.nomP = p_fc;

    %% find the best grasp on the mean shape
    meanSample = {predGrid};
    sampleGrasps = false;
    [bestQGrasp, bestPGrasp, avgSampleTime] = ...
        find_grasp_sampling(predGrid, ...
                            experimentConfig, shapeParams, meanSample, ...
                            surfaceImage, optimizationParams.scale, maxIters, ...
                            gripWidth, plateWidth, sampleGrasps);

    bestMean = bestQGrasp;
    bestMean.avgSampleTime = avgSampleTime;

    %% evaluate mean grasp experimentally using samples
    useHist = false;
    loa = create_ap_loa(bestMean.bestGrasp', gripWidth);
    meanGraspSamples = sample_grasps(loa, experimentConfig.graspSigma, numSamples);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, meanGraspSamples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, plateWidth, ...
                                    vis, useHist);

    bestMean.expQ = mn_q;
    bestMean.expV = v_q;
    bestMean.expP = p_fc;
    bestMean.expGS = meanGraspSamples;

    %% evaluate mean grasp on nominal shape
    nominalSamples = cell(1,maxIters);
    for j = 1:maxIters
        nominalSamples{j} = nominalShape;
    end
    loa = create_ap_loa(bestMean.bestGrasp', gripWidth);
    graspSamples = sample_grasps(loa, experimentConfig.graspSigma, maxIters);
    [mn_q, v_q, success, p_fc] = mc_sample_fast(nominalShape.points, ...
                                    coneAngle, graspSamples, numContacts, ...
                                    nominalSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, ...
                                    plateWidth, vis, showHist, ...
                                    experimentConfig.qScale);

    bestMean.nomQ = mn_q;
    bestMean.nomV = v_q;
    bestMean.nomP = p_fc;
    
    %% find the best grasp on the nominal shape
    
    % take 'samples' so that we can still use approach uncertainty
    nominalSamples = cell(1,maxIters);
    for j = 1:maxIters
        nominalSamples{j} = nominalShape;
    end
    %experimentConfig.visSampling = true;
    [bestQGrasp, bestPGrasp, avgSampleTime] = ...
        find_grasp_sampling(nominalShape, ...
                            experimentConfig, nominalShape, nominalSamples, ...
                            surfaceImage, scale, maxIters, gripWidth, plateWidth);

    bestNom = struct();
    bestNom.nomQGrasp = bestQGrasp;
    bestNom.nomPGrasp = bestPGrasp;
    bestNom.avgSampleTime = avgSampleTime;
    %% plot the results of pfc
    h = figure(12);
    subplot(1,2,1);
    visualize_grasp(bestMean.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Best Grasp for Mean Shape', 'FontSize', 10);
    xlabel(sprintf('Q = %.03f', bestMean.expP), 'FontSize', 10);
    subplot(1,2,2);
    visualize_grasp(bestSampling.expPGrasp.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Best Grasp for GPIS Using P(FC)', 'FontSize', 10);
    xlabel(sprintf('Q = %.03f', bestSampling.expPGrasp.P), 'FontSize', 10);

    print(h, '-depsc', sprintf('%s/%s_comp2_p.eps', outputDir, filename));
    
    %% plot the results of E[Q]
    h = figure(13);
    subplot(1,2,1);
    visualize_grasp(bestMean.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Best Grasp for Mean Shape', 'FontSize', 10);
    xlabel(sprintf('E[Q] = %.03f', bestMean.expP), 'FontSize', 10);
    subplot(1,2,2);
    visualize_grasp(bestSampling.expQGrasp.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Best Grasp for GPIS Using E[Q]', 'FontSize', 10);
    xlabel(sprintf('E[Q] = %.03f', bestSampling.expQGrasp.Q), 'FontSize', 10);

    print(h, '-depsc', sprintf('%s/%s_comp2_q.eps', outputDir, filename));
    
    %% plot the results versus nominal grasp
    h = figure(14);
    
    subplot(1,4,1);
    visualize_grasp(bestNom.nomQGrasp.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Best Grasp for Nominal Shape', 'FontSize', 6);
    xlabel(sprintf('E[Q] = %.03f\n P(FC) = %.03f', ...
        experimentConfig.qScale*bestNom.nomQGrasp.Q, ...
        bestNom.nomQGrasp.P), ...
        'FontSize', 6);
    
    subplot(1,4,2);
    visualize_grasp(bestMean.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Best Grasp for Mean Shape', 'FontSize', 6);
    xlabel(sprintf('E[Q] = %.03f\n P(FC) = %.03f', ...
        experimentConfig.qScale*bestMean.expQ, ...
        bestMean.expP), ...
        'FontSize', 6);
    
    subplot(1,4,3);
    visualize_grasp(bestSampling.expQGrasp.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Best Grasp for GPIS Using E[Q]', 'FontSize', 6);
    xlabel(sprintf('E[Q] = %.03f\n P(FC) = %.03f', ...
        experimentConfig.qScale*bestSampling.expQGrasp.Q, ...
        bestSampling.expQGrasp.P), ...
        'FontSize', 6);

    subplot(1,4,4);
    visualize_grasp(bestSampling.expPGrasp.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Best Grasp for GPIS Using P(FC)', 'FontSize', 6);
    xlabel(sprintf('E[Q] = %.03f\n P(FC) = %.03f', ...
        experimentConfig.qScale*bestSampling.expPGrasp.Q, ...
        bestSampling.expPGrasp.P), ...
        'FontSize', 6);

    
    print(h, '-depsc', sprintf('%s/%s_comp3.eps', outputDir, filename));
    
    %% plot the histograms
    dispHist = true;
    [exp_mn_q, exp_v_q, exp_success, exp_p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, bestSampling.expQGrasp.samples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, plateWidth, ...
                                    vis, dispHist, experimentConfig.qScale);
    h = figure(99);
    title('Histogram of Quality for GPIS Distribution Using E[Q]', 'FontSize', 10);
    print(h, '-depsc', sprintf('%s/%s_exp_hist_q.eps', outputDir, filename));
                                
    [exp_mn_q, exp_v_q, exp_success, exp_p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, bestSampling.expPGrasp.samples, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, plateWidth, ...
                                    vis, dispHist, experimentConfig.qScale);
    h = figure(99);
    title('Histogram of Quality for GPIS Distribution Using P(FC)', 'FontSize', 10);
    print(h, '-depsc', sprintf('%s/%s_exp_hist_pfc.eps', outputDir, filename));
    
    [pred_mn_q, pred_v_q, pred_success, pred_p_fc] = mc_sample_fast(predGrid.points, ...
                                    coneAngle, bestMean.expGS, numContacts, ...
                                    shapeSamples, shapeParams.gridDim, ...
                                    shapeParams.surfaceThresh, ...
                                    badContactThresh, plateWidth, ...
                                    vis, dispHist, experimentConfig.qScale); 
                                
    h = figure(99);
    title('Histogram of Grasp Quality for Mean Shape', 'FontSize', 10);
    print(h, '-depsc', sprintf('%s/%s_mean_hist.eps', outputDir, filename));
                                     
    %% save results
    save(sprintf('%s/%s_best_grasp_uncertainty.mat', outputDir, filename),'bestSampling');
    save(sprintf('%s/%s_best_grasp_mean.mat', outputDir, filename), 'bestMean');      
    save(sprintf('%s/%s_best_grasp_nom.mat', outputDir, filename), 'bestNom');      

    bestSamplingGrasps{i} = bestSampling;
    bestMeanGrasps{i} = bestMean;
    bestNomGrasps{i} = bestNom;
    %% load old results
    % load(sprintf('results/google_objects/%s_best_grasp_uncertainty.mat', filename));
    % load(sprintf('results/google_objects/%s_best_grasp.mat', filename));

end

end


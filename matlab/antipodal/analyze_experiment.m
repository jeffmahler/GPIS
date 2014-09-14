% script to analyze grasping experiments

load('results/google_objects/full_grasping_results_v01.mat');
shapeNames = {'loofa', 'marker', 'squirt_bottle', 'stapler', 'tape', 'water'};

%%

load('results/google_objects/full_grasping_results_v02.mat');
shapeNames = {'can_opener', 'loofa', 'marker', 'plane', 'squirt_bottle', 'stapler', 'tape', 'water'};%{'marker'};


%% plot bar graphs for each shape (only a subset of results for now)
numShapes = size(allShapeResults, 2);

for i = 1:numShapes
    experimentResults = allShapeResults{1,i}.experimentResults;
    shapeName = shapeNames{i};

    figure;
    barwitherr([experimentResults.initialGraspResults.varQ, ...
         experimentResults.randomFcGraspResults.varQ, ...
         experimentResults.ucGraspResults.varQ, ...
         experimentResults.antipodalGraspResults.varQ, ...
         experimentResults.ucMeanGraspResults.varQ
        ], ...
        [experimentResults.initialGraspResults.meanQ, ...
         experimentResults.randomFcGraspResults.meanQ, ...
         experimentResults.ucGraspResults.meanQ, ...
         experimentResults.antipodalGraspResults.meanQ, ...
         experimentResults.ucMeanGraspResults.meanQ
        ]);
    xlabel('Trial', 'FontSize', 15);
    ylabel('Expected FC', 'FontSize', 15);
    title(sprintf('Comparison of FC Quality for Shape %s', shapeName), 'FontSize', 15);
    legend('Initial', 'Predicted Best Grasp', 'Antipodal Feasibility', ...
            'Antipodal w/ UC Penalty', 'Antipodal FC Opt',...
            'Location', 'Best');
end

%% find the max, median, mean
allGraspStats = cell(1,numShapes);

for i = 1:numShapes
    experimentResults = allShapeResults{1,i}.experimentResults;
    shapeName = shapeNames{i};

    load(sprintf('results/google_objects/%s_best_grasp_uncertainty.mat', shapeName));
    load(sprintf('results/google_objects/%s_best_grasp.mat', shapeName));

    predGrid = experimentResults.constructionResults.predGrid;
    surfaceImage = experimentResults.constructionResults.surfaceImage;
    newSurfaceImage = experimentResults.constructionResults.newSurfaceImage;

    graspStats = struct();

    % get best antipodal grasps and stats
    minApOpt = min(experimentResults.antipodalGraspResults.optVals);
    bestApGraspInd = find(experimentResults.antipodalGraspResults.optVals == minApOpt);
    
    graspStats.antipodal = struct();
    graspStats.antipodal.bestGrasp = experimentResults.antipodalGraspResults.grasps(bestApGraspInd,:)';
    graspStats.antipodal.bestQ = experimentResults.antipodalGraspResults.meanQ(bestApGraspInd)';
    graspStats.antipodal.bestV = experimentResults.antipodalGraspResults.varQ(bestApGraspInd)';
    graspStats.antipodal.bestP = experimentResults.antipodalGraspResults.pFc(bestApGraspInd)';

    graspStats.initial = struct();
    graspStats.initial.bestGrasp = experimentResults.initialGraspResults.grasps(bestApGraspInd,:)';
    graspStats.initial.bestQ = experimentResults.initialGraspResults.meanQ(bestApGraspInd)';
    graspStats.initial.bestV = experimentResults.initialGraspResults.varQ(bestApGraspInd)';
    graspStats.initial.bestP = experimentResults.initialGraspResults.pFc(bestApGraspInd)';

    % best fc w/o uncertainty w/o antipodality penalty on mean shape
    minUcFcDetOpt = min(experimentResults.detMeanUcGraspResults.optVals);
    bestUcFcDetGraspInd = find(experimentResults.detMeanUcGraspResults.optVals == minUcFcDetOpt);
    
    graspStats.fcUcMeanDet = struct();
    graspStats.fcUcMeanDet.bestGrasp = experimentResults.detMeanUcGraspResults.grasps(bestUcFcDetGraspInd,:)';
    graspStats.fcUcMeanDet.bestQ = experimentResults.detMeanUcGraspResults.meanQ(bestUcFcDetGraspInd)';
    graspStats.fcUcMeanDet.bestV = experimentResults.detMeanUcGraspResults.varQ(bestUcFcDetGraspInd)';
    graspStats.fcUcMeanDet.bestP = experimentResults.detMeanUcGraspResults.pFc(bestUcFcDetGraspInd)';

    % best fc w/ uncertainty penalty on mean shape
    minUcFcUcOpt = min(experimentResults.ucMeanUcGraspResults.optVals);
    bestUcFcUcGraspInd = find(experimentResults.ucMeanUcGraspResults.optVals == minUcFcUcOpt);
    
    graspStats.fcUcMeanUc = struct();
    graspStats.fcUcMeanUc.bestGrasp = experimentResults.ucMeanUcGraspResults.grasps(bestUcFcUcGraspInd,:)';
    graspStats.fcUcMeanUc.bestQ = experimentResults.ucMeanUcGraspResults.meanQ(bestUcFcUcGraspInd)';
    graspStats.fcUcMeanUc.bestV = experimentResults.ucMeanUcGraspResults.varQ(bestUcFcUcGraspInd)';
    graspStats.fcUcMeanUc.bestP = experimentResults.ucMeanUcGraspResults.pFc(bestUcFcUcGraspInd)';

    % best fc w/o uncertainty penalty on mean shape
    minFcDetOpt = min(experimentResults.detMeanGraspResults.optVals);
    bestFcDetGraspInd = find(experimentResults.detMeanGraspResults.optVals == minFcDetOpt);
    
    graspStats.fcMeanDet = struct();
    graspStats.fcMeanDet.bestGrasp = experimentResults.detMeanGraspResults.grasps(bestFcDetGraspInd,:)';
    graspStats.fcMeanDet.bestQ = experimentResults.detMeanGraspResults.meanQ(bestFcDetGraspInd)';
    graspStats.fcMeanDet.bestV = experimentResults.detMeanGraspResults.varQ(bestFcDetGraspInd)';
    graspStats.fcMeanDet.bestP = experimentResults.detMeanGraspResults.pFc(bestFcDetGraspInd)';

    % best fc w/ uncertainty penalty on mean shape
    minFcOpt = min(experimentResults.ucMeanGraspResults.optVals);
    bestFcGraspInd = find(experimentResults.ucMeanGraspResults.optVals == minFcOpt);
    
    graspStats.fcMean = struct();
    graspStats.fcMean.bestGrasp = experimentResults.ucMeanGraspResults.grasps(bestFcGraspInd,:)';
    graspStats.fcMean.bestQ = experimentResults.ucMeanGraspResults.meanQ(bestFcGraspInd)';
    graspStats.fcMean.bestV = experimentResults.ucMeanGraspResults.varQ(bestFcGraspInd)';
    graspStats.fcMean.bestP = experimentResults.ucMeanGraspResults.pFc(bestFcGraspInd)';

    % best feasible grasp
    minFeasOpt = min(experimentResults.ucGraspResults.optVals);
    bestFeasGraspInd = find(experimentResults.ucGraspResults.optVals == minFeasOpt);
    
    graspStats.feasibility = struct();
    graspStats.feasibility.bestGrasp = experimentResults.ucGraspResults.grasps(bestFeasGraspInd,:)';
    graspStats.feasibility.bestQ = experimentResults.ucGraspResults.meanQ(bestFeasGraspInd)';
    graspStats.feasibility.bestV = experimentResults.ucGraspResults.varQ(bestFeasGraspInd)';
    graspStats.feasibility.bestP = experimentResults.ucGraspResults.pFc(bestFeasGraspInd)';

    % get predicted grasp stats
    maxPredQ = max(experimentResults.randomFcGraspResults.meanQ);
    bestPredGraspInd = find(experimentResults.randomFcGraspResults.meanQ == maxPredQ);
    
    graspStats.pred = struct();
    graspStats.pred.bestGrasp = experimentResults.randomFcGraspResults.grasps(bestPredGraspInd,:)';
    graspStats.pred.bestQ = experimentResults.randomFcGraspResults.meanQ(bestPredGraspInd)';
    graspStats.pred.bestV = experimentResults.randomFcGraspResults.varQ(bestPredGraspInd)';
    graspStats.pred.bestP = experimentResults.randomFcGraspResults.pFc(bestPredGraspInd)';

    % get mean median etc for tables
    graspStats.pred.medQ = median(experimentResults.randomFcGraspResults.meanQ);
    graspStats.pred.meanQ = mean(experimentResults.randomFcGraspResults.meanQ);

    graspStats.initial.medQ = median(experimentResults.initialGraspResults.meanQ);
    graspStats.initial.meanQ = mean(experimentResults.initialGraspResults.meanQ);

    graspStats.feasibility.medQ = median(experimentResults.ucGraspResults.meanQ);
    graspStats.feasibility.meanQ = mean(experimentResults.ucGraspResults.meanQ);

    graspStats.antipodal.medQ = median(experimentResults.antipodalGraspResults.meanQ);
    graspStats.antipodal.meanQ = mean(experimentResults.antipodalGraspResults.meanQ);

    % count how many times ap was significantly better
    graspStats.antipodal.numBetter = 0;
    graspStats.antipodal.numBetterSat = 0;
    graspStats.antipodal.numSatisfied = sum(experimentResults.antipodalGraspResults.satisfied);
    numGrasps = size(experimentResults.antipodalGraspResults.meanQ,1);
    for j = 1:numGrasps
        % TODO: statistical significance
        if experimentResults.antipodalGraspResults.meanQ(j) > ...
                experimentResults.randomFcGraspResults.meanQ(j)
            graspStats.antipodal.numBetter = graspStats.antipodal.numBetter + 1;
        end

        % only count satisfied constraints
        if experimentResults.antipodalGraspResults.satisfied(j) && ...
                experimentResults.antipodalGraspResults.meanQ(j) > ...
                experimentResults.randomFcGraspResults.meanQ(j)
            graspStats.antipodal.numBetterSat = graspStats.antipodal.numBetterSat + 1;
        end
    end

    allGraspStats{i} = graspStats;

    % plot the results

%     figure(i);
%     subplot(2,4,1);
%     visualize_grasp(graspStats.pred.bestGrasp, predGrid, newSurfaceImage, scale, ...
%         experimentConfig.arrowLength);
%     title('Best Grasp for Mean Shape', 'FontSize', 15);
%     xlabel(sprintf('Q = %f, P = %f', ...
%         experimentConfig.qScale * graspStats.pred.bestQ(1), graspStats.pred.bestP(1)), ...
%         'FontSize', 15);
%     subplot(2,4,2);
%     visualize_grasp(graspStats.initial.bestGrasp, predGrid, newSurfaceImage, scale, ...
%         experimentConfig.arrowLength);
%     title('Initial', 'FontSize', 15);
%     xlabel(sprintf('Q = %f, P = %f', ...
%         experimentConfig.qScale * graspStats.initial.bestQ(1), graspStats.initial.bestP(1)), ...
%         'FontSize', 15);
%     subplot(2,4,3);
%     visualize_grasp(graspStats.feasibility.bestGrasp, predGrid, newSurfaceImage, scale, ...
%         experimentConfig.arrowLength);
%     title('Feasibility', 'FontSize', 15);
%     xlabel(sprintf('Q = %f, P = %f', ...
%         experimentConfig.qScale * graspStats.feasibility.bestQ(1), graspStats.feasibility.bestP(1)), ...
%         'FontSize', 15);
%     subplot(2,4,4);
%     visualize_grasp(graspStats.antipodal.bestGrasp, predGrid, newSurfaceImage, scale, ...
%         experimentConfig.arrowLength);
%     title('Antipodal', 'FontSize', 15);
%     xlabel(sprintf('Q = %f, P = %f', ...
%         experimentConfig.qScale * graspStats.antipodal.bestQ(1), graspStats.antipodal.bestP(1)), ...
%         'FontSize', 15);
%     subplot(2,4,5);
%     visualize_grasp(graspStats.fcUcMeanDet.bestGrasp, predGrid, newSurfaceImage, scale, ...
%         experimentConfig.arrowLength);
%     title('FC Opt', 'FontSize', 15);
%     xlabel(sprintf('Q = %f, P = %f', ...
%         experimentConfig.qScale * graspStats.fcUcMeanDet.bestQ(1), graspStats.fcUcMeanDet.bestP(1)), ...
%         'FontSize', 15);
%     subplot(2,4,6);
%     visualize_grasp(graspStats.fcUcMeanUc.bestGrasp, predGrid, newSurfaceImage, scale, ...
%         experimentConfig.arrowLength);
%     title('FC Opt w/ UC', 'FontSize', 15);
%     xlabel(sprintf('Q = %f, P = %f', ...
%         experimentConfig.qScale * graspStats.fcUcMeanUc.bestQ(1), graspStats.fcUcMeanUc.bestP(1)), ...
%         'FontSize', 15);
%     subplot(2,4,7);
%     visualize_grasp(graspStats.fcMeanDet.bestGrasp, predGrid, newSurfaceImage, scale, ...
%         experimentConfig.arrowLength);
%     title('FC Opt w/ AP', 'FontSize', 15);
%     xlabel(sprintf('Q = %f, P = %f', ...
%         experimentConfig.qScale * graspStats.fcMeanDet.bestQ(1), graspStats.fcMeanDet.bestP(1)), ...
%         'FontSize', 15);
%     subplot(2,4,8);
%     visualize_grasp(graspStats.fcMean.bestGrasp, predGrid, newSurfaceImage, scale, ...
%         experimentConfig.arrowLength);
%     title('FC Opt w/ AP and UC', 'FontSize', 15);
%     xlabel(sprintf('Q = %f, P = %f', ...
%         experimentConfig.qScale * graspStats.fcMean.bestQ(1), graspStats.fcMean.bestP(1)), ...
%         'FontSize', 15);

    figure(i+numShapes);
    subplot(1,5,1);
    visualize_grasp(bestMean.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Best Grasp for Mean Shape', 'FontSize', 15);
    xlabel(sprintf('E[Q] = %f', ...
        experimentConfig.qScale * bestMean.expQ), ...
        'FontSize', 15);
    subplot(1,5,2);
    visualize_grasp(bestSampling.bestGrasp', predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Empirical Best Grasp', 'FontSize', 15);
    xlabel(sprintf('E[Q] = %f', ...
        experimentConfig.qScale * bestSampling.expQ), ...
        'FontSize', 15);
    subplot(1,5,3);
    visualize_grasp(graspStats.initial.bestGrasp, predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Initial Grasp', 'FontSize', 15);
    xlabel(sprintf('E[Q] = %f', ...
        experimentConfig.qScale * graspStats.initial.bestQ(1)), ...
        'FontSize', 15);
    subplot(1,5,4);
    visualize_grasp(graspStats.fcMean.bestGrasp, predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('FC Optimized Grasp', 'FontSize', 15);
    xlabel(sprintf('E[Q] = %f', ...
        experimentConfig.qScale * graspStats.fcMean.bestQ(1)), ...
        'FontSize', 15);
    subplot(1,5,5);
    visualize_grasp(graspStats.antipodal.bestGrasp, predGrid, newSurfaceImage, scale, ...
        experimentConfig.arrowLength);
    title('Antipodal Optimized Grasp', 'FontSize', 15);
    xlabel(sprintf('E[Q] = %f', ...
        experimentConfig.qScale * graspStats.antipodal.bestQ(1)), ...
        'FontSize', 15);

end



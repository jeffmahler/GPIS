% script to plot runtime data
offsets = {1.5}; % num seconds to sample shape
offset = 1.5;
shapeNames = {'can_opener', 'deodorant', 'marker', 'plane', 'squirt_bottle', 'stapler', 'tape', 'water'};
gripScales = {0.4, 0.6, 0.8, 1.2, 0.4, 0.6, 0.75, 0.4};
hashNums = {739510, 48767, 632350, 872635, 327421, 675042, 355060, 37458};%{555647, 163603, 24685, 736205, 241629, 658943, 412912, 34654};
dataDir = 'results/optimization/runtimes';
meanCompDir = 'results/mean_vs_predicted_exp/icra_long';
numShapes = size(shapeNames, 2);

avgSampleTimes = zeros(1, numShapes);
totalSampleTimes = zeros(1, numShapes);
avgOptTimes = zeros(1, numShapes);
totalOptTimes = zeros(1, numShapes);

for i = 1:numShapes
    shapeName = shapeNames{i};
    hashNum = hashNums{i};
    %offset = offsets{i};
    optResultName = sprintf('%s/experiment_%s_%d.mat', dataDir, shapeName, hashNum);
    load(optResultName);
    % now its in experimentResults
    
    % load the best grasps on the nominal, mean, and GPIS shapes
    nomGraspFilename = sprintf('%s/%s_best_grasp_nom.mat', meanCompDir, shapeName);
    meanGraspFilename = sprintf('%s/%s_best_grasp_mean.mat', meanCompDir, shapeName);
    ucGraspFilename = sprintf('%s/%s_best_grasp_uncertainty.mat', meanCompDir, shapeName);

    load(nomGraspFilename);  % named bestNom
    load(meanGraspFilename); % named bestMean
    load(ucGraspFilename);   % named bestSampling
    
    % mean shape
    meanMaxP = bestMean.sampleTimes.maxP;
    meanTimes = bestMean.sampleTimes.sampleTimes;
    adjustedMeanTimes = offset + meanTimes;
    adjustedMeanTimesCum = cumsum(adjustedMeanTimes);

    figure(i);
    close;
    figure(i);
    %plot(log(adjustedMeanTimesCum), meanMaxP, 'r', 'LineWidth', 2);
    title('Comparison of P(FC) versus Runtime', 'FontSize', 20);
    xlabel('Runtime (log(sec))', 'FontSize', 20);
    ylabel('P(FC)', 'FontSize', 20);
    hold on;
    
    % pred shape
    sampleMaxP = bestSampling.sampleTimes.maxP;
    numSamples = size(bestSampling.sampleTimes.maxP, 1);
    sampleMaxP = [0; sampleMaxP];
    sampleTimes = bestSampling.sampleTimes.sampleTimes;
    avgSampleTimes(i) = mean(sampleTimes);
    totalSampleTimes(i) = sum(sampleTimes);
    adjustedSampleTimes = offset + cumsum(sampleTimes);
    adjustedSampleTimes = [1, adjustedSampleTimes];
    
    figure(i);
    plot(adjustedSampleTimes', sampleMaxP, 'g', 'LineWidth', 2);
    
    % opt results
    
    % create min obj value
    optResults = experimentResults.ucFcOptGraspResults;
    numOpt = size(optResults.grasps, 1);
    optMaxP = zeros(1, numOpt+1);
    for j = 1:numOpt
        validOptInd = optResults.successes(1:j) > 0 & optResults.satisfied(1:j) > 0;
        if validOptInd == 0
            optMaxP(j+1) = 0;
        else
            optInd = find(optResults.optVals == ...
            min(optResults.optVals(validOptInd)));
        
            optMaxP(j+1) = optResults.pFc(optInd);
        end
    end
    
    adjustedOptTimes = optResults.times;
    avgOptTimes(i) = mean(adjustedOptTimes);
    totalOptTimes(i) = sum(adjustedOptTimes);
    adjustedOptTimesCum = cumsum(adjustedOptTimes)';
    adjustedOptTimesCum = [1, adjustedOptTimesCum];
    
    figure(i);
    plot(adjustedOptTimesCum', optMaxP, 'b', 'LineWidth', 2);
    legend('Sampling', 'Optimization', 'Location', 'Best');
    
end
% script to plot runtime data
offsets = {1.5}; % num seconds to sample shape
offset = 1.5;
worstCaseIndex = 753;
%shapeNames = {'plane'};
shapeNames = {'can_opener', 'deodorant', 'marker', 'plane', 'squirt_bottle', 'stapler', 'tape', 'water'};
%gripScales = {1.2};
gripScales = {0.38, 0.6, 0.8, 1.2, 0.4, 0.6, 0.75, 0.4};
%hashNums = {953818};
hashNums = {739510, 48767, 632350, 872635, 327421, 675042, 355060, 37458};%{555647, 163603, 24685, 736205, 241629, 658943, 412912, 34654};
dataDir = 'results/optimization/runtimes';
meanCompDir = 'results/mean_vs_predicted_exp/icra_long';
numShapes = size(shapeNames, 2);

avgSampleTimes = zeros(1, numShapes);
trueSampleTimes = zeros(1, numShapes);
totalSampleTimes = zeros(1, numShapes);
avgOptTimes = zeros(1, numShapes);
totalOptTimes = zeros(1, numShapes);
sampleBestTimes = zeros(1, numShapes);
sampleBestInds = cell(1, numShapes);
optBestTimes = zeros(1, numShapes);
optBestInds = cell(1, numShapes);

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
    trueSampleTimes(i) = sum(sampleTimes(1:worstCaseIndex));
    adjustedSampleTimes = offset + cumsum(sampleTimes);
    adjustedSampleTimes = [1, adjustedSampleTimes];
    
    
    sampleBestInds{i} = find(sampleMaxP == max(sampleMaxP));
    sampleBestTimes(i) = adjustedSampleTimes(sampleBestInds{i}(1));
    sampleBestInds{i} = sampleBestInds{i}(1);
    
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
    
    optBestInds{i} = optInd;
    optBestTimes(i) = adjustedOptTimesCum(optInd+1);
    
    figure(i);
    plot(adjustedOptTimesCum', optMaxP, 'b', 'LineWidth', 2);
    legend('Sampling', 'Optimization', 'Location', 'Best');
    
end

%%
dims = [25, 40, 55, 70];
optTimePerIter = [8.4, 22.2, 62.0, 187.9];
sampleTimePerIter = [1.12, 1.51, 1.90, 2.54];
shapeSampleTime = [1.5, 13.5, 74.4, 392.8];
figure(111);
plot(dims, optTimePerIter, '--^r', 'LineWidth', 2, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'r');
hold on;
%plot(dims, sampleTimePerIter, 'g');
plot(dims, shapeSampleTime, '-ob', 'LineWidth', 2,  ...
    'MarkerSize', 10, 'MarkerFaceColor', 'b');
%title('Runtime versus GPIS Resolution', 'FontSize', 20);
xlabel('Dimension of Grid', 'FontSize', 20);
xlim([22, 75]);
ylabel('Time (sec)', 'FontSize', 20);

h_leg = legend('Optimization (Time Per Initialization)', 'Sampling (Time Per 1000 Shape Samples)', ...
    'Location', 'Best');
set(h_leg, 'FontSize', 14);


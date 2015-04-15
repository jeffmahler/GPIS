%%
figure(1);
[x, y, z] = sphere;
surf(x, y, z);
hold on;
surf(3+2*x, 3+2*y, 3+2*z);

%% experiment with medial axis
I = imread('data/brown_dataset/bird13.pgm');
for max_moment = 0:1:40
se = strel('diamond', 1);
I_dil = I;
for i = 1:max_moment
    I_dil = imerode(I_dil, se);
end
d = bwdist(I_dil, 'cityblock');
[h, w] = size(d);
medial = zeros(h, w);
dist_in_win = [sqrt(2) 1 sqrt(2);
                1 0 1;
                sqrt(2) 1 sqrt(2)];
            
for i = 1:h
    for j = 1:w
        i_above = max(1, i-1);
        i_below = min(h, i+1);
        j_left = max(1, j-1);
        j_right = min(w, j+1);
        d_win = d(i_above:i_below, j_left:j_right);
        d_max = max(max([d_win(1,1), d_win(1,end), d_win(end,1), d_win(end,end)]));
        %dist_to_max = min(dist_in_win(d_win == d_max));
        if d(i,j) + 1 > d_max && d(i,j) > 0
            medial(i,j) = 1;
        end
    end
end
figure(1);
subplot(1,2,1);
imshow(I_dil);
subplot(1,2,2);
imshow(medial);
pause(0.1);
end
%% opt
figure(1);
clf;
title('Comparison of P(FC) versus Runtime', 'FontSize', 20);
xlabel('Runtime (log(sec))', 'FontSize', 20);
ylabel('P(FC)', 'FontSize', 20);
num_samples = size(experimentResults.allSampleTimes, 2);
num_opt = size(experimentResults.allOptTimes, 2);
plot(0:num_samples, [0, mean(experimentResults.allSampleMaxP, 1)], 'g', 'LineWidth', 2);
hold on;
plot(0:num_opt, mean(experimentResults.allOptMaxP, 1), 'b', 'LineWidth', 2);

%% look at the time to converge
numShapes = size(allShapeResults,2);
avgSampleConvTimes = zeros(numShapes,1);
avgOptConvTimes = zeros(numShapes,1);
sampleConvTimes = cell(numShapes,1);
optConvTimes = cell(numShapes,1);

allOptTimes = cell(numShapes,1);
allOptMaxP = cell(numShapes,1);
allSampleTimes = cell(numShapes,1);
allSampleMaxP = cell(numShapes,1);

trials = 3;

for j = 1:numShapes
experimentResults = allShapeResults{j}.experimentResults;
sampleConvTimes{j} = [];
optConvTimes{j} = [];

for i = 1:trials
optTimes = experimentResults.allOptTimes(i,:);
optMaxP = experimentResults.allOptMaxP(i,:);
sampleTimes = experimentResults.allSampleTimes(i,:);
sampleMaxP = experimentResults.allSampleMaxP(i,:);

sampleConvInd = find(sampleMaxP == max(sampleMaxP));
sampleConvInd = sampleConvInd(1);
optConvInd = find(optMaxP == optMaxP(end));
optConvInd = optConvInd(1);
sampleConvTime = sampleTimes(sampleConvInd);
optConvTime = optTimes(max(optConvInd-1, 1));

sampleConvTimes{j} = [sampleConvTimes{j}; sampleConvTime];
optConvTimes{j} = [optConvTimes{j}; optConvTime];

allOptTimes{j} = [allOptTimes{j}; optTimes];
allOptMaxP{j} = [allOptMaxP{j}; optMaxP];
allSampleTimes{j} = [allSampleTimes{j}; sampleTimes];
allSampleMaxP{j} = [allSampleMaxP{j}; sampleMaxP];
end
avgSampleConvTime = mean(sampleConvTimes{j});
avgOptConvTime = mean(optConvTimes{j});
avgSampleConvTimes(j) = avgSampleConvTime;
avgOptConvTimes(j) = avgOptConvTime;

fprintf('Shape %d\n', j);
fprintf('MC Sampling converged in %f sec\n', avgSampleConvTime);
fprintf('GP-GPIS-OPT converged in %f sec\n', avgOptConvTime);
end

avgOptTime = zeros(1, size(allOptTimes{1}, 2));
avgOptMaxP = zeros(1, size(allOptMaxP{1}, 2));
avgSampleTime = zeros(1, size(allSampleTimes{1}, 2));
avgSampleMaxP = zeros(1, size(allSampleMaxP{1}, 2));

optMaxP = zeros(trials*numShapes, size(allOptTimes{1}, 2)+1);
sampleMaxP = zeros(trials*numShapes, size(allOptTimes{1}, 2));

num_samples = size(allSampleTimes{1}, 2);
num_opt = size(allOptTimes{1}, 2);
num_div = numShapes - 1;
for k = 1:numShapes

    % skip the squirt bottle (outlier)
    if k ~= 5
        avgOptTime = avgOptTime + (1.0 / num_div) * mean(allOptTimes{k});
        avgOptMaxP = avgOptMaxP + (1.0 / num_div) * mean(allOptMaxP{k});
        avgSampleTime = avgSampleTime + (1.0 / num_div) * mean(allSampleTimes{k});
        avgSampleMaxP = avgSampleMaxP + (1.0 / num_div) * mean(allSampleMaxP{k});
    end
    indices = [k, k + numShapes, k + 2*numShapes]; 
    optMaxP(indices, :) = allOptMaxP{k};
    sampleMaxP(indices, :) = allSampleMaxP{k};

    figure(2+k);
    clf;
    plot(0:num_opt, mean(allOptMaxP{k}), 'b', 'LineWidth', 2);
    hold on;
    plot(0:num_samples, [0, mean(allSampleMaxP{k})], 'g', 'LineWidth', 2);
    title('Comparison of P(FC) versus Runtime', 'FontSize', 20);
    xlabel('Number of Samples', 'FontSize', 20);
    ylabel('P(FC)', 'FontSize', 20);
    legend({'GP-P', 'GP-G'}, 'Position', [0.65,0.20,0.2,0.2], 'FontSize', 15);
end

figure(1);
clf;
plot([0, avgOptTime], avgOptMaxP, 'b', 'LineWidth', 2);
hold on;
plot([0, avgSampleTime], [0, avgSampleMaxP], 'g', 'LineWidth', 2);
title('Comparison of P(FC) versus Runtime', 'FontSize', 20);
xlabel('Runtime (log(sec))', 'FontSize', 20);
ylabel('P(FC)', 'FontSize', 20);

figure(2);
clf;
plot(0:num_opt, median(optMaxP), 'b', 'LineWidth', 2);
hold on;
plot(0:num_samples, [0, median(sampleMaxP)], 'g', 'LineWidth', 2);
title('Comparison of P(FC) versus Runtime', 'FontSize', 20);
xlabel('Runtime (log(sec))', 'FontSize', 20);
ylabel('P(FC)', 'FontSize', 20);

%%
for j = 1:numShapes
    bestSamplingP = max(allShapeResults{j}.experimentResults.allSampleMaxP(:,100));
    bestOptP = max(allShapeResults{j}.experimentResults.allOptMaxP(:,11));
    fprintf('Shape %d\n', j);
    fprintf('MC Sampling best PFC = %.3f\n', bestSamplingP);
    fprintf('GP-GPIS-OPT best PFC = %.3f\n', bestOptP);
end

%% center of mass

num_samples = size(shapeSamples, 2);
com_vals = zeros(num_samples, 2);
for i = 1:num_samples
    shape_sample = shapeSamples{i};
    com = mean(shapeParams.all_points(shape_sample.tsdf < 0,:));
    com_vals(i,:) = com;
end
mean_com = mean(com_vals, 1);
mean_com_diff = com_vals - repmat(mean_com, num_samples, 1);
std_com = mean(mean_com_diff.^2, 1);
var_com = (1.0 / num_samples) * (mean_com_diff' * mean_com_diff);
[evecs, evals] = eig(var_com);

figure(77);
scatter(com_vals(:,2), com_vals(:,1), '+b');
hold on;
scatter(mean_com(2), mean_com(1), 100, 'or', 'MarkerFaceColor', 'r');
% plot([mean_com(2); mean_com(2) + evals(1,1)*evecs(1,1)], ...
%      [mean_com(1); mean_com(1) + evals(1,1)*evecs(2,1)], 'g');
% plot([mean_com(2); mean_com(2) + evals(2,2)*evecs(2,2)], ...
%      [mean_com(1); mean_com(1) + evals(2,2)*evecs(1,2)], 'g');
diffs = [max(com_vals) - min(com_vals)];
length = max(diffs);

xlim([mean_com(2) - 3 * length / 4, mean_com(2) + 3 * length / 4]);
ylim([mean_com(1) - 3 * length / 4, mean_com(1) + 3 * length / 4]);
title('Center of mass distribution', 'FontSize', 15);
xlabel('X Axis', 'FontSize', 15);
ylabel('Y Axis', 'FontSize', 15);
std_com * 3
%%
dim = shapeParams.gridDim;
gripScale = gripScales{1};
gripWidth = gripScale * experimentConfig.objScale * dim;
plateWidth = gripWidth * experimentConfig.plateScale;
plateWidth = uint16(round(plateWidth));

predGrid = experimentResults.constructionResults.predGrid;
surfaceImage = experimentResults.constructionResults.surfaceImage;
newSurfaceImage = experimentResults.constructionResults.newSurfaceImage;

% create struct for nominal shape
nominalShape = struct();
nominalShape.tsdf = shapeParams.fullTsdf;
nominalShape.normals = shapeParams.fullNormals;
nominalShape.points = shapeParams.all_points;
nominalShape.noise = zeros(size(nominalShape.tsdf,1), 1);
nominalShape.gridDim = shapeParams.gridDim;
nominalShape.surfaceThresh = shapeParams.surfaceThresh;
nominalShape.com = shapeParams.com;

newSurfaceImage = reshape(nominalShape.tsdf, 25, 25);
newSurfaceImage = imresize(newSurfaceImage, 2.0);

h = figure(12);
subplot(1,2,1);
visualize_grasp(bestMeanGrasps{1}.bestGrasp', nominalShape, newSurfaceImage, scale, ...
    experimentConfig.arrowLength, plateWidth, gripWidth);
title('Best Grasp for Mean Shape', 'FontSize', 10);
xlabel(sprintf('P(FC) = %.03f', bestMeanGrasps{1}.expP), 'FontSize', 10);
subplot(1,2,2);
visualize_grasp(bestPredGrasps{1}.expPGrasp.bestGrasp', nominalShape, newSurfaceImage, scale, ...
    experimentConfig.arrowLength, plateWidth, gripWidth);
title('Best Grasp for GPIS Using P(FC)', 'FontSize', 10);
xlabel(sprintf('P(FC) = %.03f', bestPredGrasps{1}.expPGrasp.P), 'FontSize', 10);

%print(h, '-depsc', sprintf('%s/%s_comp2_p.eps', outputDir, filename));

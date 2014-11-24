function [gpModel, predShape, testIndices] = ...
    select_active_set_straddle(shapeParams, trainingParams)

points = shapeParams.points;
tsdf = shapeParams.tsdf;
normals = shapeParams.normals;
noise = shapeParams.noise;

h = trainingParams.levelSet;
eps = trainingParams.eps;
delta = trainingParams.delta;
K = trainingParams.activeSetSize;
numIters = trainingParams.numIters;
useGradients = trainingParams.useGradients;

use_beta_lik = false;
if size(noise, 1) == 0
    use_beta_lik = true;
end

% init uncertain, high, and low sets (which store indices)
numPoints = size(points,1);
active = zeros(numPoints,1); % the points in the active set
high = zeros(numPoints,1);
low = zeros(numPoints,1);

% Create gpis representation of the specified surface
numTraining = size(tsdf,1);
inputDim = size(points,2);

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
%covfunc = {@covMaterniso, 3};
covfunc = {@covSEiso};
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
nMeanHyp = inputDim+1;

% choose first element randomly because initially variances are equal
firstIndex = trainingParams.firstIndex;% uint16(numPoints*rand());
active(firstIndex) = 1;

% creat gpmodel struct
gpModel = struct();
gpModel.alpha = [];
gpModel.hyp = [];
gpModel.meanFunc = meanfunc;
gpModel.covFunc = covfunc;
gpModel.likFunc = likfunc;

% minimize the hyperparams on a random subset of the data to get a prior
hyperSubset = 5;
hyperIndices = randperm(numTraining);
maxInd = min(hyperSubset*K, numTraining);
hyperIndices = hyperIndices(1:maxInd);
training_x = points(hyperIndices,:);
training_y = tsdf(hyperIndices,:);

training_dy = [];
if useGradients
    training_dy = normals(hyperIndices,:);
end

training_beta = [];
if ~use_beta_lik
    training_beta = noise(hyperIndices,:);
end

%hyp.cov = [0.6, 0]; hyp.mean = zeros(nMeanHyp, 1); hyp.lik = log(0.1);
hyp = trainingParams.hyp;
gpModel = create_gpis(training_x, training_y, training_dy, training_beta, ...
    true, gpModel.hyp, numIters, hyp.cov, hyp.mean, hyp.lik);

% choose the active 
chosenPoints = points(firstIndex,:);
chosenScores = 0;
chosenTsdf = 0;
chosenVars = 0;
gpAlphas = cell(1,K);

prevPoints = csvread('activePoints.csv');
prevTsdf = csvread('activeTsdf.csv');
prevVars = csvread('activeVars.csv');
prevScores = csvread('activeScores.csv');
prevAlphas = load('gpAlphas.mat');
prevAlphas = prevAlphas.gpAlphas;

for i = 2:K
    varScale = 2 * log(size(tsdf,1) * pi^2 * i^2 / (6 * delta));
    
    numActive = sum(active);
    numUpper = sum(high);
    numLower = sum(low);
    fprintf('Beta %f\n', varScale);
    fprintf('Choosing element %d\n', i);
    fprintf('Num unclassified %d\n', numPoints - numActive - numUpper - numLower);
    training_x = points(active==1,:);
    training_y = tsdf(active==1,:);
    if useGradients
        training_dy = normals(active==1,:);
    end
    if ~use_beta_lik
        training_beta = noise(active==1,:);
    end
    
    % only use uncertain points as test indices
    uncertain = (active == 0 & high == 0 & low == 0);
    testIndices = find(uncertain);
    testPoints = points(testIndices, :);
    numTest = size(testPoints,1);
    
    if numTest == 0
       break; 
    end
  
    % compute alpha from the data
    gpModel = create_gpis(training_x, training_y, training_dy, training_beta, ...
        false, gpModel.hyp);
    gpAlphas{i-1} = gpModel.alpha;

    % predict tsdf and variance on the remaining set
    [testTsdf, Mx, Kxxp] = gp_mean(gpModel, testPoints, useGradients);
    testTsdf = testTsdf(1:numTest,:);
    testVars = gp_cov(gpModel, testPoints, Kxxp, useGradients);
    testVars = diag(testVars(1:numTest,1:numTest)); % diagonal approx
    
%    allVars = gp_cov(gpModel, points, [], true);
%    allVars = diag(allVars(1:numPoints,1:numPoints)); % diagonal approx
%     allVarsGrid = reshape(allVars, uint16(sqrt(numPoints)),  uint16(sqrt(numPoints)));
%     allVarsIm = imresize(allVarsGrid, 5);
%     figure(1);
%     imshow(allVarsIm);
    
    % choose next point according to straddle rule
    maxConfidenceRegion = testTsdf + sqrt(varScale)*testVars;
    minConfidenceRegion = testTsdf - sqrt(varScale)*testVars;
    ambiguity = min([(maxConfidenceRegion-h) (h-minConfidenceRegion)], [], 2);
    maxAmbiguity = find(ambiguity == max(ambiguity));
    %randIndices = randperm(size(maxAmbiguity,1));
    bestIndex = maxAmbiguity(1); % just choose first element
    
    chosenPoint = testPoints(bestIndex, :);
    chosenPoints = [chosenPoints; chosenPoint];
    chosenTsdf = [chosenTsdf; testTsdf(bestIndex)];
    chosenVars = [chosenVars; testVars(bestIndex)];
    chosenScores = [chosenScores; max(ambiguity)];
    fprintf('Chose point %d %d\n', chosenPoint(1), chosenPoint(2));
    
    if i == 48 %i <= size(prevScores,1) && abs(chosenScores(i) - prevScores(i)) > 1e-4
        stop = 1;
    end
    
    % update sets
    active(testIndices(bestIndex)) = 1;
    newH = find(minConfidenceRegion + eps > h);
    newL = find(maxConfidenceRegion - eps < h);
    high(testIndices(newH)) = 1;
    low(testIndices(newL)) = 1;
end

csvwrite('activePoints.csv', chosenPoints);
csvwrite('activeTsdf.csv', chosenTsdf);
csvwrite('activeVars.csv', chosenVars);
csvwrite('activeScores.csv', chosenScores);
save('gpAlphas.mat', 'gpAlphas');

disp('Done selecting active set');
gpModel = create_gpis(training_x, training_y, training_dy, training_beta, true, gpModel.hyp, ...
    numIters, hyp.cov, hyp.mean, hyp.lik);

testIndices = find(active == 0);
testPoints = points(testIndices,:);
numTest = size(testPoints,1);

[testTsdf, Mx, Kxxp] = gp_mean(gpModel, testPoints, useGradients);
predShape = struct();
predShape.gridDim = shapeParams.gridDim;
predShape.vertices= shapeParams.vertices;
predShape.com = shapeParams.com;
predShape.tsdf = testTsdf(1:numTest);
if useGradients
    predShape.normals = reshape(testTsdf(numTest+1:size(testTsdf,1)), numTest,2);
end
predShape.noise = gp_cov(gpModel, testPoints, Kxxp, useGradients);
predShape.noise = diag(predShape.noise(1:numTest, 1:numTest));
predShape.points = points(testIndices,:);
predShape.activePoints = chosenPoints;
predShape.activeScores = chosenScores;

end


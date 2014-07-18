function [gpModel, predShape, testIndices] = ...
    select_active_set_straddle(shapeParams, trainingParams)

points = shapeParams.points;
tsdf = shapeParams.tsdf;
normals = shapeParams.normals;
noise = shapeParams.noise;

h = trainingParams.levelSet;
varScale = trainingParams.beta;
eps = trainingParams.eps;
K = trainingParams.activeSetSize;
numIters = trainingParams.numIters;

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
covfunc = {@covMaterniso, 3};
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
nMeanHyp = inputDim+1;

% choose first element randomly because initially variances are equal
firstIndex = uint16(numPoints*rand());
active(firstIndex) = 1;

% creat gpmodel struct
gpModel = struct();
gpModel.alpha = [];
gpModel.hyp = [];
gpModel.meanFunc = meanfunc;
gpModel.covFunc = covfunc;
gpModel.likFunc = likfunc;

% minimize the hyperparams on a random subset of the data to get a prior
hyperIndices = randperm(numTraining);
maxInd = min(5*K, numTraining);
hyperIndices = hyperIndices(1:maxInd);
training_x = points(hyperIndices,:);
training_y = tsdf(hyperIndices,:);
training_dy = normals(hyperIndices,:);
training_beta = [];
if ~use_beta_lik
    training_beta = noise(hyperIndices,:);
end
hyp.cov = [0.6, 0]; hyp.mean = zeros(nMeanHyp, 1); hyp.lik = log(0.1);
gpModel = create_gpis(training_x, training_y, training_dy, training_beta, ...
    true, gpModel.hyp, numIters, hyp.cov, hyp.mean, hyp.lik);

% choose the active set
chosenPoints = zeros(0,2);
for i = 2:K
    %varScale = 2 * log(size(tsdf,1) * 3.14^2 * i^2 / (6 * eps));
    
    %if mod(i, 25) == 0
        fprintf('Choosing element %d\n', i);
    %end
    numActive = sum(active);
    training_x = points(active==1,:);
    training_y = tsdf(active==1,:);
    training_dy = normals(active==1,:);
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

    % predict tsdf and variance on the remaining set
    [testTsdf, Mx, Kxxp] = gp_mean(gpModel, testPoints, true);
    testTsdf = testTsdf(1:numTest,:);
    testVars = gp_cov(gpModel, testPoints, Kxxp, true);
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
    randIndices = randperm(size(maxAmbiguity,1));
    bestIndex = maxAmbiguity(randIndices(1)); % just choose first element
 
    if i == 51
       stop = 1; 
    end
    
    chosenPoint = testPoints(bestIndex, :);
    fprintf('Chose point %d %d\n', chosenPoint(1), chosenPoint(2));
    
    % update sets
    active(testIndices(bestIndex)) = 1;
    newH = find(minConfidenceRegion + eps > h);
    newL = find(maxConfidenceRegion - eps < h);
    high(testIndices(newH)) = 1;
    low(testIndices(newL)) = 1;
end

disp('Done selecting active set');
gpModel = create_gpis(training_x, training_y, training_dy, training_beta, true, gpModel.hyp, ...
    numIters, hyp.cov, hyp.mean, hyp.lik);

testIndices = find(active == 0);
testPoints = points(testIndices,:);
numTest = size(testPoints,1);

[testTsdf, Mx, Kxxp] = gp_mean(gpModel, testPoints, true);
predShape = struct();
predShape.gridDim = shapeParams.gridDim;
predShape.vertices= shapeParams.vertices;
predShape.com = shapeParams.com;
predShape.tsdf = testTsdf(1:numTest);
predShape.normals = reshape(testTsdf(numTest+1:size(testTsdf,1)), numTest,2);
predShape.noise = gp_cov(gpModel, testPoints, Kxxp, true);
predShape.noise = diag(predShape.noise(1:numTest, 1:numTest));
predShape.points = points(testIndices,:);

end


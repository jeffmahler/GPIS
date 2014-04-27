function [gpModel, activePoints, testIndices, testTsdf, testVars] = ...
    select_active_set_straddle( points, tsdf, K, numIters, h, varScale, eps)

if nargin < 5
   h = 0; % select 0 by default 
end
if nargin < 6
   varScale = 1.96; % choose standard 1-d 95% confidence by default 
end
if nargin < 7
   eps = 1e-2; % choose accuracy param to 0.01 by default 
end

% init uncertain, high, and low sets (which store indices)
numPoints = size(points,1);
active = zeros(numPoints,1); % the points in the active set
uncertain = ones(numPoints,1);
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
hyperIndices = hyperIndices(1:K);
training_x = points(hyperIndices,:);
training_y = tsdf(hyperIndices,:);
hyp.cov = [4, 2]; hyp.mean = zeros(nMeanHyp, 1); hyp.lik = log(0.1);
hyp = minimize(hyp, @gp, -numIters, @infExact, meanfunc, covfunc, likfunc,...
    training_x, training_y);    
gpModel.hyp = hyp;

% choose the active set
for i = 2:K
    %fprintf('Choosing element %d\n', i);
    numActive = sum(active);
    training_x = points(active==1,:);
    training_y = tsdf(active==1,:);
    
    % only use uncertain points as test indices
    uncertain = (active == 0 & high == 0 & low == 0);
    testIndices = find(uncertain);
    testPoints = points(testIndices, :);
    numTest = size(testPoints,1);
    
    if numTest == 0
       break; 
    end
  
    %disp('Constructing GPIS representation');  
    % compute alpha from the data
    Kxx = feval(covfunc{:}, hyp.cov, training_x);
    Mx = feval(meanfunc{:}, hyp.mean, training_x);
    beta = exp(2*hyp.lik);                             
    if beta < 1e-6                 % very tiny sn2 can lead to numerical trouble
        L = chol(Kxx + beta*eye(numActive)); % Cholesky factor of covariance with noise
        sl = 1;
    else
        L = chol(Kxx/beta + eye(numActive));     % Cholesky factor of B
        sl = beta; 
    end
    alpha = solve_chol(L, training_y - Mx) / sl;
    gpModel.alpha = alpha;

    % predict tsdf and variance on the remaining set
    [testTsdf, testVars] = gp(gpModel.hyp, @infExact, gpModel.meanFunc, ...
    gpModel.covFunc, gpModel.likFunc, training_x, training_y, testPoints);

    % choose next point according to straddle rule
    maxConfidenceRegion = testTsdf + sqrt(varScale)*testVars;
    minConfidenceRegion = testTsdf - sqrt(varScale)*testVars;
    ambiguity = min([(maxConfidenceRegion-h) (h-minConfidenceRegion)], [], 2);
    maxAmbiguity = find(ambiguity == max(ambiguity));
    randIndices = randperm(1:size(maxAmbiguity,1));
    bestIndex = maxAmbiguity(randIndices(1)); % just choose first element
    
    % update sets
    active(testIndices(bestIndex)) = 1;
    newH = find(minConfidenceRegion + eps > h);
    newL = find(maxConfidenceRegion - eps < h);
    high(testIndices(newH)) = 1;
    low(testIndices(newL)) = 1;
end

gpModel.training_x = training_x;
gpModel.training_y = training_y;

% fine tune the hyperparams (we can only do better, right?)
hyp = minimize(hyp, @gp, -numIters, @infExact, meanfunc, covfunc, likfunc, training_x, training_y);    

% compute alpha from the data
gpModel.hyp = hyp;
Kxx = feval(covfunc{:}, hyp.cov, training_x);
Mx = feval(meanfunc{:}, hyp.mean, training_x);
beta = exp(2*hyp.lik);                             
if beta < 1e-6                 % very tiny sn2 can lead to numerical trouble
    low = chol(Kxx + beta*eye(numActive)); % Cholesky factor of covariance with noise
    sl = 1;
else
    low = chol(Kxx/beta + eye(numActive));     % Cholesky factor of B
    sl = beta; 
end
alpha = solve_chol(low, training_y - Mx) / sl;
gpModel.alpha = alpha;
activePoints = training_x;

testIndices = find(active == 0);
testPoints = points(testIndices,:);
[testTsdf, testVars] = gp(gpModel.hyp, @infExact, gpModel.meanFunc, ...
    gpModel.covFunc, gpModel.likFunc, training_x, training_y, testPoints);

activePoints = [points(firstIndex,:); activePoints];

end


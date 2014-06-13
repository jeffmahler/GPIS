function [gpModel, activePoints, testIndices, testTsdf, testVars] = ...
    select_active_set_entropy( points, tsdf, K, numIters )

nPoints = size(points,1);
active = zeros(nPoints,1);

% Create gpis representation of the specified surface
numTraining = size(tsdf,1);
inputDim = size(points,2);

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
covfunc = {@covMaterniso, 3};
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
nMeanHyp = inputDim+1;

firstIndex = uint16(nPoints*rand());
active(firstIndex) = 1;

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
hyp.cov = [2, 2]; hyp.mean = zeros(nMeanHyp, 1); hyp.lik = log(0.1);
hyp = minimize(hyp, @gp, -numIters, @infExact, meanfunc, covfunc, likfunc,...
    training_x, training_y);    
gpModel.hyp = hyp;

for i = 2:K
    %fprintf('Choosing element %d\n', i);
    numActive = sum(active);
    training_x = points(active==1,:);
    training_y = tsdf(active==1,:);
  
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

    testIndices = find(active == 0);
    testPoints = points(testIndices, :);
    nTest = size(testPoints,1);
    [testTsdf, testVars] = gp(gpModel.hyp, @infExact, gpModel.meanFunc, ...
    gpModel.covFunc, gpModel.likFunc, training_x, training_y, testPoints);

    informationGain = 0.5 * log(abs(ones(nTest,1) + testVars / beta));
    maxInfo = find(informationGain == max(informationGain));
    bestIndex = maxInfo(1); % just choose first element
    active(testIndices(bestIndex)) = 1;
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
    L = chol(Kxx + beta*eye(numActive)); % Cholesky factor of covariance with noise
    sl = 1;
else
    L = chol(Kxx/beta + eye(numActive));     % Cholesky factor of B
    sl = beta; 
end
alpha = solve_chol(L, training_y - Mx) / sl;
gpModel.alpha = alpha;
activePoints = training_x;

testIndices = find(active == 0);
testPoints = points(testIndices,:);
[testTsdf, testVars] = gp(gpModel.hyp, @infExact, gpModel.meanFunc, ...
    gpModel.covFunc, gpModel.likFunc, training_x, training_y, testPoints);

end


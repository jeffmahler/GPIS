function [gpModel, activePoints, testIndices, testTsdf, testVars] = ...
    select_active_set_subsample( points, tsdf, K, numIters )

% Create gpis representation of the specified surface
numTraining = size(tsdf,1);
inputDim = size(points,2);

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
covfunc = {@covMaterniso, 3};
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
nMeanHyp = inputDim+1;

dim = sqrt(numTraining);
P = sqrt(K);
D = ceil(dim / P);
P = ceil(P);
[X, Y] = meshgrid(1:P, 1:P);
trainingPoints = double(D * [X(:), Y(:)]);
trainingPoints = trainingPoints(trainingPoints(:,1) < dim & ...
    trainingPoints(:,2) < dim, :);
numTraining = size(trainingPoints,1);
active = zeros(dim, dim);
active(trainingPoints(:,1), trainingPoints(:,2)) = 1;
trainingIndices = active == 1;
trainingIndices = trainingIndices(:);
testIndices = active == 0;
testIndices = testIndices(:);
testIndices = find(testIndices == 1);
testPoints = double(points(testIndices,:));

gpModel = struct();
gpModel.alpha = [];
gpModel.hyp = [];
gpModel.meanFunc = meanfunc;
gpModel.covFunc = covfunc;
gpModel.likFunc = likfunc;
gpModel.training_x = trainingPoints;
gpModel.training_y = tsdf(trainingIndices,:);

activePoints = gpModel.training_x;

% minimize the hyperparams on a random subset of the data to get a prior
hyp.cov = [2, 2]; hyp.mean = zeros(nMeanHyp, 1); hyp.lik = log(0.1);
hyp = minimize(hyp, @gp, -numIters, @infExact, meanfunc, covfunc, likfunc,...
    gpModel.training_x,gpModel. training_y);    
gpModel.hyp = hyp;

disp('Constructing GPIS representation');  
% compute alpha from the data
Kxx = feval(covfunc{:}, hyp.cov, gpModel.training_x);
Mx = feval(meanfunc{:}, hyp.mean, gpModel.training_x);
beta = exp(2*hyp.lik);                             
if beta < 1e-6                 % very tiny sn2 can lead to numerical trouble
    L = chol(Kxx + beta*eye(numTraining)); % Cholesky factor of covariance with noise
    sl = 1;
else
    L = chol(Kxx/beta + eye(numTraining));     % Cholesky factor of B
    sl = beta;
end
alpha = solve_chol(L, gpModel.training_y - Mx) / sl;
gpModel.alpha = alpha;

[testTsdf, testVars] = gp(gpModel.hyp, @infExact, gpModel.meanFunc, ...
    gpModel.covFunc, gpModel.likFunc, gpModel.training_x, gpModel.training_y, testPoints);

end




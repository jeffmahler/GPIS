function gpModel = create_gpis(points, tsdf, numIters)
% Create gpis representation of the specified surface
numTraining = size(tsdf,1);
inputDim = size(points,2);

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
covfunc = {@covMaterniso, 3};
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
nMeanHyp = inputDim+1;

training_x = points;
training_y = tsdf;

gpModel = struct();
gpModel.alpha = [];
gpModel.hyp = [];
gpModel.meanFunc = meanfunc;
gpModel.covFunc = covfunc;
gpModel.likFunc = likfunc;
gpModel.training_x = training_x;
gpModel.training_y = training_y;

disp('Constructing GPIS representation');
hyp.cov = [2, 2]; hyp.mean = zeros(nMeanHyp, 1); hyp.lik = log(0.1);
hyp = minimize(hyp, @gp, -numIters, @infExact, meanfunc, covfunc, likfunc, training_x, training_y);    
        
% compute alpha from the data
gpModel.hyp = hyp;
Kxx = feval(covfunc{:}, hyp.cov, training_x);
Mx = feval(meanfunc{:}, hyp.mean, training_x);
beta = exp(2*hyp.lik);                             
if beta < 1e-6                 % very tiny sn2 can lead to numerical trouble
    L = chol(Kxx + beta*eye(numTraining)); % Cholesky factor of covariance with noise
    sl = 1;
else
    L = chol(Kxx/beta + eye(numTraining));     % Cholesky factor of B
    sl = beta; 
end
alpha = solve_chol(L, training_y - Mx) / sl;
gpModel.alpha = alpha;

end


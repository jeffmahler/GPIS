function gpModel = create_gpis(points, tsdf, normals, noise, trainHyp, hyp, numIters, cov_init, mean_init, lik_init)
% Create gpis representation of the specified surface

numTraining = size(tsdf,1);
inputDim = size(points,2);
nMeanHyp = inputDim+1;

use_noise = true;
if nargin < 4 || size(noise,1) == 0
    use_noise = false;
end
if nargin < 5
    trainHyp = true;
end
if nargin < 6
    trainHyp = true;
end
if nargin < 7
    numIters = 10;
end
if nargin < 8
    cov_init = [2 2];
end
if nargin < 9
    mean_init = zeros(nMeanHyp, 1);
end
if nargin < 10
    lik_init = log(0.1);
end

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
covfunc = {@covSEiso};
likfunc = @likGauss;

num_gradients = size(normals,1);
use_gradients = true;
if num_gradients == 0
    use_gradients = false;
end

training_x = points;
training_y = tsdf;
training_beta = noise;

if trainHyp
    hyp.cov = cov_init; hyp.mean = mean_init; hyp.lik = lik_init;
 %   hyp = minimize(hyp, @gp, -numIters, @infExact, meanfunc, covfunc, likfunc, training_x, training_y);    
end

if use_gradients
    reshaped_normals = reshape(normals, numTraining*inputDim, 1);
    training_y = [tsdf; reshaped_normals];
end

gpModel = struct();
gpModel.alpha = [];
gpModel.hyp = [];
gpModel.meanFunc = meanfunc;
gpModel.covFunc = covfunc;
gpModel.likFunc = likfunc;
gpModel.training_x = training_x;
gpModel.training_y = training_y;

% compute alpha from the data
gpModel.hyp = hyp;
if use_noise
    beta = training_beta;
else
    beta = exp(2*hyp.lik);
end
gpModel.beta = beta;

if use_gradients
    Kxx = se_cov_derivative(covfunc, hyp.cov, beta, training_x);%feval(covfunc{:}, hyp.cov, training_x);
    Mx = linear_mean_derivative(meanfunc, hyp.mean, training_x);
    Q = Kxx;
else 
    Kxx = feval(covfunc{:}, hyp.cov, training_x);
    Mx = feval(meanfunc{:}, hyp.mean, training_x);
    if use_noise
        Q = Kxx + diag(beta).*eye(numTraining);
    else
        Q = Kxx + beta*eye(numTraining);
    end
end                  
gpModel.Kxx = Kxx;
gpModel.Q = Q;
gpModel.Mx = Mx;
if use_gradients || use_noise || beta < 1e-6                 % very tiny sn2 can lead to numerical trouble
    L = chol(Q); % Cholesky factor of covariance with noise
    sl = 1;
else
    L = chol(Kxx/beta + eye(numTraining));     % Cholesky factor of B
    sl = beta; 
end
gpModel.L = L;
gpModel.sl = sl;
alpha = solve_chol(L, training_y - Mx) / sl;
gpModel.alpha = alpha;

end


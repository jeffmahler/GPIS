% function (script) for loading a collected tsdf and generating a gpis

% csvName = 'data/tsdf2.csv';
% tsdf = csvread(csvName);
% 
% %%
% save('data/tsdf2.mat','tsdf');
% 
% %%
% load('data/tsdf2.mat','tsdf');
%
% minDim = min(tsdf);
% maxDim = max(tsdf);


%% Collect manually specified tsdf
[points, tsdf] = manual_tsdf('Polygons');

%% Attempt to create a gpis representation from the samples
numIters = 10;
numTraining = 1e3;
% D = 8; %downsample rate
N = size(tsdf,1);
inputDim = size(tsdf,2) - 1;
outputDim = 1;

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1;
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
nMeanHyp = inputDim+1;

% reduce size of training set cause its huuuugggee
shuffledIndices = randperm(N);
trainingIndices = shuffledIndices(1:numTraining);

training_x = tsdf(trainingIndices,1:inputDim);
training_y = tsdf(trainingIndices,inputDim+1:inputDim+outputDim);

gpModel = struct();
gpModel.alpha = cell(1,outputDim);
gpModel.hyp = cell(1,outputDim);
gpModel.meanFunc = meanfunc;
gpModel.covFunc = covfunc;
gpModel.likFunc = likfunc;
gpModel.training_x = training_x;
gpModel.training_y = training_y;

%%
clear tsdf;
clear shuffledIndices;

%%

for i = 1:outputDim
    fprintf(sprintf('Predicting dimension %d\n', i));
    hyp.cov = [0; 0]; hyp.mean = zeros(nMeanHyp, 1); hyp.lik = log(0.1);
    hyp = minimize(hyp, @gp, -numIters, @infExact, meanfunc, covfunc, likfunc, training_x, training_y(:,i));    
        
    % compute alpha from the data
    gpModel.hyp{i} = hyp;
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
    alpha = solve_chol(L, training_y(:,i) - Mx) / sl;
    gpModel.alpha{i} = alpha;
end

%% evaluate model and plot
D = 40;
numPoints = D^3;
tsdf = zeros(numPoints,1);

[ix, iy, iz] = meshgrid(1:D, 1:D, 1:D);
X = maxDim(1,1)*((ix-1) / D) + minDim(1,1);
Y = maxDim(1,2)*((iy-1) / D) + minDim(1,2);
Z = maxDim(1,3)*((iz-1) / D) + minDim(1,3);

% predict new points
points = [X(:), Y(:), Z(:)];
[tsdf_vals, tsdf_vars] = gp(gpModel.hyp{1}, @infExact, gpModel.meanFunc, ...
    gpModel.covFunc, gpModel.likFunc, gpModel.training_x, gpModel.training_y, points);

%% reconstruct the surface, color by variances
thresh = 1.0;
surface_points = points(abs(tsdf_vals) < thresh,:);
surface_vars = tsdf_vars(abs(tsdf_vals) < thresh,:);
numSurface = size(surface_points,1);
surface_colors = zeros(numSurface,3);

% color by variance (blue is min, red is max)
surface_colors(:,1) = ones(numSurface, 1) .* surface_vars / max(surface_vars);
surface_colors(:,3) = ones(numSurface, 1) - ones(numSurface, 1) .* surface_vars / max(surface_vars);


%%
figure(1);
scatter3(surface_points(:,1),surface_points(:,2),surface_points(:,3), ...
    10.0, surface_colors, 'fill'); 



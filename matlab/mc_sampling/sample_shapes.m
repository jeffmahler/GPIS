function [ shape_samples, pdfs] = ...
    sample_shapes(gpModel, gridDim, numSamples, useGradients, downsample)
%SAMPLE_SHAPE Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4
    useGradients = true;
end
if nargin < 5
    downsample = 1;
end

dim = round(gridDim / downsample);
[X, Y] = meshgrid(1:dim, 1:dim);
allPoints = downsample * [X(:), Y(:)];
num_points = dim^2;

COV = gp_cov(gpModel, allPoints, [], useGradients);
MEAN = gp_mean(gpModel, allPoints, useGradients); 

% make sure the sample is somewhat probable (almost everything will
% evaluate to inf)
samples = mvnrnd(MEAN, COV + 1e-12*eye(size(COV,1)), numSamples);

shape_samples = cell(1, numSamples);
n_vals = size(samples, 2);

for i = 1:numSamples
    shape_samples{i} = struct();
    shape_samples{i}.scale = downsample;
    shape_samples{i}.points = allPoints;
    shape_samples{i}.tsdf = samples(i,1:num_points)';
    if useGradients
        shape_samples{i}.normals = reshape(samples(i,num_points+1:n_vals),num_points,2);
    end
end

pdfs = mvnpdf(samples, MEAN', COV + 1e-12*eye(size(COV,1))); 

end


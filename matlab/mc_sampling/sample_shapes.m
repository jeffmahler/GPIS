function [ shape_samples, pdfs] = ...
    sample_shapes(gpModel, gridDim, numSamples,resolution)
%SAMPLE_SHAPE Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4
    resolution = 1;
end

[X, Y] = meshgrid(1:resolution:gridDim, 1:resolution:gridDim);
allPoints = [X(:), Y(:)];
num_points = (size(allPoints,1));

COV = gp_cov(gpModel, allPoints, [], true);
MEAN = gp_mean(gpModel, allPoints, true); 


% make sure the sample is somewhat probable (almost everything will
% evaluate to inf)
samples = mvnrnd(MEAN, COV + 1e-12*eye(size(COV,1)), numSamples);

shape_samples = cell(1, numSamples);
n_vals = size(samples, 2);

for i = 1:numSamples
    shape_samples{i} = struct();
    shape_samples{i}.tsdf = samples(i,1:num_points)'; 
    shape_samples{i}.normals = reshape(samples(i,num_points+1:n_vals),num_points,2); 
end

pdfs = mvnpdf(samples, MEAN', COV + 1e-12*eye(size(COV,1))); 

end


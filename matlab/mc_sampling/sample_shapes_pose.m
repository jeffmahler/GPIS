function [ shape_samples] = ...
    sample_shapes_pose(gpModel, gridDim, numSamples, useGradients, downsample,pose_samples)
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
scale = 2; 

shape_samples = cell(1, numSamples);
grid_center = mean(gpModel.training_x)';
grid_center = repmat(grid_center, 1, size(allPoints, 1));


COV = gp_cov(gpModel, allPoints, [], useGradients);
MEAN = gp_mean(gpModel, allPoints, useGradients); 

% make sure the sample is somewhat probable (almost everything will
% evaluate to inf)
samples = mvnrnd(MEAN, COV + 1e-12*eye(size(COV,1)), numSamples);

shape_samples = cell(1, numSamples);
n_vals = size(samples, 2);

for i = 1:numSamples
    
    pose_sample = pose_samples{i}; 
    t = pose_sample(1:2); 
    phi = pose_sample(3); 
    
    
    
    Tsdf = samples(i,1:num_points)';
    %rotate pose 
    tsdfGrid = reshape(Tsdf, dim, dim);
     
%     figure;
%     tsdfGridBig = imresize(tsdfGrid, scale);
%     imshow(tsdfGridBig);
%     
    tsdfGrid = imrotate(tsdfGrid,radtodeg(phi),'crop','bilinear');
    tsdf = tsdfGrid(:); 
    M = max(tsdf); 
    tsdf(tsdf == 0) = M; 
    tsdfGrid = reshape(tsdf, dim, dim);
    tsdfGrid = imtranslate(tsdfGrid,t,M); 
%     figure;
%     tsdfGridBig = imresize(tsdfGrid, scale);
%     imshow(tsdfGridBig);
    
    
    shape_samples{i} = struct();
    shape_samples{i}.scale = downsample;
    shape_samples{i}.all_points = allPoints;
    shape_samples{i}.tsdf = tsdfGrid(:);
    shape_samples{i}.dim = dim; 
    
    if useGradients
        shape_samples{i}.normals = reshape(samples(i,num_points+1:n_vals),num_points,2);
    end
    
end

 


end


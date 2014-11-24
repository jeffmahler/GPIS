function shape_params = tsdf_to_shape_params(tsdf, noise_scale)
%TSDF_TO_SHAPE_PARAMS Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
   noise_scale = 0.1; 
end

[height, width] = size(tsdf);

shape_params = struct();

% construct GPIS using full and quadtree points
[Gx, Gy] = imgradientxy(tsdf, 'CentralDifference');
[X, Y] = meshgrid(1:height, 1:width);
noise_grid = noise_scale * ones(height, width);

shape_params.gridDim = height; % assumes square
shape_params.tsdf = tsdf(:);
shape_params.normals = [Gx(:), Gy(:)];
shape_params.points = [X(:), Y(:)];
shape_params.noise = noise_grid(:);
shape_params.all_points = shape_params.points;
shape_params.fullTsdf = shape_params.tsdf;
shape_params.fullNormals = shape_params.normals;
shape_params.com = mean(shape_params.points(shape_params.tsdf < 0,:));

end


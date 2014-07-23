function [shapeParams, shapeImage, points, com] = ...
    create_tsdf(filename, dataDir, gridDim, varParams, surfaceThresh, useCom) 
% Function for running construction experiments with GPIS
% INSTRUCTIONS:
%   1. Specify a type of shape
%   2. Click surface, interior, and exterior points using GUI to generate
%   the TSDF
%   3. Create GPIS representation
%   4. Predict values for the entire space and threshold to find the surface


% Collect manually specified tsdf
if nargin < 6
    useCom = false;
end


% param specification
shape = 'Polygons';
shapeName = sprintf('%s/%s.mat', dataDir, filename);
pointsName = sprintf('%s/%s_points.csv', dataDir, filename);


% experiment params
points = csvread(pointsName);
if size(points,1) > 1
    % convert to new format
    points = reshape(points', 1, 2*size(points,1));
    csvwrite(pointsName, points);
end

if useCom
    comName = sprintf('%s/%s_com.csv', dataDir, filename);
    com = csvread(comName);
end
[shapeParams, shapeImage] = ...
    auto_tsdf(shape, gridDim, points, varParams);
shapeParams.surfaceThresh = surfaceThresh;
save(shapeName, 'shapeParams');

if useCom
    shapeParams.com = com;
else
    com = shapeParams.com;
end






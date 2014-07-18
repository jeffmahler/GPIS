function [shapeParams, shapeImage] = ...
    create_tsdf(filename, dataDir, gridDim, varParams) 
% Function for running construction experiments with GPIS
% INSTRUCTIONS:
%   1. Specify a type of shape
%   2. Click surface, interior, and exterior points using GUI to generate
%   the TSDF
%   3. Create GPIS representation
%   4. Predict values for the entire space and threshold to find the surface


% Collect manually specified tsdf

% param specification
shape = 'Polygons';
shapeName = sprintf('%s/%s.mat', dataDir, filename);
pointsName = sprintf('%s/%s_points.csv', dataDir, filename);
comName = sprintf('%s/%s_com.csv', dataDir, filename);

% experiment params
points = csvread(pointsName);
com = csvread(comName);
[shapeParams, shapeImage] = ...
    auto_tsdf(shape, gridDim, points, com, varParams);
save(shapeName, 'shapeParams');






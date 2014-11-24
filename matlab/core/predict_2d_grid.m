function [predShape, surfaceShape] ...
    = predict_2d_grid( gpModel, gridDim, thresh, useGradients, downsample)
% Predict tsdf values for 2d grid and display the result

if nargin < 4
   useGradients = true; 
end
if nargin < 5
   downsample = 1; 
end

dim = round(gridDim / downsample);
[X, Y] = meshgrid(1:dim, 1:dim);
testPoints = downsample * [X(:), Y(:)];
numTest = size(testPoints,1);

[allTsdf, Mx, Kxxp] = gp_mean(gpModel, testPoints, useGradients);

predShape = struct();
predShape.gridDim = dim;
if useGradients
    predShape.normals = reshape(allTsdf(numTest+1:size(allTsdf,1)), numTest, 2);
end
predShape.tsdf = allTsdf(1:numTest);
predShape.noise = gp_cov(gpModel, testPoints, Kxxp, useGradients);
predShape.noise = diag(predShape.noise(1:numTest, 1:numTest));
predShape.points = testPoints;
predShape.surfaceThresh = thresh;
predShape.com = mean(predShape.points(predShape.tsdf < 0,:));

% reconstruct the surface, color by variances
surfaceIndices = find(abs(predShape.tsdf) < thresh);
surfaceShape = struct();
surfaceShape.points = testPoints(surfaceIndices, :);
surfaceShape.tsdf = predShape.tsdf(surfaceIndices, :);
if useGradients
    surfaceShape.normals = predShape.normals(surfaceIndices, :);
end
surfaceShape.noise = predShape.noise(surfaceIndices, :);

numSurface = size(surfaceShape.points,1);

% if numSurface > 0
%     surfaceColors = zeros(numSurface,3);
% 
%     % color by variance (blue is min, red is max)
%     surfaceColors(:,1) = ones(numSurface, 1) .* surfaceVars / max(surfaceVars);
%     surfaceColors(:,3) = ones(numSurface, 1) - ones(numSurface, 1) .* surfaceVars / max(surfaceVars);
% 
%     % display the gpis zero crossing
% %     figure(1);
% %     scatter(surfacePoints(:,1),surfacePoints(:,2), 40.0, surfaceColors, 'fill');
% %     axis([1 gridDim 1 gridDim]);
% %     title('Predicted Surface Colored by Variance (Red = high, Blue = low)');
% %     xlabel('X');
% %     ylabel('Y');
% %     set(gca,'YDir','reverse');
% end

end


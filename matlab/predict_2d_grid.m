function [predShape, surfaceShape] ...
    = predict_2d_grid( gpModel, gridDim, thresh)
% Predict tsdf values for 2d grid and display the result

[X, Y] = meshgrid(1:gridDim, 1:gridDim);
testPoints = [X(:), Y(:)];
numTest = size(testPoints,1);

[allTsdf, Mx, Kxxp] = gp_mean(gpModel, testPoints, true);

predShape = struct();
predShape.gridDim = gridDim;
predShape.normals = reshape(allTsdf(numTest+1:size(allTsdf,1)), numTest, 2);
predShape.tsdf = allTsdf(1:numTest);
predShape.noise = gp_cov(gpModel, testPoints, Kxxp, true);
predShape.noise = diag(predShape.noise(1:numTest, 1:numTest));
predShape.points = testPoints;

% reconstruct the surface, color by variances
surfaceIndices = find(abs(predShape.tsdf) < thresh);
surfaceShape = struct();
surfaceShape.points = testPoints(surfaceIndices, :);
surfaceShape.tsdf = predShape.tsdf(surfaceIndices, :);
surfaceShape.normals = predShape.normals(surfaceIndices, :);
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


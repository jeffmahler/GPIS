function [gpModel, predGrid, tsdfReconError, normalError, surfaceImage, selectionTime] = ...
    create_sparse_gpis(shapeParams, trainingParams, scale)
%CREATE_SPARSE_GPIS Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    scale = 1;
end

activeSetMethod = trainingParams.activeSetMethod;
activeSetSize = trainingParams.activeSetSize;
fprintf('Creating sparse GP using %s with size %d\n', ...
    activeSetMethod, activeSetSize);

startTime = tic;
[gpModel, predShape, testIndices] = ...
    select_active_set(shapeParams, trainingParams);
elapsedTime = toc(startTime);
selectionTime = elapsedTime;

fprintf('Active set learning for %d elements took %f sec\n', activeSetSize, elapsedTime);

% Test the squared error from the true tsdf values for the parts of the
% tsdf not used in training
tsdfReconError = ...
    evaluate_errors(predShape.tsdf, shapeParams.tsdf(testIndices,:), activeSetSize);
normalError = ...
    evaluate_errors(predShape.normals, shapeParams.normals(testIndices,:), activeSetSize);

fprintf('Test error\n');
fprintf('TSDF: Mean error: %f, std error %f\n', ...
    tsdfReconError.meanError, ...
    tsdfReconError.stdError);
fprintf('Normals: Mean error: %f, std error %f\n', ...
    normalError.meanError, ...
    normalError.stdError); 

% Display resulting TSDF
[predGrid, predSurface] = predict_2d_grid( gpModel, shapeParams.gridDim,...
    trainingParams.surfaceThresh);

[testImageDarkened, combImageBig] = create_tsdf_image(predGrid, scale);
surfaceImage = combImageBig;

figure(2);
imshow(combImageBig);
hold on;
scatter(scale*gpModel.training_x(1,1), scale*gpModel.training_x(1,2), 150.0, 'x', 'LineWidth', 1.5);
scatter(scale*gpModel.training_x(:,1), scale*gpModel.training_x(:,2), 50.0, 'x', 'LineWidth', 1.5);
hold off;

% Write to file
imwrite(testImageDarkened, ...
        sprintf('results/active_set/tsdf%s%d.jpg', activeSetMethod, activeSetSize));



end


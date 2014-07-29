function [gpModel, predGrid, tsdfReconError, normalError, surfaceImage, constructionTime] = ...
    create_dense_gpis(shapeParams, trainingParams, scale)
%CREATE_DENSE_GPIS Create a GPIS using all samples from the surface
numPoints = shapeParams.gridDim * shapeParams.gridDim;

startTime = tic;
gpModel = create_gpis(shapeParams.points, shapeParams.tsdf, ...
    shapeParams.normals, shapeParams.noise, trainingParams.trainHyp, ...
    trainingParams.hyp, trainingParams.numIters);
constructionTime = toc(startTime);

[predGrid, predSurface] = predict_2d_grid(gpModel, shapeParams.gridDim,...
    trainingParams.surfaceThresh);

% noiseGrid = reshape(predGrid.noise, 25 ,25);
% a = prctile(noiseGrid(:), 80);
% noiseGrid(noiseGrid > a) = a;
% figure;
% surf(noiseGrid);

tsdfReconError = ...
    evaluate_errors(predGrid.tsdf, shapeParams.fullTsdf, numPoints);
normalError = ...
    evaluate_errors(predGrid.normals, shapeParams.fullNormals, numPoints);

[testImageDarkened, combImageBig] = create_tsdf_image(predGrid, scale);
surfaceImage = combImageBig;

figure;
imshow(combImageBig);
hold on;
scatter(scale*gpModel.training_x(1,1), scale*gpModel.training_x(1,2), 150.0, 'x', 'LineWidth', 1.5);
scatter(scale*gpModel.training_x(:,1), scale*gpModel.training_x(:,2), 50.0, 'x', 'LineWidth', 1.5);
hold off;

end


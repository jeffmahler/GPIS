function [gpModel, shapeParams,img] = ...
    run_gpis_2D_experiment(dim, filename, dataDir, outputDir, newShape,varParams, trainingParams,experimentConfig)
%RUN_ANTIPODAL_EXPERIMENT Summary of this function goes here
%   Detailed explanation goes here

% create a new shape
if newShape
    [points, com] = new_shape(filename, dataDir, dim);
end

% create the tsdf
[shapeParams, shapeImage, points, com] =...
    create_tsdf(filename, dataDir, dim, varParams, experimentConfig.surfaceThresh);

% construct a gpis either from the full matrix or otherwise
if strcmp(trainingParams.activeSetMethod, 'Full') == 1

    startTime = tic;
    gpModel = create_gpis(shapeParams.points, shapeParams.tsdf, ...
        shapeParams.normals, shapeParams.noise, true, ...
        trainingParams.hyp, trainingParams.numIters);
    constructionTime = toc(startTime);
   
    [predGrid, predSurface] = predict_2d_grid(gpModel, shapeParams.gridDim,...
        trainingParams.surfaceThresh);
    
    [testImageDarkened, combImageBig] = create_tsdf_image(predGrid, scale);
    
    shapeParams.combImageBig; 
   
    tsdfReconError = ...
        evaluate_errors(predGrid.tsdf, shapeParams.fullTsdf, activeSetSize);
    normalError = ...
        evaluate_errors(predGrid.normals, shapeParams.fullNormals, activeSetSize);

else
    [gpModel, predGrid, tsdfReconError, normalError, surfaceImage, constructionTime] = ...
        create_sparse_gpis(shapeParams, trainingParams, trainingParams.scale);
    
    [testImageDarkened, combImageBig] = create_tsdf_image(predGrid, trainingParams.scale);
    img = struct(); 
    img.mean = testImageDarkened; 
    img.var = combImageBig;
    figure;
    imshow(testImageDarkened);
    
    
end
fprintf('Construction took %d seconds\n', constructionTime);

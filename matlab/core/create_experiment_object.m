function [gpModel, shapeParams, shapeSamples, constructionResults] = ...
    create_experiment_object(dim, filename, dataDir, newShape, ...
                             experimentConfig, varParams, trainingParams, scale)
% create a new shape
if newShape
    [points, com] = new_shape(filename, dataDir, dim);
end

% create the tsdf
[shapeParams, shapeImage, points, com] =...
    create_tsdf(filename, dataDir, dim, varParams, experimentConfig.surfaceThresh);

% construct a gpis either from the full matrix or otherwise
if strcmp(trainingParams.activeSetMethod, 'Full') == 1
    [gpModel, predGrid, tsdfReconError, normalError, surfaceImage, constructionTime] = ...
        create_dense_gpis(shapeParams, trainingParams, trainingParams.scale);
else
    [gpModel, predGrid, tsdfReconError, normalError, surfaceImage, constructionTime] = ...
        create_sparse_gpis(shapeParams, trainingParams, trainingParams.scale);
end

% sample shapes
startTime = tic;
[shapeSamples, pdfs] = ...
    sample_shapes(gpModel, shapeParams.gridDim, experimentConfig.numSamples);
samplingTime = toc(startTime);

fprintf('Sampled %d shapes in %f sec\n', experimentConfig.numSamples, samplingTime);


% fill in construction results
constructionResults = struct();
constructionResults.predGrid = predGrid;
constructionResults.tsdfReconError = tsdfReconError;
constructionResults.normalError = normalError;
constructionResults.surfaceImage = surfaceImage;
constructionResults.pdfs = pdfs;
constructionResults.constructionTime = constructionTime;
constructionResults.samplingTime = samplingTime;

% save GPIS and shape
shapeName = sprintf('%s/%s.mat', dataDir, filename);
gpisName = sprintf('%s/%s_gpis.mat', dataDir, filename);
samplesName = sprintf('%s/%s_samples.mat', dataDir, filename);
constructName = sprintf('%s/%s_construction.mat', dataDir, filename);
varName = sprintf('%s/%s_variance_params.mat', dataDir, filename);

save(shapeName, 'shapeParams');
save(gpisName, 'gpModel');
save(samplesName, 'shapeSamples');
save(constructName, 'constructionResults');
save(varName, 'varParams');

constructionResults.newSurfaceImage = ...
    create_tsdf_image_sampled(constructionResults.predGrid, ...
        shapeSamples, scale, 1.0);

end


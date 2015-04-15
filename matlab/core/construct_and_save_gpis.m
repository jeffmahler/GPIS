function [gpModel, shapeSamples, constructionResults] = ...
    construct_and_save_gpis(filename, dataDir, shapeParams, ...
                            trainingParams, numSamples, image_scale)

% construct a gpis either from the full matrix or otherwise
if strcmp(trainingParams.activeSetMethod, 'Full') == 1
    [gpModel, predGrid, tsdfReconError, normalError, surfaceImage, constructionTime] = ...
        create_dense_gpis(shapeParams, trainingParams, trainingParams.scale, ...
        trainingParams.downsample);
else
    [gpModel, predGrid, tsdfReconError, normalError, surfaceImage, constructionTime] = ...
        create_sparse_gpis(shapeParams, trainingParams, trainingParams.scale);
end

% sample shapes
startTime = tic;
%shapeSamples = {predGrid};
%pdfs = {};
[shapeSamples, pdfs] = ...
     sample_shapes(gpModel, shapeParams.gridDim, numSamples, ...
        trainingParams.useGradients, trainingParams.downsample);
samplingTime = toc(startTime);

fprintf('Sampled %d shapes in %f sec\n', numSamples, samplingTime);

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

save(shapeName, 'shapeParams');
save(gpisName, 'gpModel');
save(samplesName, 'shapeSamples');
save(constructName, 'constructionResults');

contrast = 1.0;
constructionResults.newSurfaceImage = ...
    create_tsdf_image_sampled(constructionResults.predGrid, ...
        shapeSamples, image_scale, contrast, false, false);
% figure(88);
% imshow(constructionResults.newSurfaceImage);
end


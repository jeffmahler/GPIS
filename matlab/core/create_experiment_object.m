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
[gpModel, shapeSamples, constructionResults] = ...
    construct_and_save_gpis(filename, dataDir, shapeParams, ...
                            trainingParams, experimentConfig.numSamples, ...
                            scale);

constructionResults.newSurfaceImage = ...
    create_tsdf_image_sampled(constructionResults.predGrid, ...
        shapeSamples, scale, 1.0, false, false);
figure(17);
imshow(constructionResults.newSurfaceImage);
                        
varName = sprintf('%s/%s_variance_params.mat', dataDir, filename);
save(varName, 'varParams');

end


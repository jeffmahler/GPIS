function [gpModel, shapeParams, shapeSamples, constructionResults] = ...
    create_experiment_object(dim, filename, dataDir, newShape, ...
                             experimentConfig, varParams, trainingParams, scale)
% create a new shape
if newShape
    [points, com] = new_shape(filename, dataDir, scale * dim, ...
        trainingParams.image);
end

% create the tsdf
use_com = 1;
[shapeParams, shapeImage, points, com] =...
    create_tsdf(filename, dataDir, dim, varParams, experimentConfig.surfaceThresh, ...
    use_com, scale);

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
                   
constructionResults.predGrid.com = ...
    mean(constructionResults.predGrid.points(constructionResults.predGrid.tsdf < 0,:));

meanSamples = {constructionResults.predGrid};
meanSurfaceImage = ...
    create_tsdf_image_sampled(constructionResults.predGrid, ...
        meanSamples, scale, 1.0, false, true);
figure(49);
meanSurfaceImage = high_res_surface(imresize(double(meanSurfaceImage), 0.5), scale / 2);
%meanSurfaceImage = imresize(meanSurfaceImage, 0.5);
imshow(meanSurfaceImage);
constructionResults.meanSurfaceImage = meanSurfaceImage;
    
varName = sprintf('%s/%s_variance_params.mat', dataDir, filename);
save(varName, 'varParams');

end


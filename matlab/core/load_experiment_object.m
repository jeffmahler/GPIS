function [gpModel, shapeParams, shapeSamples, constructionResults] = ...
    load_experiment_object(filename, dataDir, scale)

% load GPIS and shape
shapeName = sprintf('%s/%s.mat', dataDir, filename);
gpisName = sprintf('%s/%s_gpis.mat', dataDir, filename);
samplesName = sprintf('%s/%s_samples.mat', dataDir, filename);
constructName = sprintf('%s/%s_construction.mat', dataDir, filename);
%varName = sprintf('%s/%s_variance_params.mat', dataDir, filename);

load(shapeName, 'shapeParams');
load(gpisName, 'gpModel');
load(samplesName, 'shapeSamples');
load(constructName, 'constructionResults');
%load(varName, 'varParams');

% hack to fix bug earlier
if ~isfield(shapeParams, 'com')
    shapeParams.com = ...
        mean(shapeParams.points(shapeParams.tsdf < 0,:));
end

constructionResults.predGrid.com = ...
    mean(constructionResults.predGrid.points(constructionResults.predGrid.tsdf < 0,:));

constructionResults.newSurfaceImage = ...
    create_tsdf_image_sampled(constructionResults.predGrid, ...
        shapeSamples, scale, 1.0, false, false);
    
figure;
imshow(constructionResults.newSurfaceImage);
end


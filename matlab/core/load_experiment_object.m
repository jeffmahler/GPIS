function [gpModel, shapeParams, shapeSamples, constructionResults] = ...
    load_experiment_object(filename, dataDir, scale)

% load GPIS and shape
shapeName = sprintf('%s/%s.mat', dataDir, filename);
gpisName = sprintf('%s/%s_gpis.mat', dataDir, filename);
samplesName = sprintf('%s/%s_samples.mat', dataDir, filename);
constructName = sprintf('%s/%s_construction.mat', dataDir, filename);
%varName = sprintf('%s/%s_variance_params.mat', dataDir, filename);

gpModel = {};
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

meanSamples = {constructionResults.predGrid};
meanSurfaceImage = ...
    create_tsdf_image_sampled(constructionResults.predGrid, ...
        meanSamples, scale, 1.0, false, false);
figure(49);
imshow(meanSurfaceImage);

constructionResults.newSurfaceImage = ...
    create_tsdf_image_sampled(constructionResults.predGrid, ...
        shapeSamples, scale, 1.0, false, false);
    
figure(50);
imshow(constructionResults.newSurfaceImage);

 % create struct for nominal shape
tsdfGrid = reshape(shapeParams.fullTsdf, [shapeParams.gridDim, shapeParams.gridDim]);
a = imresize(tsdfGrid, 10.0);
csvwrite('shape.csv', a);
G = fspecial('gaussian', [3, 3], 1.5);
tsdfGridSmooth = imfilter(tsdfGrid, G, 'same', 'replicate');
[smoothGx, smoothGy] = imgradientxy(tsdfGridSmooth, 'CentralDifference');

nominalShape = struct();
nominalShape.tsdf = tsdfGridSmooth(:);
nominalShape.normals = [smoothGx(:), smoothGy(:)];
nominalShape.points = shapeParams.all_points;
nominalShape.noise = zeros(size(nominalShape.tsdf,1), 1);
nominalShape.gridDim = shapeParams.gridDim;
nominalShape.surfaceThresh = shapeParams.surfaceThresh;
nominalShape.com = shapeParams.com;

win = 1;
sig = 0.001;
tsdfGrid = reshape(nominalShape.tsdf, [nominalShape.gridDim, nominalShape.gridDim,]);
nominalShape.shapeImage = high_res_tsdf(tsdfGrid, scale, win, sig);

figure(51);
imshow(nominalShape.shapeImage);

end


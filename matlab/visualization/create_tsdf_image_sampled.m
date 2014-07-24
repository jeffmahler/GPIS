function [shapeImageScaled, shapeSurfaceImage] = ...
    create_tsdf_image_sampled(shapeParams, shapeSamples, scale, black)

if nargin < 4
    black = false;
end

% Alpha blend all samples
numSamples = size(shapeSamples,2);
dim = scale*shapeParams.gridDim;
shapeImage = zeros(dim, dim);
shapeSurfaceImage = zeros(dim, dim);
alpha = 1.0 / double(numSamples);
w = 0;
w_total = 0;
j = 1;
for i = 1:numSamples
    tsdf = shapeSamples{i}.tsdf;
    tsdfGrid = reshape(tsdf, shapeParams.gridDim, shapeParams.gridDim);
    tsdfBig = imresize(tsdfGrid, scale);
    
    
    shapeSurfaceIndices = find(abs(tsdfBig) < shapeParams.surfaceThresh);
    tsdfSurface = ones(dim, dim);
    tsdfSurface(shapeSurfaceIndices) = 0;
%     if mod(i,51) == 0
% %         figure(111);
% %         subplot(1,4,j);
% %         imshow(abs(tsdfBig));
%         w = alpha;
%         j = j+1;
%     else
%         w = 0;
%     end
    w = alpha;
    shapeImage = shapeImage + w * tsdfBig;
    shapeSurfaceImage = shapeSurfaceImage + w * tsdfSurface;
    w_total = w_total + w;
end
shapeImage = shapeImage ./ w_total;
shapeSurfaceImage = shapeSurfaceImage ./ w_total;

% display many different scaling versions to check best visualization
% beta = 0.1;
% max_beta = 2.0;
% beta_inc = 0.25;
% num_beta = round((max_beta - beta) / beta_inc);
% k = 1;
% 
% figure(5);
% while beta < max_beta
%     subplot(1, num_beta, k);
%     shapeImageScaled = abs(shapeImage).^beta;
%     shapeImageScaled = (shapeImageScaled - min(min(shapeImageScaled))) / ...
%         (max(max(shapeImageScaled)) - min(min(shapeImageScaled)));
% 
%     imshow(shapeImageScaled);
%     title(sprintf('BETA = %f', beta));
%     k = k+1;
%     beta = beta + beta_inc;
% end

% rescale shapeImage so that the zero crossing appears black
beta = 0.35;
shapeImageScaled = abs(shapeImage).^beta;

% normalize
shapeImageScaled = (shapeImageScaled - min(min(shapeImageScaled))) / ...
    (max(max(shapeImageScaled)) - min(min(shapeImageScaled)));
shapeSurfaceImage = (shapeSurfaceImage - min(min(shapeSurfaceImage))) / ...
    (max(max(shapeSurfaceImage)) - min(min(shapeSurfaceImage)));

if black
    shapeImageScaled = max(max(shapeImageScaled))*ones(dim, dim) - shapeImageScaled;
    shapeSurfaceImage = max(max(shapeSurfaceImage))*ones(dim, dim) - shapeSurfaceImage;
end

figure(4);
subplot(1,2,1);
imshow(shapeImageScaled);
title('Avg Scaled Tsdfs');
subplot(1,2,2);
imshow(shapeSurfaceImage);
title('Avg Tsdf Zero Crossings');

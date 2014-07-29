function shapeSurfaceImage = ...
    create_tsdf_image_sampled(shapeParams, shapeSamples, scale, contrastRatio, black, vis)

if nargin < 4
    contrastRatio = 0.7;
end
if nargin < 5
    black = false;
end
if nargin < 6
    vis = false;
end

% Alpha blend all samples
numSamples = size(shapeSamples,2);
dim = scale*shapeParams.gridDim;
shapeSurfaceImage = zeros(dim, dim);
alpha = contrastRatio / double(numSamples);

for i = 1:numSamples
    tsdf = shapeSamples{i}.tsdf;
    tsdfGrid = reshape(tsdf, shapeParams.gridDim, shapeParams.gridDim);
    tsdfBig = imresize(tsdfGrid, scale);
    
    
    shapeSurfaceIndices = find(abs(tsdfBig) < shapeParams.surfaceThresh);
    tsdfSurface = ones(dim, dim);
    tsdfSurface(shapeSurfaceIndices) = 0;
    
    if vis
        figure(1);
        imshow(tsdfSurface);
        pause(0.5);
    end
    
    shapeSurfaceImage = shapeSurfaceImage + alpha * tsdfSurface;
end
shapeSurfaceImage = shapeSurfaceImage + (1.0 - contrastRatio) * zeros(dim, dim);

% figure;
% subplot(1,4,1);
% w = shapeSurfaceImage.^3;
% imshow(w);
% 
% subplot(1,4,2);
% f = histeq(shapeSurfaceImage, 100);
% x = f;
% x(x == 0) = 5e-3; % remove 0 values
% G = fspecial('gaussian',[5 5], 0.5);
% y = imfilter(x, G, 'same');
% a = y.^0.15;
% imshow(a);
% 
% subplot(1,4,3);
% G = fspecial('gaussian',[5 5], 0.5);
% Ig = imfilter(shapeSurfaceImage, G, 'same');
% fp = histeq(Ig,100);
% imshow(fp);
% 
% subplot(1,4,4);
% q = 0.6*w + 0.4*a;
% imshow(q);

gamma = 3;
beta = 0.15;
sig = 0.5;
nbins = 100;
blend = 0.6;
siContrastEnhanced = shapeSurfaceImage.^gamma;
siEqualized = histeq(shapeSurfaceImage, nbins);
siEqualized(siEqualized == 0) = 5e-3; % remove 0 values
G = fspecial('gaussian',[5 5], sig);
siEqualizedFilt = imfilter(siEqualized, G, 'same');
siEqFlat = siEqualizedFilt.^beta;

shapeSurfaceImage = blend * siContrastEnhanced + (1 - blend) * siEqFlat;
% figure;
% imshow(shapeSurfaceImage);

% normalize the values to 0 and 1
% shapeSurfaceImage =apeSurfaceImage - min(min(shapeSurfaceImage))) / ...
%     (max(max(sha (shpeSurfaceImage)) - min(min(shapeSurfaceImage)));

if black
    shapeSurfaceImage = max(max(shapeSurfaceImage))*ones(dim, dim) - shapeSurfaceImage;
end

if false
    figure(4);
    % subplot(1,2,1);
    % imshow(shapeImageScaled);
    % title('Avg Scaled Tsdfs');
    % subplot(1,2,2);
    imshow(shapeSurfaceImage);
    %title('Avg Tsdf Zero Crossings');
end

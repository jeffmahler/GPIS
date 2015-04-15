function shapeSurfaceImage = ...
    create_tsdf_image_sampled(shapeParams, shapeSamples, scale, contrastRatio, black, vis, heq)

if nargin < 4
    contrastRatio = 0.7;
end
if nargin < 5
    black = false;
end
if nargin < 6
    vis = false;
end
if nargin < 7
    heq = true;
end
% Alpha blend all samples
sample_dim = shapeParams.gridDim;
numSamples = size(shapeSamples,2);
dim = 2 * sample_dim;
hd_dim = scale / 2 * dim;
shapeSurfaceImage = zeros(dim, dim);
alpha = contrastRatio / double(numSamples);

%writerObj = VideoWriter('results/shape_samples/noise_level_010.avi');
%open(writerObj);

for i = 1:numSamples
    tsdf = shapeSamples{i}.tsdf;
    tsdfGrid = reshape(tsdf, sample_dim, sample_dim);
    tsdfBig = imresize(tsdfGrid, 2);
    
    % find surface...
    tsdfThresh = tsdfBig > 0;
    SE = strel('square', 3);
    I_d = imdilate(tsdfThresh, SE);
    
    % create border masks
    insideMaskOrig = (tsdfThresh == 0);
    outsideMaskDi = (I_d == 1);
    tsdfSurface = double(~(outsideMaskDi & insideMaskOrig));
    
    shapeSurfaceIndices = find(abs(tsdfBig) < shapeParams.surfaceThresh);
    tsdfSurfaceThresh = ones(dim, dim);
    tsdfSurfaceThresh(shapeSurfaceIndices) = 0;
    
    tsdfSurface = tsdfSurface & tsdfSurfaceThresh;
    
    if vis
        figure(100);
        H = high_res_surface(double(tsdfSurface), scale / 2);
        imshow(H);
%         frame = getframe;
%         for a = 1:20
%             writeVideo(writerObj,frame);
%         end
        pause(0.5);
    end
    
    shapeSurfaceImage = shapeSurfaceImage + alpha * tsdfSurface;
end
shapeSurfaceImage = shapeSurfaceImage + (1.0 - contrastRatio) * zeros(dim, dim);
%close(writerObj);

if heq
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
shapeSurfaceImage = high_res_gpis(shapeSurfaceImage, scale / 2);
end
% figure;
% imshow(shapeSurfaceImage);

% normalize the values to 0 and 1
% shapeSurfaceImage =apeSurfaceImage - min(min(shapeSurfaceImage))) / ...
%     (max(max(sha (shpeSurfaceImage)) - min(min(shapeSurfaceImage)));

if black
    shapeSurfaceImage = max(max(shapeSurfaceImage))*ones(hd_dim, hd_dim) - shapeSurfaceImage;
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

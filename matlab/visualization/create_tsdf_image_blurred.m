function [shapeImage, surfaceImage] = ...
    create_tsdf_image_blurred(shapeParams, scale, black)

if nargin < 3
    black = false;
end


% Alpha blend all samples
dim = scale*shapeParams.gridDim;
shapeImage = zeros(dim, dim);
surfaceImage = zeros(dim, dim);
varGrid= reshape(shapeParams.noise, shapeParams.gridDim, shapeParams.gridDim);
varScaled = imresize(varGrid, scale);
varScaled(varScaled < 0) = 1e-6;

beta = 0.35;
meanShapeImage = reshape(shapeParams.tsdf, shapeParams.gridDim, shapeParams.gridDim);
meanShapeImage = imresize(meanShapeImage, scale);

meanSurfaceImage = ones(dim, dim);
meanSurfaceImage(abs(meanShapeImage) < shapeParams.surfaceThresh) = 0;
    
meanShapeImage = abs(meanShapeImage).^beta;
meanShapeImage = (meanShapeImage - min(min(meanShapeImage))) / ...
        (max(max(meanShapeImage)) - min(min(meanShapeImage)));

win = 17;
h = floor(win/2);
meanShapeImagePadded = padarray(meanShapeImage, [h, h], 'replicate');
meanSurfaceImagePadded = padarray(meanSurfaceImage, [h, h], 'replicate');

gamma = 6.0;
gamma_inc = 2.0;
max_gamma = 8.0;
num_gamma = round((max_gamma - gamma) / gamma_inc);
k = 1;
while gamma < max_gamma

    for i = h+1:dim+h
        for j = h+1:dim+h
            i_low = max(0, i-h);
            i_high = min(h+1+dim, i+h);
            j_low = max(0, j-h);
            j_high = min(h+1+dim, j+h);
            g = fspecial('gaussian', win, gamma*varScaled(i-h,j-h));
            W = meanShapeImagePadded(i_low:i_high, j_low:j_high);
            q = conv2(W, g, 'same');
            shapeImage(i-h,j-h) = q(h+1,h+1);
            
            W = meanSurfaceImagePadded(i_low:i_high, j_low:j_high);
            q = conv2(W, g, 'same');
            surfaceImage(i-h,j-h) = q(h+1,h+1);
        end
    end
%     figure(6);
%     subplot(1,num_gamma,k);
%     imshow(shapeImage);
%     title(sprintf('Gamma = %f', gamma));
%     
    figure(9);
    subplot(1,num_gamma,k);
    imshow(surfaceImage);
    title(sprintf('Gamma = %f', gamma));
    
    k = k+1;
    gamma = gamma + gamma_inc;
end

if black
    shapeImage = max(max(shapeImage))*ones(dim, dim) - shapeImage;
    surfaceImage = max(max(surfaceImage))*ones(dim, dim) - surfaceImage;
end

figure(7);
subplot(1,2,1);
imshow(meanShapeImage);
title('Mean');
subplot(1,2,2);
imshow(shapeImage);
title('Blurred');




function [tsdfAbsImage, varColoredImage] = create_tsdf_image(shapeParams, scale)

% Create color image
numTest = size(shapeParams.points, 1);
testColors = repmat(ones(numTest,1) - ...
    abs(shapeParams.tsdf) / max(abs(shapeParams.tsdf)) .* ones(numTest,1), 1, 3);

testImage = reshape(testColors(:,1), shapeParams.gridDim, shapeParams.gridDim); 
testImage = imresize(testImage, scale*size(testImage));
testImageDarkened = max(0, testImage - ...
                           0.3*ones(scale*shapeParams.gridDim, scale*shapeParams.gridDim)); % darken
tsdfAbsImage = testImageDarkened;

colorImage = 255*ones(shapeParams.gridDim, shapeParams.gridDim, 'uint8');
tsdfImage = imresize(testImageDarkened, 0.5);
allVarsGrid = reshape(shapeParams.noise, shapeParams.gridDim, shapeParams.gridDim);

% take percentiles for coloring to prevent outliers from messing up vis 
minVar = prctile(shapeParams.noise(abs(shapeParams.tsdf) < 0.1), 10);
maxVar = prctile(shapeParams.noise(abs(shapeParams.tsdf) < 0.1), 90);
varImage = zeros(shapeParams.gridDim, shapeParams.gridDim, 3, 'uint8');
maxVarIm = (maxVar - allVarsGrid) / (1*(maxVar - minVar)) .* double(colorImage);
maxVarIm(maxVarIm < 0) = 0;
maxVarIm(maxVarIm > 255) = 255;
minVarIm = (allVarsGrid - minVar) / (1*(maxVar - minVar)) .* double(colorImage);
minVarIm(minVarIm < 0) = 0;
minVarIm(minVarIm > 255) = 255;

varImage(:,:,2) = uint8(0*colorImage) + uint8(maxVarIm);
varImage(:,:,1) = uint8(0*colorImage) + uint8(minVarIm);
combImage = zeros(shapeParams.gridDim, shapeParams.gridDim, 3, 'uint8');
combImage(:,:,1) = uint8(double(varImage(:,:,1)) .* tsdfImage);
combImage(:,:,2) = uint8(double(varImage(:,:,2)) .* tsdfImage);
varColoredImage = imresize(combImage, scale);

end


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
minVar = min(min(allVarsGrid));
maxVar = max(max(allVarsGrid));
varImage = zeros(shapeParams.gridDim, shapeParams.gridDim, 3, 'uint8');
varImage(:,:,2) = uint8(0*colorImage) + uint8((maxVar - allVarsGrid) / (1*(maxVar - minVar)) .* double(colorImage));
varImage(:,:,1) = uint8(0*colorImage) + uint8((allVarsGrid - minVar) / (1*(maxVar - minVar)) .* double(colorImage));
combImage = zeros(shapeParams.gridDim, shapeParams.gridDim, 3, 'uint8');
combImage(:,:,1) = uint8(double(varImage(:,:,1)) .* tsdfImage);
combImage(:,:,2) = uint8(double(varImage(:,:,2)) .* tsdfImage);
varColoredImage = imresize(combImage, scale);

end


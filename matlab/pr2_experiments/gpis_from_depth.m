function [gpModel, shapeSamples, constructionResults, transformResults] = ...
    gpis_from_depth(sourceDir, numImages, cfg, trainingParams)

cbSquareMeters = cfg.cbSquareMeters;
depthThresh = cfg.depthThresh;
truncation = cfg.truncation;
noiseScale = cfg.noiseScale;
disagreePenalty = cfg.disagreePenalty;
gridDim = cfg.gridDim;
numSamples = cfg.numSamples;
scale = cfg.scale;
objName = cfg.objName;
insideScale = cfg.insideScale;
colorSlack = cfg.colorSlack;

depthImages = cell(1, numImages);
cbPoses = cell(1, numImages);

% read in all depths / poses
for i = 0:(numImages-1)
    fprintf('Reading image %d\n', i);
    depthFilename = sprintf('%s/depth_%d.csv', sourceDir, i);
    poseFilename = sprintf('%s/pr2_cb_transform_%d.csv', sourceDir, i);
    depthImages{i+1} = csvread(depthFilename);
%     figure(i+1);
%     imshow(uint8(255*histeq(depthImages{i+1})));
    cbPoses{i+1} = csvread(poseFilename);
end

[depthHeight, depthWidth] = size(depthImages{1});

% read in camera matrices and RGB image
rgbFilename = sprintf('%s/rgb_0.png', sourceDir);
rgbImage = imread(rgbFilename);

KrgbFilename = sprintf('%s/K_rgb.csv', sourceDir);
Krgb = csvread(KrgbFilename);
Krgb = reshape(Krgb, [3, 3])';

KdFilename = sprintf('%s/K_depth.csv', sourceDir);
Kd = csvread(KdFilename);
Kd = reshape(Kd, [3, 3])';

% get user-selected grid coordinates 
figure(60);
imshow(rgbImage);
hold on;
disp('Click the top-left corner of the grid, followed by the bottom-right corner');
[x, y] = ginput(2);

gridStartPix = [x(1); y(1)];
gridWidthPix = max(abs(x(2) - x(1)), abs(y(2) - y(1)));
gridHeightPix = gridWidthPix;

% get user-selected scales
disp('Click the top-left and top-right corner of chessboard square');
[x, y] = ginput(2);
cbSquarePix = abs(x(2) - x(1));
pixTo3D = cbSquareMeters / cbSquarePix;

% get user-selected center
disp('Click the center of the chessboard');
[x, y] = ginput(1);
centerCbPix = [x(1); y(1)];
scatter(centerCbPix(1), centerCbPix(2), 20, 'r'); % diplay center for debugging

% form cb to grid transformation in SE(3)
% NOTE: grid coordinates are located at top-left corner with image X and Y

% get 3d diff
diffGridFrame = pixTo3D * (gridStartPix - centerCbPix);
gridCbFrame = [-diffGridFrame(2); -diffGridFrame(1); 0];

% form transformation
R = [ 0, -1, 0;
     -1,  0, 0;
      0,  0, -1];
t = -R' * gridCbFrame;
T_cb_grid = [R, t;
             0, 0, 0, 1]; 
%T_cb_grid = inv(T_cb_grid);

scatter(centerCbPix(1) + (1.0 / pixTo3D) * diffGridFrame(1), ...
        centerCbPix(2) + (1.0 / pixTo3D) * diffGridFrame(2), 20, 'r'); % diplay center for debugging

% average depth images together (basically, assume zero-mean Gaussian noise on depth)
combinedDepth = zeros(depthHeight, depthWidth);
validDepthCounts = zeros(depthHeight, depthWidth);
for i = 1:numImages
    depthNansRemoved = depthImages{i};
    depthNansRemoved(isnan(depthNansRemoved)) = 0;
    validDepthCounts = validDepthCounts + (~isnan(depthImages{i}));
    combinedDepth = combinedDepth + (~isnan(depthImages{i}) .* depthNansRemoved);
end
combinedDepth = combinedDepth ./ validDepthCounts;
combinedDepth(isinf(combinedDepth)) = nan; % remove infs from divide by zero

% figure(61);
% imshow(combinedDepth);
%  
% average cb poses (but not now)
T_cam_cb = inv(cbPoses{1}); % want camera in cb frame

% project grid region of depth image into 3d space
[Ucoords, Vcoords] = meshgrid(1:gridWidthPix, 1:gridHeightPix);
Ucoords = Ucoords + gridStartPix(1);
Vcoords = Vcoords + gridStartPix(2);
gridImageCoords = [Ucoords(:), Vcoords(:), ones(gridHeightPix*gridWidthPix, 1)]';
gridDepths = combinedDepth(gridStartPix(2):(gridStartPix(2) + gridHeightPix - 1), ...
                           gridStartPix(1):(gridStartPix(1) + gridWidthPix  - 1));
gridDepths = gridDepths(:)';
gridDepthMult = repmat(gridDepths, 3, 1);

% create cropped images
depthCrop = combinedDepth(gridStartPix(2):(gridStartPix(2) + gridHeightPix - 1), ...
                           gridStartPix(1):(gridStartPix(1) + gridWidthPix  - 1));
rgbCrop = rgbImage(gridStartPix(2):(gridStartPix(2) + gridHeightPix - 1), ...
                 gridStartPix(1):(gridStartPix(1) + gridWidthPix  - 1));                       
                       
grid3Dcam = inv(Kd) * (gridDepthMult .* gridImageCoords);

% transform points into cb frame
grid3DcamHomog = [grid3Dcam; ones(1, size(grid3Dcam, 2))];
grid3Dcb = T_cam_cb * grid3DcamHomog;
figure;
scatter3(grid3Dcb(1, :), grid3Dcb(2, :), grid3Dcb(3, :), 25);

% segment image based on depth (should be flat on z in cb frame)
objectPointMask = grid3Dcb(3,:) > depthThresh;
objectPoints3Dcb = grid3Dcb(:,objectPointMask);

figure;
scatter3(objectPoints3Dcb(1, :), objectPoints3Dcb(2, :), objectPoints3Dcb(3, :), 25);

% transform into grid frame
objectPoints3Dgrid = T_cb_grid * objectPoints3Dcb;

figure;
scatter3(objectPoints3Dgrid(1, :), objectPoints3Dgrid(2, :), objectPoints3Dgrid(3, :), 25);

% convert back to pixels for grid purposes
objectPoints2Dgrid = objectPoints3Dgrid(1:2,:);
objectPointsPix = (1.0 / pixTo3D) * objectPoints2Dgrid;
objectPointsPix = round(objectPointsPix);

objectMask = zeros(gridHeightPix, gridWidthPix);
for i = 1:size(objectPointsPix, 2)
    if objectPointsPix(2,i) > 0 && objectPointsPix(2,i) < gridHeightPix && ...
            objectPointsPix(1,i) > 0 && objectPointsPix(1,i) < gridWidthPix
        objectMask(objectPointsPix(2,i), objectPointsPix(1,i)) = 1;
    end
end

% display object mask and rgb crop for testing
% figure;
% subplot(1,2,1);
% imshow(objectMask);
% subplot(1,2,2);
% imshow(rgbCrop);


% show the depth inaccuracy
figure(19);
subplot(1,2,1);
imshow(0.3*uint8(255*histeq(depthCrop)) + 0.7*rgbCrop);
subplot(1,2,2);
imshow(0.5*uint8(255*objectMask) + 0.5*rgbCrop);

% compute an intensity based color segmentation (assume corners are on
% background) 
backGroundColor = rgbCrop(4, 4);
rgbMask = rgbCrop < backGroundColor - colorSlack;

figure(21);
subplot(1,3,1);
imshow(rgbMask);
title('RGB Object Mask');
subplot(1,3,2);
imshow(objectMask);
title('Depth Object Mask');
subplot(1,3,3);
imshow(rgbMask | objectMask);
title('Combined Mask');

% compute a signed distance function using brute force
sdfDepthGrid = realmax * ones(gridHeightPix, gridWidthPix) .* (~objectMask);
%sdfRgbGrid = realmax * ones(gridHeightPix, gridWidthPix) .* (~rgbMask);
for i = 1:gridWidthPix
    for j = 1:gridHeightPix
        point = [i; j];
        
        % fill depth sdf
        if objectMask(j,i) == 0
            for k = 1:size(objectPointsPix,2)
                % check outside region
                signedDist = norm(point - objectPointsPix(:,k));
                if 0 < signedDist && signedDist < sdfDepthGrid(j,i)
                    sdfDepthGrid(j,i) = min(signedDist, truncation); 
                end
            end
        else
            winX = [max(1, point(1) - 1), min(gridWidthPix, point(1) + 1)];
            winY = [max(1, point(2) - 1), min(gridHeightPix, point(2) + 1)];

            % check on border
            if sum(sum(sdfDepthGrid(winY(1):winY(2), winX(1):winX(2)))) > 0
                sdfDepthGrid(point(2), point(1)) = 0;
            else
                sdfDepthGrid(point(2), point(1)) = -realmax;
                for k = 1:size(objectPointsPix,2)
                    otherPoint = objectPointsPix(:,k);
                    signedDist =  insideScale * (-norm(point - otherPoint));
                    if signedDist < 0 && signedDist > sdfDepthGrid(point(2), point(1))
                        sdfDepthGrid(point(2), point(1)) = signedDist;
                    end
                end  
            end
        end
    end
end

% figure(23);
% imshow(sdfDepthGrid);

% create uncertainty grid (noiseScale is from 1 measurement)
countCrop = validDepthCounts(gridStartPix(2):(gridStartPix(2) + gridHeightPix - 1), ...
                             gridStartPix(1):(gridStartPix(1) + gridWidthPix  - 1));
                      
% now resize to desired res
gpScale = double(gridDim) / gridWidthPix;
sdfGrid = imresize(sdfDepthGrid, gpScale);
countCrop = imresize(countCrop, gpScale);
objectMask = imresize(objectMask, gpScale);
rgbMask = imresize(rgbMask, gpScale);

objectMask = objectMask > 0.25;
rgbMask = rgbMask > 0.25;

countCrop = round(countCrop);
countCrop(countCrop < 0) = 0;

% create uncertainty mask
uncertaintyGrid = ones(gridDim, gridDim);
uncertaintyGrid = noiseScale * uncertaintyGrid ./ countCrop;

% multiplier for disagreement with rgb
disagreementMask = (objectMask & ~rgbMask) | (~objectMask & rgbMask);
uncertaintyGrid = ...
    disagreePenalty * (disagreementMask == 1) .* uncertaintyGrid + ...
    (disagreementMask == 0) .* uncertaintyGrid;

%sdfGrid(disagreementMask) = -1.0;
[sdfGx, sdfGy] = imgradientxy(sdfGrid, 'CentralDifference');

% figure(46);
% imshow(disagreementMask);

[Xcoord, Ycoord] = meshgrid(1:gridDim, 1:gridDim);
validIndices = ~isinf(uncertaintyGrid) & ~isnan(uncertaintyGrid);

% create shape params
shapeParams = struct();
shapeParams.gridDim = gridDim;
shapeParams.tsdf = sdfGrid(validIndices);
shapeParams.noise = uncertaintyGrid(validIndices);
shapeParams.normals = [sdfGx(validIndices), sdfGy(validIndices)];
shapeParams.points = [Xcoord(validIndices), Ycoord(validIndices)];
shapeParams.all_points = [Xcoord(:), Ycoord(:)];
shapeParams.fullTsdf = sdfGrid(:);
shapeParams.fullNormals = [sdfGx(:), sdfGy(:)];
shapeParams.surfaceThresh = trainingParams.surfaceThresh;

[gpModel, shapeSamples, constructionResults] = ...
    construct_and_save_gpis(objName, sourceDir, shapeParams, ...
                            trainingParams, numSamples, scale);

transformResults = struct();
transformResults.T_cb_grid = T_cb_grid;
transformResults.pixTo3D = pixTo3D;
transformResults.gpScale = gpScale;
                        
end


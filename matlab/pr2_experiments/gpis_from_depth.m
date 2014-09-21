function gpModel = gpis_from_depth(sourceDir, numImages, cfg)

cbSize = cfg.cbSize;
cbSquareMeters = cfg.cbSquareMeters;
depthThresh = cfg.depthThresh;
truncation = cfg.truncation;

depthImages = cell(1, numImages);
cbPoses = cell(1, numImages);

% read in all depths / poses
for i = 0:(numImages-1)
    fprintf('Reading image %d\n', i);
    depthFilename = sprintf('%s/depth_%d.csv', sourceDir, i);
    poseFilename = sprintf('%s/pr2_cb_transform_%d.csv', sourceDir, i);
    depthImages{i+1} = csvread(depthFilename);
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
gridWidthPix = abs(x(2) - x(1));
gridHeightPix = abs(y(2) - y(1));

% get user-selected scales
disp('Click the top-left and top-right corner of chessboard square');
[x, y] = ginput(2);
tlCbCorner = [x(1); y(1)];
cbSquarePix = abs(x(2) - x(1));
pixTo3D = cbSquareMeters / cbSquarePix;

% form cb to grid transformation in SE(3)
% NOTE: grid coordinates are located at top-left corner with image X and Y
% axis convention (X to right, Y to bottom)
tlToCenterCbSquares = [(cbSize(2) + 1) / 2; (cbSize(1) + 1) / 2];
centerCbPix = tlCbCorner + cbSquarePix * tlToCenterCbSquares;
scatter(centerCbPix(1), centerCbPix(2), 20, 'r'); % diplay center for debugging

% get 3d diff
diffGridFrame = pixTo3D * (gridStartPix - centerCbPix);

% form transformation
R = [ 0, -1, 0;
     -1,  0, 0;
      0,  0, -1];
t = [-diffGridFrame(2); -diffGridFrame(1); 0];
T_cb_grid = [R, t;
             0, 0, 0, 1]; 
T_cb_grid = inv(T_cb_grid);

% average depth images together (basically, assume zero-mean Gaussian noise on depth)
combinedDepth = zeros(depthHeight, depthWidth);
validDepthCounts = zeros(depthHeight, depthWidth);
for i = 1:numImages
    validDepthCounts = validDepthCounts + (~isnan(depthImages{i}));
    combinedDepth = combinedDepth + (~isnan(depthImages{i})) .* depthImages{i};
end
combinedDepth = combinedDepth ./ validDepthCounts;
combinedDepth(isinf(combinedDepth)) = nan; % remove infs from divide by zero

figure(61);
imshow(combinedDepth);
 
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

grid3Dcam = inv(Kd) * (gridDepthMult .* gridImageCoords);

% transform points into cb frame
grid3DcamHomog = [grid3Dcam; ones(1, size(grid3Dcam, 2))];
grid3Dcb = T_cam_cb * grid3DcamHomog;

% now segment image
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
    objectMask(objectPointsPix(2,i), objectPointsPix(1,i)) = 1;
end

figure;
imshow(objectMask);

% compute a signed distance function using brute force
sdfGrid = realmax * ones(gridHeightPix, gridWidthPix) .* (~objectMask);
for i = 1:gridWidthPix
    for j = 1:gridHeightPix
        point = [i; j];
        if objectMask(j,i) == 0
            for k = 1:size(objectPointsPix,2)
                % check outside region
                signedDist = norm(point - objectPointsPix(:,k));
                if 0 < signedDist && signedDist < sdfGrid(j,i)
                    sdfGrid(j,i) = min(signedDist, truncation); 
                end
            end
        else
            winX = [max(1, point(1) - 1), min(gridWidthPix, point(1) + 1)];
            winY = [max(1, point(2) - 1), min(gridHeightPix, point(2) + 1)];

            % check on border
            if sum(sum(sdfGrid(winY(1):winY(2), winX(1):winX(2)))) > 0
                sdfGrid(point(2), point(1)) = 0;
            else
                sdfGrid(point(2), point(1)) = -realmax;
                for k = 1:size(objectPointsPix,2)
                    otherPoint = objectPointsPix(:,k);
                    signedDist = -norm(point - otherPoint);
                    if signedDist < 0 && signedDist > sdfGrid(point(2), point(1))
                        sdfGrid(point(2), point(1)) = signedDist;
                    end
                end  
            end
        end
    end
end
    
% for i = 1:size(objectPointsPix,2)
%     point = objectPointsPix(:,i);
%     winX = [max(1, point(1) - 1), min(gridWidthPix, point(1) + 1)];
%     winY = [max(1, point(2) - 1), min(gridHeightPix, point(2) + 1)];
% 
%     % check on border
%     if sum(sum(sdfGrid(winY(1):winY(2), winX(1):winX(2)))) > 0
%         sdfGrid(point(2), point(1)) = 0;
%     else
%         sdfGrid(point(2), point(1)) = -realmax;
%         for j = 1:size(objectPointsPix,2)
%             otherPoint = objectPointsPix(:,j);
%             signedDist = -norm(point - otherPoint);
%             if signedDist > sdfGrid(point(2), point(1))
%                 sdfGrid(point(2), point(1)) = signedDist;
%             end
%         end  
%     end
% end

figure;
imshow(sdfGrid);

gpModel = {};

end


function [shapeParams, shapeIm] = auto_tsdf(shape, dim, pts, varParams, com, scale)
% Displays a random item of the specified shape and automatically extracs
% ALL points on the surface, inside the surface, and outside the surface
% using simple image processing ops.
% Available shapes are:
%   'Rectangles'
%   'Lines'
%   'Polygons'
%   'Circles'

if nargin < 2
    dim = 100;
end
if nargin < 3
   pts = [];
end

useCom = false;
if nargin > 5 && ndims(com) > 1
   useCom = true;
else
   com = [];
end

dim = dim * scale;
I = 255*ones(dim, dim);
S = vision.ShapeInserter;
S.Shape = shape;
S.Fill = true;
S.FillColor = 'Black';

% generate random shape properties for display
if strcmp(shape, 'Rectangles') == 1
    disp('Generating random rectangle');
    % fixed center, random width and height
    minDim = 20;
    x = 30;
    y = 30;
    width = uint16((dim - x - minDim) * rand() + minDim);
    height = uint16((dim - y - minDim) * rand() + minDim);
    pts = [x, y, width, height];
elseif strcmp(shape, 'Lines') == 1
    disp('Generating random line');
    % random endpoints
    x1 = uint16(dim*rand());
    y1 = uint16(dim*rand()); 
    x2 = uint16(dim*rand());
    y2 = uint16(dim*rand());
    pts = [x1, y1, x2, y2];
elseif strcmp(shape, 'Polygons') == 1
    randomPoly = false;
    if randomPoly
        minVertices = 3;
        maxVertices = 8;
        numVertices = uint16((maxVertices - minVertices) * rand() + minVertices);
        fprintf('Generating random polygon with %d vertices\n', numVertices);

        % generate points for top half of image
        xTop = [];
        yTop = [];
        for i =1:uint16((numVertices/2)-1)
            xNew = uint16(dim * rand());
            yNew = uint16((dim / 2) * rand());
            xTop = [xTop xNew];
            yTop = [yTop yNew];
        end
        xTop = sort(xTop, 'ascend');

        % generate points for top half of image
        xBottom = [];
        yBottom = [];
        for i = uint16((numVertices/2)):numVertices
            xNew = uint16(dim * rand());
            yNew = uint16((dim / 2) * rand() + (dim / 2));
            xBottom = [xBottom xNew];
            yBottom = [yBottom yNew];
        end
        xBottom = sort(xBottom, 'descend');

        % assemble after sorting to make closed polygons more likely
        pts = [xTop, xBottom; yTop, yBottom];
        pts = pts(:)';
    else
        if size(pts,1) == 0
        % Tape Container
            pts = [2, 10, ...
                5, 10, ...
                5, 12, ...
                6, 13, ...
                7, 12, ...
                7, 10, ...
                10, 8, ...
                12, 5, ...
                16, 5, ...
                22, 8, ...
                23, 10, ...
                23, 15,...
                20, 20, ...
                5, 20, ...
                2, 18];
            % RECTANGLE THING
            %pts = [5, 5, 20, 5, 10, 20, 5, 20];
            %pts = [10, 30, 80, 30, 50, 70, 10, 70];
            %pts = [10, 30, 80, 100, 10, 170, 100, 120, ...
             %  190, 170, 120, 100, 190, 30, 100, 80];
            % STAR
            %pts = [100, 10, 120, 80, 190, 100, 120, 120, ...
            %    100, 190, 80, 120, 10, 100, 80, 80];
        end
    end
elseif strcmp(shape, 'Circles') == 1
    disp('Generating random circle');
    % random endpoints
    minRad = 20;
    x = uint16((dim-minRad)*rand()+minRad);
    y = uint16((dim-minRad)*rand()+minRad);
    radius = min(dim - x - minRad, dim - y - minRad) * rand() + minRad;
    pts = [x, y, radius];
else
    disp('Error: Invalid shape string. Options are:');
    disp('\t Rectangles');
    disp('\t Lines');
    disp('\t Polygons');
    disp('\t Circles');
end

% create shape and display
shapeIm = uint8(step(S, I, pts));
nomIm = imresize(shapeIm, 8.0);
G = fspecial('gaussian', [20,20], 8.0);
nomIm = imfilter(nomIm, G);
nomIm = imsharpen(nomIm, 'Amount', 10);
imshow(shapeIm);

occMap = shapeIm;
occMap(shapeIm == 255) = 0;
occMap(shapeIm == 102) = 1;
tsdf = trunc_signed_distance(occMap, 2);
tsdf = imresize(tsdf, 1.0 / scale);
measuredTsdf = tsdf;

% create noise with user-specified parameters
dim = dim / scale; % now dim is that of new grid
noise = ones(dim, dim);
for i = 1:dim
    for j = 1:dim
        i_low = max(1,i-varParams.edgeWin);
        i_high = min(dim,i+varParams.edgeWin);
        j_low = max(1,j-varParams.edgeWin);
        j_high = min(dim,j+varParams.edgeWin);
        tsdfWin = tsdf(i_low:i_high, j_low:j_high);

        % add in transparency, occlusions
        if ((i > varParams.transp_y_thresh1_low && i <= varParams.transp_y_thresh1_high && ...
              j > varParams.transp_x_thresh1_low && j <= varParams.transp_x_thresh1_high) || ...
              (i > varParams.transp_y_thresh2_low && i <= varParams.transp_y_thresh2_high && ...
              j > varParams.transp_x_thresh2_low && j <= varParams.transp_x_thresh2_high) )
            % occluded regions
            if tsdf(i,j) < 0.6 % only add noise to ones that were actually in the shape
                measuredTsdf(i,j) = 0.5; % set outside shape
                noise(i,j) = varParams.transpScale;
            end
        
        elseif min(min(tsdfWin)) < 0.6 && ((i > varParams.y_thresh1_low && i <= varParams.y_thresh1_high && ...
                j > varParams.x_thresh1_low && j <= varParams.x_thresh1_high) || ...
                (i > varParams.y_thresh2_low && i <= varParams.y_thresh2_high && ... 
                j > varParams.x_thresh2_low && j <= varParams.x_thresh2_high) || ...
                (i > varParams.y_thresh3_low && i <= varParams.y_thresh3_high && ... 
                j > varParams.x_thresh3_low && j <= varParams.x_thresh3_high))
            
            noise(i,j) = varParams.occlusionScale;
        elseif ((i > varParams.occ_y_thresh1_low && i <= varParams.occ_y_thresh1_high && ...
                j > varParams.occ_x_thresh1_low && j <= varParams.occ_x_thresh1_high) || ... 
                (i > varParams.occ_y_thresh2_low && i <= varParams.occ_y_thresh2_high && ...
                j > varParams.occ_x_thresh2_low && j <= varParams.occ_x_thresh2_high) )
            % occluded regions
            noise(i,j) = varParams.occlusionScale;
        
        elseif tsdf(i,j) < -0.5 % only use a few interior points (since realistically we wouldn't measure them)
            if rand() > (1-varParams.interiorRate)
               noise(i,j) = varParams.noiseScale;
            else
               noise(i,j) = varParams.occlusionScale; 
            end
        else
            noiseVal = 1; % scaling for noise

            % add specularity to surface
            if varParams.specularNoise && min(min(abs(tsdfWin))) < 0.6
                noiseVal = rand();
                
                if rand() > (1-varParams.sparsityRate)
                    noiseVal = varParams.occlusionScale / varParams.noiseScale; % missing data not super noisy data
                    %noiseVal = noiseVal * varParams.sparseScaling;
                end
            end
            
            % scale the noise by the location in the image
            if strcmp(varParams.noiseGradMode, 'TLBR')
                noise(i,j) = noiseVal * ...
                    (varParams.noiseScale * varParams.horizScale * j + ...
                    varParams.noiseScale * varParams.vertScale * i);
            elseif strcmp(varParams.noiseGradMode, 'TRBL')
                noise(i,j) = noiseVal * ...
                    (varParams.noiseScale * varParams.horizScale * (dim-j+1) + ...
                    varParams.noiseScale * varParams.vertScale * i);
            elseif strcmp(varParams.noiseGradMode, 'BLTR')
                noise(i,j) = noiseVal * ...
                    (varParams.noiseScale * varParams.horizScale * j + ...
                    varParams.noiseScale * varParams.vertScale * (dim-i+1));
            elseif strcmp(varParams.noiseGradMode, 'BRTL')
                noise(i,j) = noiseVal * ...
                    (varParams.noiseScale * varParams.horizScale * (dim-j+1) + ...
                    varParams.noiseScale * varParams.vertScale * (dim-i+1));
            else
                noise(i,j) = noiseVal * varParams.noiseScale; 
            end
        end
    end
end

figure(4);
imagesc(noise);

% get gradients and points
[Gx, Gy] = imgradientxy(measuredTsdf, 'CentralDifference');
[X, Y] = meshgrid(1:dim, 1:dim);
[Xall, Yall] = meshgrid(1:dim, 1:dim);
fullTsdf = tsdf(:);
fullNormals = [Gx(:), Gy(:)];

% subsample both points and tsdf
validIndices = find(noise < varParams.occlusionScale);
X = X(validIndices);
Y = Y(validIndices);
measuredTsdf = measuredTsdf(validIndices);
Gx = Gx(validIndices);
Gy = Gy(validIndices);
noise = noise(validIndices);

% convert back to grid form
% X = reshape(X, dim, dim);
% Y = reshape(Y, dim, dim);
% tsdf = reshape(tsdf, dim, dim);
% Gx = reshape(Gx, dim, dim);
% Gy = reshape(Gy, dim, dim);
% noise = reshape(noise, dim, dim);

% store shape in struct
shapeParams = struct();
shapeParams.gridDim = dim;
shapeParams.vertices = pts;
shapeParams.tsdf = measuredTsdf(:);
shapeParams.noise = noise(:);
shapeParams.normals = [Gx(:), Gy(:)];
shapeParams.points = [X(:), Y(:)];
shapeParams.all_points = [Xall(:), Yall(:)];
shapeParams.fullTsdf = fullTsdf;
shapeParams.fullNormals = fullNormals;
shapeParams.nominalImage = nomIm;

if ~useCom
    shapeParams.com = mean(shapeParams.points(shapeParams.tsdf < 0,:));
else
    shapeParams.com = com;
end


    



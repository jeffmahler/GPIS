function [points, tsdf, J] = manual_tsdf(shape, dim ,fill)
% Displays a random item of the specified shape and allows the user to
% choose points on the surface, inside the surface, and outside the surface
% by clicking on images.
% Available shapes are:
%   'Rectangles'
%   'Lines'
%   'Polygons'
%   'Circles'

if nargin < 2
    dim = 100;
end
if nargin < 3
    fill = false;
end

I = 255*ones(dim, dim);
S = vision.ShapeInserter;
S.Shape = shape;
S.Fill = fill;
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
        fprintf('Generating random polygon with %d vertices', numVertices);

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
        pts = [10, 30, 80, 30, 50, 70, 10, 70]; 
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

disp('Displaying shape');
J = uint8(step(S, I, pts));
f = imshow(J);

disp('Click points on shape surface. Press ENTER when finished');
surfacePts = ginput;
numSurface = size(surfacePts, 1);
surfaceTsdf = zeros(numSurface, 1);

if strcmp(shape, 'Lines') == 1
    insidePts = [];
    insideTsdf = [];
else
    disp('Click points inside of shape. Press ENTER when finished');
    insidePts = ginput;
    numInside = size(insidePts, 1);
    insideTsdf = -1*ones(numInside, 1);
end

disp('Click points outside of shape. Press ENTER when finished');
outsidePts = ginput;
numOutside = size(outsidePts, 1);
outsideTsdf = 1*ones(numOutside, 1);

close all;

% pcombine points and tsdf valuess
points = [surfacePts; insidePts; outsidePts];
tsdf = [surfaceTsdf; insideTsdf; outsideTsdf];
    


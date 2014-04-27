function [points, tsdf, J] = auto_tsdf(shape, dim)
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
        %pts = [10, 30, 80, 30, 50, 70, 10, 70];
        %pts = [10, 30, 80, 100, 10, 170, 100, 120, ...
         %  190, 170, 120, 100, 190, 30, 100, 80];
        pts = [100, 10, 120, 80, 190, 100, 120, 120, ...
            100, 190, 80, 120, 10, 100, 80, 80];
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
imshow(J);

SE = strel('square', 3);
J_d = imdilate(J, SE);
%imshow(J_d);
J_e = imerode(J, SE);
%imshow(J_e);
J_2d = imdilate(J_d, SE);
%imshow(J_2d);

outsideMaskOrig = (J == 255);
insideMaskOrig = (J == 102);
outsideMaskEr = (J_e == 255);
insideMaskEr = (J_e == 102);
outsideMaskDi = (J_d == 255);
insideMaskDi = (J_d == 102);
outsideMaskDi2 = (J_2d == 255);
insideMaskDi2 = (J_2d == 102);

surfaceMask = outsideMaskDi & insideMaskOrig;
surfaceMaskOut = outsideMaskOrig & insideMaskEr;
surfaceMaskIn = outsideMaskDi2 & insideMaskDi;
insideMask = insideMaskDi2;
outsideMask = outsideMaskEr; 

% generate tsdf values
tsdf = zeros(dim, dim);
tsdf(surfaceMask) = 0;
tsdf(surfaceMaskIn) = -0.5;
tsdf(surfaceMaskOut) = 0.5;
tsdf(insideMask) = -1;
tsdf(outsideMask) = 1;

[X, Y] = meshgrid(1:dim, 1:dim);
points = [X(:), Y(:)];
tsdf = tsdf(:);
    



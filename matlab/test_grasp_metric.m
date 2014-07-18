function [Q] = test_grasp_metric(shape, dim ,fill)
% Displays a random item of the specified shape and allows the user to
% select grasp points. Returns Q the evaluated grasp quality at the points. 
% Available shapes are:
%   'Rectangles'
%   'Lines'
%   'Polygons'
%   'Circles'

fric_coef = 0.1; 
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

% disp('Click points that represent Contact Points. Press ENTER when finished');
% contactPts = ginput;
% 
% disp('Click the Center of Mass. Press ENTER when finished');
% center_of_mass = ginput;

%temp 
center_of_mass = [30; 30; 0]; 
h = double(height); 
w = double(width); 
contactPts = [30 30; 30+h/2 30-h/2; 0 0];

%Caculate Friction Cone 
cone_angle = atan(fric_coef);
index = 1; 
for i=1:size(contactPts,2)
    %Find the normalized direction of the normal force 
    f = contactPts(:,i) - center_of_mass;
%     [cd,i_max] = max(f);
%     [c,i_min] = min(f); 
%     f(i_max,1) = cd;
%     f(i_min,1) = 0; 
    f = f/norm(f,1)
    %Compute the extrema of the friction cone 
    y_d = norm(f,1)*tan(cone_angle);
    x_d = norm(f,1)/sin(cone_angle);
    
    f_r = [f(1,1) + x_d; f(2,1)+y_d; 0]; 
    f_l = [f(1,1) - x_d; f(2,1)+y_d; 0]; 
    
    %Normalize friction 
    f_r = f_r/norm(f_r,1); 
    f_l = f_l/norm(f_l,1); 
   
    forces(:,index) = f_r; 
    index = index+1;
    forces(:,index) = f_l; 
    index = index+1;
end
 size(forces)
 Q = ferrari_canny(center_of_mass,contactPts,forces); 
    
end
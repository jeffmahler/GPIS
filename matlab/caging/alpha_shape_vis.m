% generate set of random points and plot as union of balls, triangles
close all;
rng('default');
dim = 2;
min_radius = 95;
max_radius = 100;
marker_size = 100;
marker_color = 'k';

overlay_filename = '/Users/jeff/Documents/Research/implicit_surfaces/docs/caging/caging_figures/cage_collision_space.png';
overlay_im = imread(overlay_filename);

% random points, radii
points = points;%get_points_gui(overlay_im);%gallery('uniformdata', [num_points, dim], 2);
num_points = size(points, 1);
%points = rand(num_points, dim);
radii = (max_radius - min_radius) * rand(num_points,1) + min_radius;
areas = pi * radii.^2;

x_limits = [-0.5, 1.5];
y_limits = [-0.5, 1.5];

% raw points scattering
figure('Color',[1 1 1], 'Position',[100 100 900 600]);
scatter(points(:,1), points(:,2), marker_size, ...
    'MarkerEdgeColor', marker_color, 'MarkerFaceColor', marker_color);
axis off;
xlim(x_limits);
ylim(y_limits);

% plot bounding circles
figure('Color',[1 1 1], 'Position',[100 100 900 600]);
scatter(points(:,2), points(:,1), areas, ...
    'MarkerEdgeColor', marker_color, 'MarkerFaceColor', 'w'); % radii
hold on;
scatter(points(:,2), points(:,1), marker_size, ...
    'MarkerEdgeColor', marker_color, 'MarkerFaceColor', marker_color);
axis off;
xlim(x_limits);
ylim(y_limits);

%% power diagram plotter

% Set all other values to a high number
max_area = max(areas);
inv_radii = 1 ./ radii;
max_radius = max(radii) / 2;
im_dim = 2000;
alpha = 50 * im_dim * (120 / max_radius); % controls the power diagram radii
min_border = 0.1;
max_border = 0.8;
points_im = [round(im_dim * ((max_border - min_border) * points(:,2) + min_border)), ...
    round(im_dim * ((max_border - min_border) * points(:,1) + min_border))]; 
point_ind = sub2ind([im_dim, im_dim], points_im(:,1), points_im(:,2));
img = zeros(im_dim);
img(point_ind) = 50 * im_dim * (radii / max_radius);
img(img==0) = alpha;
[D, R] = DT(img);

% clean up power diagram
[counts, unique_elements] = hist(R(:), unique(R(:)));
denoised_ind = find(counts > 10);
denoised_vals = unique_elements(denoised_ind);
num_denoised = size(denoised_vals, 1);
power_diagram = zeros(size(R));
for i = 1:num_denoised
    power_diagram(R == denoised_vals(i)) = i;
end
[power_gmag, power_gdir] = imgradient(power_diagram);
power_diagram_boundary = zeros(size(power_diagram));
power_diagram_boundary(power_gmag > 0) = 1; 

% get boundary
rad = 11;
se = strel('disk', rad, 0);
% power_dilated = imdilate(power_diagram, se);
% power_diagram_boundary = (power_diagram == 0 & power_dilated == 255);
power_diagram_boundary = imdilate(power_diagram_boundary, se);
power_diagram_boundary = double(1 - power_diagram_boundary);

% win = 11;
% scale = 4;
% sig = 2;
% power_diagram_boundary = imresize(power_diagram_boundary, scale, 'nearest');
% G = fspecial('gaussian', [win, win], sig);
% power_diagram_boundary = imfilter(power_diagram_boundary, G, 'replicate');
% %power_diagram_boundary = histeq(power_diagram_boundary);
% power_diagram_boundary = imsharpen(power_diagram_boundary, 'Amount', 100);

%5 plot the results
h = figure(2);
clf;
% subplot(1,2,1);
% imagesc(D);
% title('Generalized Distance transform');
% axis image;
% subplot(1,2,2);
%imagesc(power_diagram);
imshow(power_diagram_boundary);
hold on;
scatter(points_im(:,2), points_im(:,1), marker_size, ...
    'MarkerEdgeColor', marker_color, 'MarkerFaceColor', marker_color);
%title('Power diagram');
axis image;
h.Color = 'none';

%% add edges, triangles
edges = [];
triangles = [1, 2, 4;
    1, 3, 4;
    2, 4, 8;
    3, 4, 5;
    4, 6, 7;
    4, 8, 7;
    4, 5, 6];
num_tris = size(triangles, 1);
tris = cell(num_tris, 1);
for i = 1:num_tris
    tris{i} = zeros(3,2);
    tris{i}(1,:) = points_im(triangles(i,1),:);
    tris{i}(2,:) = points_im(triangles(i,2),:);
    tris{i}(3,:) = points_im(triangles(i,3),:);
    edges = [edges;
        triangles(i,1), triangles(i,2);
        triangles(i,2), triangles(i,3);
        triangles(i,3), triangles(i,1)];
end

num_edges = size(edges, 1);
lines = cell(num_edges, 1);
for i = 1:num_edges
    lines{i} = zeros(2,2);
    lines{i}(1,:) = points_im(edges(i,1),:);
    lines{i}(2,:) = points_im(edges(i,2),:);
end

figure(3);
imshow(power_diagram_boundary);
hold on;
for i = 1:num_tris
    fill(tris{i}(:,2), tris{i}(:,1), 'r');
end

for i = 1:num_edges
    plot(lines{i}(:,2), lines{i}(:,1), 'b', 'LineWidth', 4);
end

scatter(points_im(:,2), points_im(:,1), marker_size, ...
    'MarkerEdgeColor', marker_color, 'MarkerFaceColor', marker_color);


%title('Power diagram');
axis image;

figure(4);
imshow(ones(im_dim));
hold on;
for i = 1:num_tris
    fill(tris{i}(:,2), tris{i}(:,1), 'r');
end

for i = 1:num_edges
    plot(lines{i}(:,2), lines{i}(:,1), 'b', 'LineWidth', 4);
end

scatter(points_im(:,2), points_im(:,1), marker_size, ...
    'MarkerEdgeColor', marker_color, 'MarkerFaceColor', marker_color);


%title('Power diagram');
axis image;


function tsdf_reg = standardize_tsdf( tsdf, vis, scale)
%STANDARDIZE_POSE Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
   vis = true; 
end
if nargin < 3
   scale = 25; 
end

[width, height] = size(tsdf);
[tsdf_surface, surf_points, inside_points, outside_points] = ...
    compute_tsdf_surface(tsdf);

centroid = mean(inside_points);
centroid = [centroid(2) centroid(1)];

centroid_diff = surf_points - repmat(centroid, size(surf_points,1), 1);
dist_to_center = sqrt(centroid_diff(:,1).^2 + centroid_diff(:,2).^2);
cur_scale = max(dist_to_center);

desired_scale = 7 * min(width, height) / 16;
scale_factor = desired_scale / cur_scale;

grid_center = ones(1,2);
grid_center(1) = width / 2;
grid_center(2) = height / 2;

U = pca(surf_points);
princ_dir1 = [U(2,1), U(1,1)];
princ_dir2 = [-U(1,1), U(2,1)];

if vis
    figure;
    subplot(1,2,1);
    imshow(tsdf_surface);
    hold on;
    arrow(centroid, centroid+scale*princ_dir1, 'FaceColor', 'r', ...
        'EdgeColor', 'r', 'Length', 10, 'Width', 5, 'TipAngle', 30);
end

registration = struct();
registration.R = [princ_dir2', -princ_dir1']';
registration.t = (grid_center' - registration.R * centroid');
registration.s_center = scale_factor;
registration.s_trans = 1.0;

tsdf_reg = warp_grid(registration, tsdf, centroid);

if vis
    subplot(1,2,2);
    imshow(tsdf_reg);
end

end


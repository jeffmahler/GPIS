function [ ] = plot_grasp_3d_arrows(grasp, sdf_centroid, sdf_res, config )

% get params
arrow_length = config.arrow_length_3d;

% setup grasp vars

[g1_obj, g2_obj] = grasp_points_grid_to_obj(grasp.g1, grasp.g2, ...
    sdf_centroid, sdf_res);
g_loa = [g1_obj'; g2_obj'];
%g_contacts = [grasp.g1; grasp.g2];
g1 = g1_obj';
g2 = g2_obj';
grasp_dir = g2 - g1;
grasp_dir = grasp_dir / norm(grasp_dir);

start1 = g1 - arrow_length * grasp_dir;
start2 = g2 + arrow_length * grasp_dir;

% scatter points, contacts, etc
hold on;
scatter3(g_loa(:,1), g_loa(:,2), g_loa(:,3), 5, 'rx');
%scatter3(g_contacts(:,1), g_contacts(:,2), g_contacts(:,3), 10, 'gx', 'LineWidth', 5);
%arrow3d(start1, g1, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 5, 'Width', 5, 'TipAngle', 45);
%arrow(start2, g2, 'FaceColor', 'c', 'EdgeColor', 'c', 'Length', 5, 'Width', 5, 'TipAngle', 45);
head_frac = 0.6;
rad1 = 0.01;
rad2 = 2 * rad1;

h = arrow3d([start1(1), g1(1)], [start1(2), g1(2)], [start1(3), g1(3)], head_frac, rad1, rad2);
set(h,'facecolor',[0 1 0]);
h = arrow3d([start2(1), g2(1)], [start2(2), g2(2)], [start2(3), g2(3)], head_frac, rad1, rad2);
set(h,'facecolor',[0 1 0]);

xlabel('X');
ylabel('Y');
zlabel('Z');

end


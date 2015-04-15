function [ ] = plot_grasp_3d(grasp, sdf, sdf_x, sdf_y, sdf_z, config )

% get params
sdf_dims = size(sdf);
arrow_length = config.arrow_length;
friction_coef = config.friction_coef;
n_cone_faces = config.n_cone_faces;

% setup grasp vars
g_loa = [grasp.g1_open; grasp.g2_open];
g_contacts = [grasp.g1; grasp.g2];
g1 = grasp.g1;
g2 = grasp.g2;
grasp_dir = g2 - g1;
grasp_dir = grasp_dir / norm(grasp_dir);

start1 = g1 - arrow_length * grasp_dir;
start2 = g2 + arrow_length * grasp_dir;

% scatter points, contacts, etc
scatter3(sdf_x, sdf_y, sdf_z);
hold on;
scatter3(g_loa(:,1), g_loa(:,2), g_loa(:,3), 5, 'rx');
scatter3(g_contacts(:,1), g_contacts(:,2), g_contacts(:,3), 10, 'gx', 'LineWidth', 5);
arrow(start1, g1, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 5, 'Width', 5, 'TipAngle', 45);
arrow(start2, g2, 'FaceColor', 'c', 'EdgeColor', 'c', 'Length', 5, 'Width', 5, 'TipAngle', 45);

[Gx, Gy, Gz] = gradient(sdf);
[forces, ~] = compute_forces(g_contacts, Gx, Gy, Gz, friction_coef, n_cone_faces);
num_forces = size(forces, 2);

for k = 1:num_forces
    f = forces{k};
    n_forces = size(f, 2);
    contact = g_contacts(k,:);

    for j = 1:n_forces
        f_pt = contact - 2*f(:,j)';
        scatter3(f_pt(:,1), f_pt(:,2), f_pt(:,3), 10, 'mx', 'LineWidth', 5);
    end
end

% scatter3(g1(1), g1(2), g1(3), 100, 'rx', 'LineWidth', 5);
% scatter3(g2(1), g2(2), g2(3), 100, 'rx', 'LineWidth', 5);
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([1, sdf_dims(1)]);
ylim([1, sdf_dims(2)]);
zlim([1, sdf_dims(3)]);

end


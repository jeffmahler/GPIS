% Code for running analyses on Amazon Picking Challenge items

%% read in sdf file
filename = 'data/apc/dove_beauty_bar/completed_tsdf_texture_mapped_mesh_clean_25.sdf';
sdf_file = textread(filename);
sdf_dims = sdf_file(1,:);
sdf_origin = sdf_file(2,:);
sdf_res = sdf_file(3,1);
sdf_vals = sdf_file(4:end,1);
sdf = reshape(sdf_vals, sdf_dims);

num_perturbations = 100;
num_random_grasps = 100;
sigma_trans = 0.5; % in grid cells
sigma_centroid = 0.5; % in grid cells
sigma_rot = 0.1;
arrow_length = 2;
step_size = 1;

plot_grasps = 1;
friction_coef = 0.5;
n_cone_faces = 2;
n_contacts = 2;
eps = 0;
theta_res = 0.25;
K = 10;
dist_thresh = 0.1;

pr2_grip_width_m = 0.15;
pr2_grip_width_grid = pr2_grip_width_m / sdf_res;
pr2_grip_offset = [-3.75, 0, 0]'; % offset from points at which pr2 contacts on closing...


%% display raw sdf points
% sdf_thresh = 0.001;
% sdf_zc = find(abs(sdf) < sdf_thresh);
% [sdf_x, sdf_y, sdf_z] = ind2sub(sdf_dims, sdf_zc);
[sdf_surf_mask, surf_points, inside_points] = compute_tsdf_surface(sdf);
sdf_x = surf_points(:,1);
sdf_y = surf_points(:,2);
sdf_z = surf_points(:,3);
n_surf = size(surf_points,1);
centroid = mean(surf_points);

figure(1);
scatter3(sdf_x, sdf_y, sdf_z);
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([1, sdf_dims(1)]);
ylim([1, sdf_dims(2)]);
zlim([1, sdf_dims(3)]);

% get sdf gradients
[Gx, Gy, Gz] = gradient(sdf);

%% generate pose samples to evaluate probability of force closure subject to pose samples
t_perturb = normrnd(0, sigma_trans, num_perturbations, 3);
c_perturb = normrnd(0, sigma_centroid, num_perturbations, 3);
rot_perturb = normrnd(0, sigma_rot, num_perturbations, 3);
R_sdf = cell(1, num_perturbations);
for i = 1:num_perturbations
    R_sdf{i} = angle2dcm(rot_perturb(i,1), rot_perturb(i,2), rot_perturb(i,3));
end

sdf_samples = cell(1, num_perturbations);
for i = 1:num_perturbations
    fprintf('Warping sdf %d\n', i); 
    tf = struct();
    tf.R = R_sdf{i};
    tf.t = t_perturb(i,:)';
    tf.s_center = 1;
    tf.s_trans = 1;
    
    sdf_samples{i} = warp_grid_3d(tf, sdf, centroid);
    
    [~, surf_points_samp, ~] = compute_tsdf_surface(sdf_samples{i});
    sdf_x_samp = surf_points_samp(:,1);
    sdf_y_samp  = surf_points_samp(:,2);
    sdf_z_samp  = surf_points_samp(:,3);

    figure(2);
    scatter3(sdf_x_samp , sdf_y_samp , sdf_z_samp);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    xlim([1, sdf_dims(1)]);
    ylim([1, sdf_dims(2)]);
    zlim([1, sdf_dims(3)]);
end

%% get 'antipodal' grasp
rng(100);
grasps = cell(1, num_random_grasps);
for i = 1:num_random_grasps
    fprintf('Evaluating grasp %d\n', i);

    % new random grasps   
    grasp_success = false;
    while ~grasp_success
        [contacts_mean, grasp_success] = get_random_antipodal_grasp_3d(sdf, pr2_grip_width_grid);
    end
    % contacts
    g1 = contacts_mean(1,:);
    g2 = contacts_mean(2,:);
    grasp_center = (g1 + g2) / 2;
    grasp_diff = g2 - g1;
    grasp_dir = grasp_diff / norm(grasp_diff);

    % set max width
    g1_open = grasp_center - (pr2_grip_width_grid / 2) * grasp_dir;
    g2_open = grasp_center + (pr2_grip_width_grid / 2) * grasp_dir;

    start1 = g1 - arrow_length * grasp_dir;
    start2 = g2 + arrow_length * grasp_dir;

    g1_gp = [g1_open; g2_open];
    g1_loa = compute_loa(g1_gp, step_size);

    g2_gp = [g2_open; g1_open];
    g2_loa = compute_loa(g2_gp, step_size);

    loas = {g1_loa, g2_loa};
    num_fc = 0;
    for j = 1:num_perturbations
        if mod(j, 10) == 0
            fprintf('Evaluating pose %d\n', j);
        end

        % get gradients, compute contacts
        [contacts, success] = find_contacts(loas, sdf_samples{j});
        [Gx, Gy, Gz] = gradient(sdf_samples{j});

        if success
            % get grasp quality
            [forces, failed] = compute_forces(contacts, Gx, Gy, Gz, friction_coef, n_cone_faces);
            if ~failed
                Q = ferrari_canny_3d(centroid', contacts', forces);
                if Q > eps
                    num_fc = num_fc + 1;
                end
            end
        end
    end

    pfc = num_fc / num_perturbations;
    fprintf('PFC = %f\n', pfc);

    %% plot grasp on mean sdf
    if plot_grasps
        figure(1);
        clf;
        scatter3(sdf_x, sdf_y, sdf_z);
        hold on;
        scatter3(g1_loa(:,1), g1_loa(:,2), g1_loa(:,3), 5, 'rx');
        scatter3(contacts_mean(:,1), contacts_mean(:,2), contacts_mean(:,3), 10, 'gx', 'LineWidth', 5);
        arrow(start1, g1, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 5, 'Width', 5, 'TipAngle', 45);
        arrow(start2, g2, 'FaceColor', 'c', 'EdgeColor', 'c', 'Length', 5, 'Width', 5, 'TipAngle', 45);

        [Gx, Gy, Gz] = gradient(sdf);
        [forces, failed] = compute_forces(contacts_mean, Gx, Gy, Gz, friction_coef, n_cone_faces);
        Q = ferrari_canny_3d(centroid', contacts', forces);

        for k = 1:n_contacts
            f = forces{k};
            n_forces = size(f, 2);
            contact = contacts_mean(k,:);

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

    %% transform grasp to object reference frame
    vol_origin_grid = [1, 1, 1]';
    vol_origin_m = grid_to_m(vol_origin_grid, sdf_res);

    % centroid in volume frame of ref
    centroid_m_vol = grid_to_m(centroid', sdf_res);
    centroid_m_obj = [0,0,0]';

    R_vol_obj = eye(3);
    t_vol_obj = -centroid_m_vol;

    g1_vol = grid_to_m(g1', sdf_res);
    g2_vol = grid_to_m(g2', sdf_res);
    g1_obj = R_vol_obj * g1_vol + t_vol_obj;
    g2_obj = R_vol_obj * g2_vol + t_vol_obj;

    g_center_obj = (g1_obj + g2_obj) / 2;
    g_center_obj = g_center_obj + pr2_grip_offset;

    %% get rotational axes of grasp
    g_y_axis = g1_obj - g2_obj;
    g_y_axis = g_y_axis / norm(g_y_axis);
    g_x_ref = [g_y_axis(2) -g_y_axis(1), 0]'; % dir orth to y axis
    g_x_ref = g_x_ref / norm(g_x_ref);
    g_z_ref = cross(g_x_ref, g_y_axis);
    R_canon_g = [g_z_ref, g_x_ref, g_y_axis];

    theta_vals = 0:theta_res:(2*pi - theta_res);
    num_theta = size(theta_vals, 2);
    R_grasp_obj_list = cell(1, num_theta);

    for k = 1:num_theta
        theta = theta_vals(k);
        g_x_axis = [cos(theta), sin(theta), 0]';
        g_x_axis = R_canon_g * g_x_axis;
        g_x_axis = g_x_axis / norm(g_x_axis);
        g_z_axis = cross(g_x_axis, g_y_axis);

        R_obj_grasp = [g_x_axis, g_y_axis, g_z_axis];
        R_grasp_obj = R_obj_grasp';
        R_grasp_obj_list{k} = R_grasp_obj;
    end

    grasp = struct();
    grasp.t_grasp_obj = g_center_obj;
    grasp.R_grasp_obj_list = R_grasp_obj_list;
    grasp.contacts_mean = contacts_mean;
    grasp.g1_open_grid = g1_open;
    grasp.g2_open_grid = g2_open;
    grasp.dir = grasp_dir;
    grasp.pfc = pfc;
    grasps{i} = grasp;
end

%% find K grasps with highest pfc
best_grasps = cell(1, K);

for k = 1:K
    best_pfc = -1;
    best_ind = 0;
    for i = 1:num_random_grasps
        grasp = grasps{i};
        
        if k > 1
            dists = zeros(1, k-1);
            for j = 1:k-1
                dists(j) = grasp_dist(best_grasps{j}, grasp);
            end
        else
            dists = 100;
        end
        
        if grasp.pfc > best_pfc && min(dists) > dist_thresh
            best_pfc = grasp.pfc;
            best_ind = i;
        end
    end
    best_grasps{k} = grasps{best_ind};
end

%% display the octree and whatnot
% pts = [sdf_x, sdf_y, sdf_z];
% oct = OcTree(pts, 'binCapacity', 10);
% figure
% boxH = oct.plot; 
% cols = lines(oct.BinCount); 
% doplot3 = @(p,varargin) plot3(p(:,1), p(:,2), p(:,3), varargin{:}); 
% for i = 1:oct.BinCount 
%    set(boxH(i),'Color',cols(i,:),'LineWidth', 1+oct.BinDepths(i)) 
%    doplot3(pts(oct.PointBins==i,:),'.','Color',cols(i,:)) 
% end 
% axis image, view(3)
% 
% bin_centers = octree_bin_centers(oct);


%% get a random grasp
% sigma_c = 5.0;
% eps = 10.0;
% arrow_length = 5;
% step_size = 0.1;
% 
% % sample random spherical coords
% theta = 2 * pi * rand();
% phi = pi * rand();
% r = max(max(surf_points - repmat(centroid, n_surf, 1))) + eps;
% 
% % sample grasp center
% center = normrnd(centroid, sigma_c);
% 
% % convert spherical to cartesian
% [g1_x, g1_y, g1_z] = sph2cart(theta, phi, r);
% g1 = [g1_x g1_y g1_z] + center;
% [g2_x, g2_y, g2_z] = sph2cart(theta, phi, -r);
% g2 = [g2_x g2_y g2_z] + center;
% 
% grasp_diff = g2 - g1;
% grasp_dir = grasp_diff / norm(grasp_diff);
% start1 = g1 - arrow_length * grasp_dir;
% start2 = g2 + arrow_length * grasp_dir;
% 
% g1_gp = [g1; g2];
% g1_loa = compute_loa(g1_gp, step_size);
% 
% g2_gp = [g2; g1];
% g2_loa = compute_loa(g2_gp, step_size);
% 
% loas = {g1_loa, g2_loa};
% contacts = find_contacts(loas, sdf);

%  sobel filts
% sobel_3d = zeros(3,3,3);
% sobel_2d = [-1 -2 -1;
%              0  0  0;
%              1  2  1];
% sobel_3d(:,:,1) = sobel_2d;
% sobel_3d(:,:,2) = 2*sobel_2d;
% sobel_3d(:,:,3) = sobel_2d;


    % plot
    % [~, surf_points_samp, ~] = compute_tsdf_surface(sdf_samples{j});
    % sdf_x_samp = surf_points_samp(:,1);
    % sdf_y_samp  = surf_points_samp(:,2);
    % sdf_z_samp  = surf_points_samp(:,3);
    % 
    % figure(1);
    % clf;
    % scatter3(sdf_x_samp, sdf_y_samp, sdf_z_samp);
    % hold on;
    % scatter3(g1_loa(:,1), g1_loa(:,2), g1_loa(:,3), 5, 'rx');
    % scatter3(contacts(:,1), contacts(:,2), contacts(:,3), 10, 'gx', 'LineWidth', 5);
    % arrow(start1, g1, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 5, 'Width', 5, 'TipAngle', 45);
    % arrow(start2, g2, 'FaceColor', 'c', 'EdgeColor', 'c', 'Length', 5, 'Width', 5, 'TipAngle', 45);
    % 
    % for k = 1:n_contacts
    %     f = forces{k};
    %     n_forces = size(f, 2);
    %     contact = contacts(k,:);
    %    
    %     for m = 1:n_forces
    %         f_pt = contact - 2*f(:,m)';
    %         scatter3(f_pt(:,1), f_pt(:,2), f_pt(:,3), 10, 'mx', 'LineWidth', 5);
    %     end
    % end
    % 
    % % scatter3(g1(1), g1(2), g1(3), 100, 'rx', 'LineWidth', 5);
    % % scatter3(g2(1), g2(2), g2(3), 100, 'rx', 'LineWidth', 5);
    % xlabel('X');
    % ylabel('Y');
    % zlabel('Z');
    % xlim([1, sdf_dims(1)]);
    % ylim([1, sdf_dims(2)]);
    % zlim([1, sdf_dims(3)]);
    % pause(5);


%% read in sdf file
object_name = 'dove_beauty_bar';
filename = sprintf('data/apc/%s/completed_tsdf_texture_mapped_mesh_clean_25.sdf', object_name);
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
out_dir = 'results/apc';

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
pr2_grip_offset = [-0.0375, 0, 0]'; % offset from points at which pr2 contacts on closing...


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
clf;
scatter3(sdf_x, sdf_y, sdf_z);
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([1, sdf_dims(1)]);
ylim([1, sdf_dims(2)]);
zlim([1, sdf_dims(3)]);

% get sdf gradients
[Gx, Gy, Gz] = gradient(sdf);

%% run grasps
config = struct();

config.num_samples = 2;
config.grasp_width = pr2_grip_width_grid;
config.friction_coef = friction_coef;
config.n_cone_faces = n_cone_faces;
config.dir_prior = 1.0 / 2.0;
config.alpha_thresh = pi / 32;
config.rho_thresh = 0.9 * norm(max(surf_points,[],1) - min(surf_points,[],1));

config.theta_res = 2 * pi / 10;
config.grid_res = sdf_res;
config.grasp_offset = pr2_grip_offset;
config.constrain_2d = true;

config.arrow_length = 3;
config.vis = false;
config.scale = 10;
config.plate_width = 2;

rng(100);
grasps = get_antipodal_grasp_candidates(sdf, config);

%% Thompson Sampling 
num_grasps = size(grasps,2); 
set_size = num_grasps/10; 
i_p = 1; 
top_grasps = {}; 
for i = 1:set_size:num_grasps
    grasp_set = grasps(i_p:i); 
    i_p = i; 
    best_grasp = thompson_apc(grasp_set,poses,config); 
    top_grasp{idx} = best_grasp; 
end


%% plot 2d
num_grasps = size(grasps, 2);
figure(1);
for j = 1:num_grasps
    clf;
    if grasps{j}.slice == 3
        plot_grasp_2d(grasps{j}, sdf, config);
        pause(0.1);
    end
end

%% plot 3d
num_grasps = size(grasps, 2);
figure(1);
for j = 1:num_grasps
    clf;
    plot_grasp_3d(grasps{j}, sdf, sdf_x, sdf_y, sdf_z, config);
    pause(0.01);
end

%% convert grasps to json
all_grasps_json = [];
figure(1);
for j = 100:10:200
    grasp_json = grasp_to_json(grasps{j});
    clf;
    plot_grasp_3d(grasps{j}, sdf, sdf_x, sdf_y, sdf_z, config);
    all_grasps_json = [all_grasps_json, grasp_json];
    pause(1);
end

out_filename = sprintf('%s/%s.json', out_dir, object_name);
savejson([], all_grasps_json, out_filename);

    
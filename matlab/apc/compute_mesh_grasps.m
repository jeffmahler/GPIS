function [results] = compute_mesh_grasps(object_name, config)
%COMPUTE_MESH_GRASPS Compute grasps for mesh

% filenames for sdf, obj
sdf_filename = sprintf('%s/%s/%s', config.root_dir, object_name, config.sdf_filename);
obj_filename = sprintf('%s/%s/%s', config.root_dir, object_name, config.obj_filename);

% read sdf
sdf_file = textread(sdf_filename);
sdf_dims = sdf_file(1,:);
sdf_origin = sdf_file(2,:);
sdf_res = sdf_file(3,1);
sdf_vals = sdf_file(4:end,1);
sdf = reshape(sdf_vals, sdf_dims);

sigma_trans_old = config.sigma_trans;
config.sigma_trans = (1.0 / sdf_res) * config.sigma_trans;

pr2_grip_width_grid = config.pr2_grasp_width_m / sdf_res;
config.grasp_width = pr2_grip_width_grid;

% get sdf surface points
[~, surf_points, ~] = ...
    compute_tsdf_surface_thresholding(sdf, config.surf_thresh);
sdf_x = surf_points(:,1);
sdf_y = surf_points(:,2);
sdf_z = surf_points(:,3);
centroid = mean(surf_points);

if config.vis_sdf
    figure(1);
    clf;
    scatter3(sdf_x, sdf_y, sdf_z);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    xlim([1, sdf_dims(1)]);
    ylim([1, sdf_dims(2)]);
    zlim([1, sdf_dims(3)]);
end

% read mesh
obj = read_wobj(obj_filename);
mesh.vertices = obj.vertices;
mesh.faces = obj.objects(3).data.vertices;

% get grasp candidates
config.grid_res = sdf_res;
config.rho_thresh = config.rho_scale * norm(max(surf_points,[],1) - min(surf_points,[],1));
grasps = get_antipodal_grasp_candidates(sdf, config);
num_grasps = size(grasps, 2);

% exit if no grasps found
if num_grasps == 0
    results = struct();
    results.object_name = object_name;
    results.mc = struct();
    results.thompson = struct();
    return;
end

% plot 2d
if config.plot_all_grasps_2d
    figure(1);
    for j = 1:num_grasps
        clf;
        if grasps{j}.slice == 3
            plot_grasp_2d(grasps{j}, sdf, config);
            pause(0.1);
        end
    end
end

% plot 3d
if config.plot_all_grasps_3d
    figure(1);
    for j = 1:num_grasps
        clf;
        plot_grasp_3d(grasps{j}, sdf, sdf_x, sdf_y, sdf_z, config);
        pause(0.01);
    end
end

% sample poses for perturbation analysis
fprintf('Sampling pose\n');
sdf_samples = pose_sample_apc(config.num_perturbations, sdf, ...
    centroid, config.sigma_trans, config.sigma_rot, ...
    config.pose_sampling_2d, config.vis_pose_perturb);

% bin grasps
fprintf('Partitioning grasps\n');
old_num_bins = config.num_bins;
if num_grasps < config.num_bins
    config.num_bins = num_grasps-1;
end
if num_grasps  > 5 % need more bins than vectors
    if config.use_uniform_space_part
        grasp_bins = uniformly_partition_grasps(grasps, config.num_bins);
    else
        grasp_bins = space_partition_grasps(grasps, config.num_bins);
    end
end

grasp_eval_fn = @(x, y, z) grasp_quality_apc(x, y, z, config.step_size);

% run bandits over grasp bins
best_grasps_thomp = cell(1, config.num_bins);
best_qualities_thomp = zeros(1, config.num_bins);
all_values_thomp = [];
for i = 1:config.num_bins
    fprintf('Running bandits on bin %d\n', i);
    [grasps_thomp, qualities_thomp, values_thomp] = ...
        thompson_apc(grasp_bins{i}, sdf_samples, grasp_eval_fn, config);
    best_grasps_thomp{i} = grasps_thomp{1};
    best_qualities_thomp(i) = qualities_thomp(1);
    all_values_thomp = [all_values_thomp; values_thomp];
end

% for comparison, run monte-carlo
fprintf('Runnning monte carlo\n');
old_candidates = config.num_candidate_grasps;
config.num_candidate_grasps = config.num_bins;
[best_grasps_mc, best_qualities_mc, all_values_mc] = ...
    monte_carlo_apc(grasps, sdf_samples, grasp_eval_fn, config);
config.num_candidate_grasps = old_candidates;

%  plot grasps on 3d mesh
if config.vis_best_grasps
    min_v = min(mesh.vertices, [], 1);
    max_v = max(mesh.vertices, [], 1);
    center_v = (max_v + min_v) / 2;
    max_extent = max(max_v - min_v);
    limits = [center_v - max_extent / 2;
              center_v + max_extent / 2];

    figure;%('Color',[1 1 1], 'Position',[100 100 900 600]);
    for j = 1:config.num_bins
        clf;
        patch(mesh, 'facecolor',[1 0 0]); % red color
        hold off;
        camlight;
        %axis off;

        plot_grasp_3d_arrows(best_grasps_thomp{j}, centroid, sdf_res, config );

        xlim(limits(:,1));
        ylim(limits(:,2));
        zlim(limits(:,3));
        view(-45, 30);

        pause(1);
    end
end

% convert best mc grasps to json and save
fprintf('Saving mc json\n');
grasps_json_mc = [];
for j = 1:config.num_bins
    grasp_json = grasp_to_json(best_grasps_mc{j}, best_qualities_mc(j));
    grasps_json_mc = [grasps_json_mc, grasp_json];
end
out_filename_mc = sprintf('%s/%s_mc.json', config.out_dir, object_name);
savejson([], grasps_json_mc, out_filename_mc);

% convert best bandit grasps to json and save
fprintf('Saving bandits json\n');
grasps_json_thomp = [];
for j = 1:config.num_bins
    grasp_json = grasp_to_json(best_grasps_thomp{j}, best_qualities_thomp(j));
    grasps_json_thomp = [grasps_json_thomp, grasp_json];
end
out_filename_thomp = sprintf('%s/%s.json', config.out_dir, object_name);
savejson([], grasps_json_thomp, out_filename_thomp);

% return grasps, qualities, and values in structure
results = struct();
results.object_name = object_name;

results.thompson = struct();
results.thompson.grasps = best_grasps_thomp;
results.thompson.qualities = best_qualities_thomp;
results.thompson.values= all_values_thomp;

results.mc = struct();
results.mc.grasps = best_grasps_mc;
results.mc.qualities = best_qualities_mc;
results.mc.values= all_values_mc;

% reset num bins
config.num_bins = old_num_bins;
config.sigma_trans = sigma_trans_old;

end


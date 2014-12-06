function comparison_results = compare_bandits_for_grasp_transfer(shape_indices, config)
% Compares the performance of various bandit methods for grasp transfer
% with shapes specified as rows in a data matrix.
% The config file contains relevant info for running the experiment

% read config to local variables
save_dir = config.mat_save_dir;
grip_scale = config.grip_scale;
plate_width = config.plate_width;
tsdf_thresh = config.tsdf_thresh;
padding = config.padding;
vis_std = config.vis_std;

model_dir = config.model_dir;
downsample = config.downsample;

noise_params = config.noise_params;
construction_params = config.construction_params;

% load in data matrix
caltech_data = load(config.test_data_file);
num_points = size(caltech_data.X, 2);
query_dim = sqrt(num_points);
num_shapes = size(shape_indices, 1);

comparison_results = cell(1,num_shapes);

% load models
filenames_filename = sprintf('%s/filenames.mat', model_dir);
tsdf_filename = sprintf('%s/tsdf_vectors_%d.mat', model_dir, downsample);
kd_tree_filename = sprintf('%s/kd_tree_%d.mat', model_dir, downsample);
grasps_filename = sprintf('%s/grasps_%d.mat', model_dir, downsample);
grasp_q_filename = sprintf('%s/grasp_q_%d.mat', model_dir, downsample);

S = load(filenames_filename);
filenames = S.filenames;

S = load(tsdf_filename);
tsdf_vectors = S.X;

S = load(kd_tree_filename);
kd_tree = S.kd_tree;

S = load(grasps_filename);
grasps = S.grasps;

S = load(grasp_q_filename);
grasp_q = S.grasp_qualities;

num_training = size(tsdf_vectors, 1);
data_dim = sqrt(size(tsdf_vectors, 2));

% gamma = 0.98
compute_gitttin_indices(config.gittins_in_filename1, ...
    config.gittins_out_filename1);

% gamma = 0.9
compute_gitttin_indices(config.gittins_in_filename2, ...
    config.gittins_out_filename2);

% create bayes ucb table
num_alpha_samp = 250;
num_beta_samp = 250;
num_samples = 1000;
samples_table = zeros(num_alpha_samp, num_beta_samp, num_samples);
for i = 1:num_alpha_samp
    for j = 1:num_beta_samp
        samples_table(i,j,:) = betarnd(i,j,1,num_samples);
    end
end

for i = 1:num_shapes
    shape_index = shape_indices(i);
    class_name = caltech_data.classnames{caltech_data.Y(shape_index)};
    fprintf('Shape type: %s\n', class_name);

    %% construct gpis
    filt_win = config.filter_win;
    filt_sigma = config.filter_sigma;
    vis_std = config.vis_std;
    min_dim = config.quad_min_dim;
    max_dim = config.quad_max_dim;

    tsdf_smoothing_filter = fspecial('gaussian', filt_win, filt_sigma);
    outside_mask = caltech_data.X(shape_index,:);
    occupancy_grid = 1 - reshape(outside_mask, [query_dim, query_dim]);

    % create smooth tsdf
    tsdf = tsdf_from_occupancy_grid(occupancy_grid, padding, ...
        tsdf_thresh, tsdf_smoothing_filter, data_dim, vis_std);

    % convert to shape params using a quadtree decomposition
    shape_params = shape_params_from_tsdf_quadtree(tsdf, noise_params, ...
        min_dim, max_dim);

    % construct gpis
    num_samples = config.num_shape_samples;
    image_scale = config.image_scale;
    shape_name = sprintf('%s_%d', class_name, shape_index);
    [gp_model, shape_samples, construction_results] = ...
                construct_and_save_gpis(shape_name, save_dir, shape_params, ...
                                        construction_params, num_samples, image_scale);


    %% lookup nearest neighbors
    K = config.knn;
    idx = knnsearch(kd_tree, tsdf(:)', 'K', K);

    if config.vis_knn
        figure(6);
        imshow(tsdf);
        title('Original TSDF');

        figure(7);
    end

    grasps_per_shape = min(size(grasps, 2), config.num_knn_grasps);
    grasp_size = size(grasps, 3);
    num_grasp_candidates = K * grasps_per_shape;

    grasps_neighbor = zeros(grasps_per_shape, grasp_size);
    grasp_candidates = zeros(num_grasp_candidates, 4);

    start_I = 1;
    end_I = start_I + grasps_per_shape - 1;

    for j = 1:K
        % load neighbor attributes
        tsdf_neighbor = tsdf_vectors(idx(j),:);
        tsdf_neighbor = reshape(tsdf_neighbor, [data_dim, data_dim]);

        % neighbor shape params
        neighbor_shape_params = tsdf_to_shape_params(tsdf_neighbor);
        neighbor_shape_params.surfaceThresh = 0.1;

        % transfer neighbor grasp
        grasp_q_neighbor = grasp_q(idx(j), :);
        grasps_neighbor(:,:) = grasps(idx(j), 1:grasps_per_shape, :);
        grasp_candidates(start_I:end_I, :) = grasps_neighbor;

        start_I = start_I + grasps_per_shape;
        end_I = start_I + grasps_per_shape - 1;

        % visualization
        if config.vis_knn
            subplot(sqrt(K),sqrt(K),j);
            visualize_grasp(grasps_neighbor(1,:)', neighbor_shape_params, tsdf_neighbor, ...
                 config.scale, config.arrow_length, config.plate_width, data_dim);
            title(sprintf('Neighbor %d', j));
        end
    end

    % use bandits to select the best grasp
    num_grasp_samples = config.num_grasp_samples;
    grasp_samples = collect_samples_grasps(gp_model, grasp_candidates, ...
        num_grasp_samples, config, shape_params);

    transfer_results = struct();
    
    transfer_results.gp_model = gp_model;
    transfer_results.shape_samples = shape_samples;
    transfer_results.construction_results = construction_results;
    
    transfer_results.grasp_candidates = grasp_candidates;
    [~, transfer_results.grasp_values] = ...
        monte_carlo(grasp_samples, num_grasp_candidates, shape_params, config, tsdf);
 
    % random
    random_results = struct();
    [random_results.best_grasp, random_results.regret, random_results.values] = ...
        random(grasp_samples, num_grasp_candidates, ...
               shape_params, config, tsdf, config.vis_bandits);
    transfer_results.random = random_results;
    
    % successive rejects
    successive_rejects_results = struct();
    [successive_rejects_results.best_grasp, successive_rejects_results.regret, successive_rejects_results.values] = ...
        succesive_rejects(grasp_samples, num_grasp_candidates, ...
                           shape_params, config, tsdf, config.vis_bandits);
    transfer_results.successive_rejects = successive_rejects_results;
    
    % hoeffding races
    hoeffding_races_results = struct();
    [hoeffding_races_results.best_grasp, hoeffding_races_results.regret, hoeffding_races_results.values] = ...
        hoeffding_races(grasp_samples, num_grasp_candidates, ...
                        shape_params, config, tsdf, config.vis_bandits);
    transfer_results.hoeffding_races = hoeffding_races_results;
    
    % ucb
    ucb_results = struct();
    [ucb_results.best_grasp, ucb_results.regret, ucb_results.values] = ...
        ucb(grasp_samples, num_grasp_candidates, ...
            shape_params, config, tsdf, config.vis_bandits);
    transfer_results.ucb = ucb_results;

    % bayes ucb
    bayes_ucb_results = struct();
    [bayes_ucb_results.best_grasp, bayes_ucb_results.regret, bayes_ucb_results.values] = ...
       bayes_ucb(grasp_samples, num_grasp_candidates, ...
                shape_params, config, tsdf, samples_table, config.vis_bandits);
    transfer_results.bayes_ucbs = bayes_ucb_results;
    
    % thompson sampling
    thompson_results = struct();
    [thompson_results.best_grasp, thompson_results.regret, thompson_results.values] = ...
        thompson_sampling(grasp_samples, num_grasp_candidates, ...
            shape_params, config, tsdf, config.vis_bandits);
    transfer_results.thompson = thompson_results;
    
    % gittins index policy gamma = 0.90
    gittins90_results = struct();
    [gittins90_results.best_grasp, gittins90_results.regret, gittins90_results.values] = ...
        gittins_index(grasp_samples, num_grasp_candidates, ...
            shape_params, config, tsdf, config.vis_bandits, config.gittins_out_filename2);
    transfer_results.gittins90 = gittins90_results;
    
    % gittins index policy gamma = 0.98
    gittins98_results = struct();
    [gittins98_results.best_grasp, gittins98_results.regret, gittins98_results.values] = ...
        gittins_index(grasp_samples, num_grasp_candidates, ...
            shape_params, config, tsdf, config.vis_bandits, config.gittins_out_filename1);
    transfer_results.gittins98 = gittins98_results;

    comparison_results{i} = transfer_results;

    if mod(i, config.snapshot_iter) == 0
        save('results/bandits/comparison_results_snapshot.mat', 'comparison_results');
    end
end

end


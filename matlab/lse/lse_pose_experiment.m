function experiment_results = lse_pose_experiment(experiment_config, lse_config)
%LSE_EXPERIMENT Cycle through combinations of parameters and test lse

height = lse_config.height;
width = lse_config.width;
num_points = height * width;
grid_center = [height / 2, width / 2];

% setup function mean and variance
f_mu = zeros(num_points, 1);
f_Sigma = feval(lse_config.cov_func, ...
    [log(lse_config.sigma_kernel), log(lse_config.kernel_scale)], ...
    lse_config.points);
f_Sigma = 1e-10 + (f_Sigma + f_Sigma') / 2; 

f_resample_Sigma = feval(lse_config.cov_func, ...
    [log(lse_config.sigma_kernel), log(lse_config.resample_kernel_scale)], ...
    lse_config.points);
f_resample_Sigma = 1e-10 + (f_resample_Sigma + f_resample_Sigma') / 2; 

% setup horizons
num_horizons = size(experiment_config.horizons, 2);
num_path_penalties = size(experiment_config.path_penalties, 2);
num_beams = size(experiment_config.beam_sizes, 2);
num_dpp = size(experiment_config.dec_pps, 2);
output_dir = experiment_config.output_dir;
num_function_samples = experiment_config.num_function_samples;

num_methods = num_horizons * num_path_penalties * num_beams * num_dpp + 2;
yaw = 0;  
roll = 0; 

experiment_results = cell(num_methods, num_function_samples);

for k = 1:num_function_samples
    % sample a zero-mean function from the gp
    f = mvnrnd(f_mu, f_Sigma);
    f_grid = reshape(f, [height, width]);
    
    % warp function by random tf
    pitch = (lse_config.max_rot - lse_config.min_rot) * ...
        2 * (rand() - 0.5) + lse_config.min_rot;
    dcm = angle2dcm(pitch, roll, yaw);
    R = dcm(1:2, 1:2);

    tf = struct();
    tf.t = (lse_config.max_trans - lse_config.min_trans) .* ...
        [2 * (rand() - 0.5), 2 * (rand() - 0.5)] + lse_config.min_trans;
    tf.t = tf.t';
    tf.R = R;
    tf.s = (lse_config.max_scale - lse_config.min_scale) * ...
        2 * (rand() - 0.5) + lse_config.min_scale;

    mu_grid = warp_mean_function(tf, f_grid, grid_center);
    lse_config.true_tf = tf;

    % sample a function from the warped mean grid (with different noise)
    f_mu_warped = mu_grid(:);
    f = mvnrnd(f_mu_warped, f_resample_Sigma);
    f_grid = reshape(f, [height, width]);
    
    start_ind = randsample(num_points, 1);
    lse_config.mean_func = [];
    
    % random results
    if experiment_config.log_results
        fprintf('Evaluating random on function %d\n', k);
    end
    trial_results = struct();
    [class_results, path_lengths, losses, class_images] = ...
        discrete_gp_ls_random(f_grid, lse_config, start_ind);
    trial_results.class_results = class_results;
    trial_results.path_lengths = path_lengths;
    trial_results.losses = losses;
    trial_results.class_images = class_images;
    trial_results.f = f;
    experiment_results{1, k} = trial_results;
                    
    % subsampling results
    if experiment_config.log_results
        fprintf('Evaluating subsample on function %d\n', k);
    end
    trial_results = struct();
    [class_results, path_lengths, losses, class_images] = ...
        discrete_gp_ls_subsample(f_grid, lse_config, start_ind);
    trial_results.class_results = class_results;
    trial_results.path_lengths = path_lengths;
    trial_results.losses = losses;
    trial_results.class_images = class_images;
    trial_results.f = f;
    experiment_results{2, k} = trial_results;
    
    % discrete gp lse
    trial_results = struct();
    [class_results, path_lengths, losses, class_images] = ...
        discrete_gp_lse(f_grid, lse_config, start_ind);
    trial_results.class_results = class_results;
    trial_results.path_lengths = path_lengths;
    trial_results.losses = losses;
    trial_results.class_images = class_images;
    trial_results.f = f;
    experiment_results{3, k} = trial_results;
    
    % discrete gp lse
    lse_config.mean_func = mu_grid;
    trial_results = struct();
    [class_results, path_lengths, losses, class_images, iter_times] = ...
        discrete_gp_lse_pose_prior(f_grid, lse_config, start_ind);
    trial_results.class_results = class_results;
    trial_results.path_lengths = path_lengths;
    trial_results.losses = losses;
    trial_results.class_images = class_images;
    trial_results.iter_times = iter_times;
    trial_results.f = f;
    experiment_results{4, k} = trial_results;
    
    save(sprintf('%s/temp_lse_pose_results.mat', output_dir), ...
        'experiment_results');
end

end


function experiment_results = lse_experiment(experiment_config, lse_config)
%LSE_EXPERIMENT Cycle through combinations of parameters and test lse

height = lse_config.height;
width = lse_config.width;
num_points = height * width;

% setup function mean and variance
f_mu = zeros(num_points, 1);
f_Sigma = feval(lse_config.cov_func, ...
    [log(lse_config.sigma_kernel), log(lse_config.kernel_scale)], ...
    lse_config.points);
f_Sigma = 1e-10 + (f_Sigma + f_Sigma') / 2; 

% setup horizons
num_horizons = size(experiment_config.horizons, 2);
num_path_penalties = size(experiment_config.path_penalties, 2);
num_beams = size(experiment_config.beam_sizes, 2);
num_dpp = size(experiment_config.dec_pps, 2);
output_dir = experiment_config.output_dir;
num_function_samples = experiment_config.num_function_samples;

num_methods = num_horizons * num_path_penalties * num_beams * num_dpp + 2;

experiment_results = cell(num_methods, num_function_samples);

for k = 1:num_function_samples
    % sample a function from the gp
    f = mvnrnd(f_mu, f_Sigma);
    f_grid = reshape(f, [height, width]);
    start_ind = randsample(num_points, 1);
    
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
    experiment_results{2, k} = trial_results;
    class_results_index = 3;
    
    % loop through param variations
    for a = 1:num_horizons
        cur_horizon = experiment_config.horizons(a);
        for b = 1:num_path_penalties
            cur_path_penalty = experiment_config.path_penalties(b);
            for c = 1:num_beams
                cur_beam_size = experiment_config.beam_sizes(c);
                for d = 1:num_dpp
                    cur_dpp = experiment_config.dec_pps(d);
                    
                    if experiment_config.log_results
                        fprintf('Evaluating LSE performance on function %d:\n', k);
                        fprintf('horizon: %d\n', cur_horizon);
                        fprintf('path penalty: %f\n', cur_path_penalty);
                        fprintf('beam size: %d\n', cur_beam_size);
                        fprintf('dec pp: %d\n', cur_dpp);
                    end

                    % set new config
                    lse_config.horizon = cur_horizon;
                    lse_config.path_penalty = cur_path_penalty;
                    lse_config.beam_size = cur_beam_size;
                    lse_config.use_dec_path_penalty = cur_dpp;
                    
                    % discrete gp lse
                    trial_results = struct();
                    [class_results, path_lengths, losses, class_images] = ...
                        discrete_gp_lse(f_grid, lse_config, start_ind);
                    trial_results.class_results = class_results;
                    trial_results.path_lengths = path_lengths;
                    trial_results.losses = losses;
                    trial_results.class_images = class_images;
                    
                    % store experiment results
                    experiment_results{class_results_index, k} = trial_results;
                    save(sprintf('%s/temp_lse_results.mat', output_dir), ...
                        'experiment_results');
                    
                    class_results_index = class_results_index + 1;
                end
            end
        end
    end
end

end


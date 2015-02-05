% try out some path gp ucb stuff

experiment_config = struct();
experiment_config.path_penalties = [0, 1e-3, 1e-2, 1e-1];
experiment_config.horizons = [1, 3, 5];
experiment_config.beam_sizes = [1, 3, 5];
experiment_config.dec_pps = [0, 1];

experiment_config.log_results = 1;
experiment_config.num_function_samples = 4;
experiment_config.output_dir = 'results/lse';

lse_config = struct();
% params of grid
lse_config.height = 50;
lse_config.width = 50;
num_points = lse_config.height * lse_config.width;
[X, Y] = meshgrid(1:lse_config.width, 1:lse_config.height);
lse_config.points = [X(:), Y(:)];

% params of the gp
lse_config.sigma_kernel = 4.5;
lse_config.kernel_scale = 1;
lse_config.sigma_noise = 0.1;
lse_config.mean_func = [];
lse_config.cov_func = @covSEiso;
lse_config.lik_func = @likGauss;

% algor params
lse_config.num_iters = 250;
lse_config.path_penalty = 25e-2;
lse_config.beam_size = 10;
lse_config.horizon = 3;

% results
lse_config.use_dec_path_penalty = 1;%1;
lse_config.store_classification_images = 1;
lse_config.class_res_rate = 10;
lse_config.class_image_rate = 10;
lse_config.tp_len = 10;
lse_config.vis_path = 0;
lse_config.vis_class_im = 0;

% delta
lse_config.delta = 0.99;
lse_config.h = 0; % explicit level we are trying to estimate
lse_config.tol = 1e-2;
lse_config.f_rkhs_norm = 1;

experiment_results = lse_experiment(experiment_config, lse_config);


%% analyze results
[avg_u, avg_f, avg_p, labels] = analyze_lse_results(experiment_results, experiment_config);



function [class_results, path_lengths, class_losses, class_images, iter_times] = ...
    discrete_gp_lse_pose_prior(f_grid, config, start_ind)
%GP_LSE runs the gaussian process level set estimation algorithms with
%various parameters (uses uniform initial prior atm)
%   f_grid - grid of samples of true function to test on
%   config - struct with params of algorithm to run
%   start_ind - index to seed algorithm with

% data params
f = f_grid(:);
[height, width] = size(f_grid);
num_points = height * width;
sigma_kernel = config.sigma_kernel;
kernel_scale = config.kernel_scale;
cov_func = config.cov_func;
mean_grid = config.mean_func; % ONLY SUPPORTS THOSE WITHOUT HYPERS
lik_func = config.lik_func;

[X, Y] = meshgrid(1:width, 1:height);
points = [X(:), Y(:)];

% randomly choose initial index if not specified
if nargin < 3
    start_ind = randsample(num_points, 1); 
end

% algorithm params
num_iters = config.num_iters;
path_penalty = config.path_penalty; %penalty for traveling from x to y
sigma_noise = config.sigma_noise; % measurement noise
horizon = config.horizon; % planning horizon
beam_size = config.beam_size; % size of beam used for BSP
f_rhks_norm = config.f_rkhs_norm; % necessary for non-GP functions

% math params
delta = config.delta;
h = config.h; % explicit level we are trying to estimate
tol = config.tol; % tolerance of lse algorithm

% setup hyperparameters
hyp = struct();
hyp.cov = [log(sigma_kernel), log(kernel_scale)];
hyp.lik = log(sigma_noise);

% read in pose params
num_pose_particles = config.num_pose_particles;
r_sigma_resample = config.r_sigma_resample;
t_sigma_resample = config.t_sigma_resample;
s_sigma_resample = config.s_sigma_resample;
min_rot = config.min_rot;
max_rot = config.max_rot;
min_trans = config.min_trans;
max_trans = config.max_trans;
min_scale = config.min_scale;
max_scale = config.max_scale;
num_f_samples = config.num_f_samples;
grid_center = [width / 2, height / 2]; % for pose transforms

% get ground truth values
true_above = f > h;
true_below = f < h;
f_above_grid = reshape(true_above, [height, width]);
f_below_grid = reshape(true_below, [height, width]);
truth_im = zeros(height, width, 3);
truth_im(:,:,1) = f_above_grid; % red above
truth_im(:,:,2) = f_below_grid; % green below

% set up active point and function sample buffers
eps_noise = normrnd(0, sigma_noise);
h_points = h * ones(num_points, 1);
active_points = [points(start_ind, :)];
active_y= [f(start_ind) + eps_noise];

% buffers for classification results
pred_above= zeros(num_points, 1);
pred_below = zeros(num_points, 1);
class_losses = zeros(num_iters, 1);
iter_times = zeros(num_iters, 1);
class_results = cell(1, 1);
class_result_index = 1;
class_images = cell(1, 1);
if config.store_classification_images
    class_im_index = 1;
end

% sample initial pose and scale particles from uniform
t_particles = zeros(num_pose_particles, 2); % translation particles
r_particles = zeros(num_pose_particles, 1); % rotation particles
s_particles = zeros(num_pose_particles, 1); % scale particles
particle_weights = (1.0 / num_pose_particles) * ...
    ones(num_pose_particles, 1); % all same relatively bc of unif prior

fig1 = figure(20);
fig2 = figure(21);
fig3 = figure(22);

t_particles(:,1) = unifrnd(min_trans(1) * ones(num_pose_particles,1), ...
                           max_trans(1) * ones(num_pose_particles,1));
t_particles(:,2) = unifrnd(min_trans(2) * ones(num_pose_particles,1), ...
                           max_trans(2) * ones(num_pose_particles,1));
r_particles(:) = unifrnd(min_rot * ones(num_pose_particles,1), ...
                         max_rot * ones(num_pose_particles,1));
s_particles(:) = unifrnd(min_scale * ones(num_pose_particles,1), ...
                         max_scale * ones(num_pose_particles,1));
t = 1; 
num_unclassified = num_points;
while t <= num_iters && num_unclassified > 0
    fprintf('Iteration %d\n', t);
    start_time = tic;
    
    % calculate new noise scaling
    beta_t = 2 * log(num_points * t * pi^2 / (6 * delta));
    num_unclassified = sum(pred_above == 0 & pred_below == 0);
    path_penalty_t = path_penalty;
    if config.use_dec_path_penalty == 1
        path_penalty_t = (num_unclassified / num_points) * path_penalty;
    end
    if config.use_dec_path_penalty == 2
        path_penalty_t = sqrt(num_unclassified / num_points) * path_penalty;
    end
    if config.use_dec_path_penalty == 3
        path_penalty_t = (num_unclassified / num_points)^(0.25) * path_penalty;
    end
    
    % predict values over grid
    [path_score, ucb_max, ucb_min, dist_penalty, mu_pred, sig_pred] = ...
        next_best_points(active_points, active_y, points, beta_t, h_points, ...
                         path_penalty_t, hyp, mean_grid, cov_func, lik_func, ...
                         r_particles, t_particles, s_particles, ...
                         particle_weights, grid_center, num_f_samples);           
                     
    % classify points according to ambiguity score
    pred_below(ucb_max < tol & pred_above == 0) = 1;
    pred_above(ucb_min < tol & pred_below == 0) = 1;

    % reduce score of classified points to prevent selecting
    path_score(pred_above == 1 | pred_below == 1) = -1e6;                 
    
    num_unclassified = sum(~pred_below & ~pred_above);
    fprintf('Num unclassified: %d\n', num_unclassified);
    
    % go back through path to find best
    best_score_ind  = find(path_score == max(path_score));
    next_ind = best_score_ind(1);
    next_point = points(next_ind, :);
    
    % compute the loss at each timestep
    ambig = min([ucb_max, ucb_min], [], 2);
    ambig(pred_above == 1 | pred_below == 1) = 0; % zero out classified points
    class_losses(t) = max(ambig) + dist_penalty(next_ind);
    
    % compute classification error at this timestep
    if mod(t, config.class_res_rate) == 0
        class_results{class_result_index} = ...
            lse_class_accuracy(true_above, true_below, pred_above, pred_below);
        class_results{class_result_index}.iteration = t;
        class_result_index = class_result_index + 1;
    end
    
    % add next point chosen to the active set
    eps_noise = normrnd(0, sigma_noise);
    next_y = f(next_ind) + eps_noise;
    
    % update particle weights
    particle_weights = update_particles(active_points, active_y,  ...
        next_point, next_y, num_f_samples, sigma_noise, hyp, ...
        mean_grid, cov_func, lik_func, r_particles, t_particles, ...
        s_particles, particle_weights, grid_center, height, width);
    
    mid_time = toc(start_time);
    
    % create visualizations
    bp_ind = find(particle_weights == max(particle_weights));
    mean_r  = sum(r_particles .* particle_weights);
    mean_t  = sum(t_particles .* [particle_weights particle_weights]);
    mean_s  = sum(s_particles .* particle_weights);
    tf = tf_from_particle(r_particles(bp_ind,:), t_particles(bp_ind,:), s_particles(bp_ind,:));
    mean_warped = warp_mean_function(tf, mean_grid, grid_center);    
    tf_mean = tf_from_particle(mean_r, mean_t, mean_s);
    mean_mean_warped = warp_mean_function(tf_mean, mean_grid, grid_center);    
    
    set(0, 'CurrentFigure', fig1);
    clf;
    % figure(20);
    subplot(1,3,1);
    imagesc(mean_warped);
    hold on;
    scatter(active_points(:,1), active_points(:,2), '+y', 'LineWidth', 2);
    title('Best Hypothesis');
    subplot(1,3,2);
    imagesc(mean_mean_warped);
    hold on;
    scatter(active_points(:,1), active_points(:,2), '+y', 'LineWidth', 2);
    title('Mean Hypothesis');
    subplot(1,3,3);
    imagesc(reshape(f, [height, width]));
    hold on;
    scatter(active_points(:,1), active_points(:,2), '+y', 'LineWidth', 2);
    title('True Function');
    drawnow;
    
    set(0, 'CurrentFigure', fig3);
    %figure(22);
    clf;
    for i = 1:num_pose_particles
        tf = tf_from_particle(r_particles(i,:), t_particles(i,:), s_particles(i,:));
        mean_warped = warp_mean_function(tf, mean_grid, grid_center); 
        subplot(ceil(sqrt(num_pose_particles)), ceil(sqrt(num_pose_particles)), i);
        imagesc(mean_warped);
        hold on;
        title(sprintf('Particle %d\nweight %.04f', i, particle_weights(i)));
    end
    drawnow;
    
	% update active set
    active_points = [active_points; next_point];
    active_y = [active_y; next_y];
    
    % resample the particles (if necessary)
    start_time = tic;
    
    effective_sample_size = 1.0 / sum(particle_weights.^2);
    fprintf('Effective sample size: %f\n', effective_sample_size);
    if effective_sample_size < config.resample_size
        resample_inds = mnrnd(1, particle_weights, num_pose_particles);
        [~, resample_inds] = find(resample_inds == 1);
        r_particles = r_particles(resample_inds);
        t_particles = t_particles(resample_inds,:);
        s_particles = s_particles(resample_inds);
        particle_weights = (1.0 / num_pose_particles) * ...
            ones(num_pose_particles, 1);
        
        % mixture of gaussian version
        if config.gaussian_resample
            % fit indep kernel distributions
            new_r_particles = mvnrnd(r_particles, r_sigma_resample);
            r_pdf = mvnpdf(new_r_particles, r_particles, r_sigma_resample);
            new_t_particles = mvnrnd(t_particles, [t_sigma_resample, t_sigma_resample]);
            t_pdf = mvnpdf(new_t_particles, t_particles, [t_sigma_resample, t_sigma_resample]);
            new_s_particles = mvnrnd(s_particles, s_sigma_resample);
            s_pdf = mvnpdf(new_s_particles, s_particles, s_sigma_resample);
            r_particles = new_r_particles;
            t_particles = new_t_particles;
            s_particles = new_s_particles;
            
            % normalize weights
            particle_weights = r_pdf .* t_pdf .* s_pdf;
            particle_weights = particle_weights / sum(particle_weights);
        end
        
        % kernel version (assume independence)
        if config.kernel_resample
            % fit indep kernel distributions
            r_pd = fitdist(r_particles, 'Kernel');
            t_pd = fitdist(t_particles, 'Kernel');
            s_pd = fitdist(s_particles, 'Kernel');
            r_particles = random(r_pd, num_pose_particles, 1);
            r_pdf = pdf(r_pd, r_particles);
            t_particles = random(t_pd, num_pose_particles, 1);
            t_pdf = pdf(t_pd, t_particles);
            s_particles = random(s_pd, num_pose_particles, 1);
            s_pdf = pdf(s_pd, s_particles);

            % normalize weights
            particle_weights = r_pdf .* t_pdf .* s_pdf;
            particle_weights = particle_weights / sum(particle_weights);
        end
    end
    end_time = toc(start_time);
    iter_times(t) = mid_time + end_time;
    
    % compute visualization of region classification
    if config.store_classification_images && mod(t, config.class_image_rate) == 0
        points_above_grid = reshape(pred_above, [height, width]);
        points_below_grid = reshape(pred_below, [height, width]);
        classification_im = zeros(height, width, 3);
        classification_im(:,:,1) = points_above_grid;
        classification_im(:,:,2) = points_below_grid;
        class_images{class_im_index} = classification_im;
        class_im_index = class_im_index + 1;
        
        set(0, 'CurrentFigure', fig2);
        clf;
        imshow(class_images{class_im_index-1});
        title('Classification');
        drawnow;
    end
    
    if config.vis_path
        path_lin_ind = sub2ind([height, width], active_points(:,2), active_points(:,1));
        figure(10);
        clf;
        surf(X, Y, f_grid);
        view(45,85);
        hold on;

        level_set = h * ones(height, width);
        ls_colors = ones(height, width, 3);
        surf(level_set, ls_colors);

        if t > config.tp_len
            plot3(active_points(end-config.tp_len:end,1), active_points(end-config.tp_len:end,2), ...
                f(path_lin_ind(end-config.tp_len:end))+1, 'b', 'LineWidth', 5);
            scatter3(active_points(end,1), active_points(end,2), f(path_lin_ind(end))+1, 100, 'm', 'filled');
            pause(0.05);
        end
    end
    
    t = t+1;
end

% analyze path lengths
path_lengths = zeros(num_iters, 1);
path_diff= active_points(2:end,:) - active_points(1:(end-1),:);
path_norm = sqrt(sum(path_diff.^2, 2));
path_lengths(1:size(path_norm,1)) = cumsum(path_norm);
path_lengths(size(path_norm,1)+1:end) = path_lengths(size(path_norm,1));
times = (1:num_iters);

class_losses = class_losses + path_penalty * path_lengths ./ times';

if t < num_iters
    class_results{class_result_index} = ...
            lse_class_accuracy(true_above, true_below, pred_above, pred_below);
end

if config.vis_class_im
    figure(2);
    subplot(1,2,1);
    imshow(class_images{end});
    title('LSE Predicted');
    subplot(1,2,2);
    imshow(truth_im);
    title('Ground Truth');
end

end

function [scores, ucb_max, ucb_min, dist_penalty, mu_pred, sig_pred] = ...
    next_best_points(active_points, active_y, points, beta, h_points, ...
                     path_penalty, hyp, mean_grid, cov_func, lik_func, ...
                     r_particles, t_particles, s_particles, particle_weights, ...
                     grid_center, num_f_samples)
    
    % predict function value at points and compute upper confidence bound
    num_points = size(points, 1);
    num_particles = size(r_particles, 1);
    height = max(points(:,1));
    width = max(points(:,2));
    
    mu_pred_arr = zeros(num_points, num_particles);
    sig_pred_arr = zeros(num_points, num_particles);
    for i = 1:num_particles
        % transform mean by particle
        tf = tf_from_particle(r_particles(i,:), t_particles(i,:), s_particles(i,:));
        mean_warped = warp_mean_function(tf, mean_grid, grid_center);
        
        % create mean function (TODO: double check)
        mean_func = @(h, x) mean_warped(sub2ind([height, width], x(:,2), x(:,1)));
        
        % loop through particles and transform
        [mu_pred, sig_pred] = ...
            gp(hyp, @infExact, mean_func, cov_func, lik_func, ...
               active_points, active_y, points);
        
        % weighted sum by particles
        mu_pred_arr(:,i) = mu_pred;
        sig_pred_arr(:,i) = sig_pred;
    end
    mu_weighted = mu_pred_arr .* repmat(particle_weights', [num_points, 1]);
    mu_pred = sum(mu_weighted, 2) / sum(particle_weights);
    
    % compute sigma
    sig_pred = zeros(num_points, 1);
    for i = 1:num_particles
        f_samples = mvnrnd(mu_pred_arr(:,i), diag(sig_pred_arr(:,i)), num_f_samples);
        f_samples = f_samples'; % fix mvnrnd auto-transpose
        sig_i = (f_samples - repmat(mu_pred, [1, num_f_samples])).^2;
        sig_pred = sig_pred + sum(sig_i, 2) * particle_weights(i) / num_f_samples;
    end
    
    % compute upper / lower confidence bounds and overlap with level set
    ucb_max = mu_pred + sqrt(beta) * sig_pred - h_points;
    ucb_min = h_points - mu_pred + sqrt(beta) * sig_pred;
    ambig = min([ucb_max, ucb_min], [], 2);
    
    % compute and apply distance penalty
    prev_point = active_points(end,:);
    dist_penalty = points - repmat(prev_point, [num_points, 1]);
    dist_penalty = path_penalty * sqrt(sum(dist_penalty.^2, 2));
    scores = ambig - dist_penalty;
end

% call before adding next point to active points
function [new_particle_weights] = ...
    update_particles(active_points, active_y, next_point, next_y, ...
                     num_f_samples, sigma_noise, hyp, mean_grid, cov_func, lik_func, ...
                     r_particles, t_particles, s_particles, ...
                     particle_weights, grid_center, height, width)
    
    % predict function value at points and compute upper confidence bound
    num_particles = size(r_particles, 1);
    new_particle_weights = zeros(size(particle_weights));
    [X, Y] = meshgrid(1:width, 1:height);
    points = [X(:), Y(:)];
    
    for i = 1:num_particles
        % transform mean by particle
        tf = tf_from_particle(r_particles(i,:), t_particles(i,:), s_particles(i,:));
        mean_warped = warp_mean_function(tf, mean_grid, grid_center);
        
        % create mean function (TODO: double check)
        mean_func = @(h, x) mean_warped(sub2ind([height, width], x(:,2), x(:,1)));
        
        % loop through particles and transform
        [mu_pred, sig_pred] = ...
            gp(hyp, @infExact, mean_func, cov_func, lik_func, ...
               active_points, active_y, next_point);
        
        % generate samples from distribution on f
        f_samples = normrnd(mu_pred, sig_pred, num_f_samples, 1);
        
        % integrate by computing pdf of y given f samples and summing
        % uses importance sampling
        y_vec = repmat(next_y, [num_f_samples, 1]);
        y_liks = normpdf(y_vec, f_samples, sigma_noise);
        new_particle_weights(i) = sum(y_liks) * particle_weights(i) / sum(particle_weights);
    end
    new_particle_weights = new_particle_weights / sum(new_particle_weights);
end

function tf = tf_from_particle(r, t, s)
    tf = struct();
    pitch = r;  
    dcm = angle2dcm(pitch, 0, 0);
    R_mat = dcm(1:2, 1:2);
    
    tf.t = t';
    tf.R = R_mat;
    tf.s = s;
end


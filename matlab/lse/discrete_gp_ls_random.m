function [class_results, path_lengths, class_losses, class_images] = ...
    discrete_gp_ls_random(f_grid, config, start_ind)
%GP_LSE runs the gaussian process level set estimation algorithms with
%various parameters
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
mean_func = config.mean_func; % ONLY SUPPORTS THOSE WITHOUT HYPERS
lik_func = config.lik_func;
sigma_noise = config.sigma_noise;
path_penalty_t = 0;
delta = config.delta;

[X, Y] = meshgrid(1:width, 1:height);
points = [X(:), Y(:)];

% randomly choose initial index if not specified
if nargin < 3
    start_ind = randsample(num_points, 1); 
end

% algorithm params
num_iters = config.num_iters;

% math params
h = config.h; % explicit level we are trying to estimate
tol = config.tol; % tolerance of lse algorithm

% setup hyperparameters
hyp = struct();
hyp.cov = [log(sigma_kernel), log(kernel_scale)];
hyp.lik = log(sigma_noise);

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
pred_above = zeros(num_points, 1);
pred_below = zeros(num_points, 1);
class_losses = zeros(num_iters, 1);
class_results = cell(1, 1);
class_result_index = 1;
class_images = cell(1, 1);
if config.store_classification_images
    class_im_index = 1;
end

for t = 1:num_iters
    beta_t = 2 * log(num_points * t * pi^2 / (6 * delta));
    
    % classify points
    [~, ucb_max, ucb_min, dist_penalty, ~, ~] = ...
        next_best_points(active_points, active_y, points, beta_t, h_points, ...
                         path_penalty_t, hyp, mean_func, cov_func, lik_func);
                     
    % classify points according to ambiguity score
    pred_below(ucb_max < tol & pred_above == 0) = 1;
    pred_above(ucb_min < tol & pred_below == 0) = 1;

    % compute the loss
%     above_penalty = 0;
%     if sum(pred_above) > 0
%         above_penalty = (1.0 / sum(pred_above)) * ...
%             sum(pred_above .* max(zeros(num_points, 1), f - h_points));
%     end
%     below_penalty = 0;
%     if sum(pred_below) > 0
%         below_penalty = (1.0 / sum(pred_below)) * ...
%             sum(pred_below .* max(zeros(num_points, 1), h_points - f));
%     end
%     class_losses(t) = above_penalty + below_penalty;
%     
    
    % compute classification error at this timestep
    if mod(t, config.class_res_rate) == 0
        class_results{class_result_index} = ...
            lse_class_accuracy(true_above, true_below, pred_above, pred_below);
        class_results{class_result_index}.iteration = t;
        class_result_index = class_result_index + 1;
    end
    
    % add next point chosen to the active set
    next_ind = randsample(num_points, 1);
    next_point = points(next_ind, :);
    eps_noise = normrnd(0, sigma_noise);
    next_y = f(next_ind) + eps_noise;
    active_points = [active_points; next_point];
    active_y = [active_y; next_y];
    
    ambig = min([ucb_max, ucb_min], [], 2);
    ambig(pred_above == 1 | pred_below == 1) = 0; % zero out classified points
    class_losses(t) = max(ambig) + dist_penalty(next_ind);
    
    % compute visualization of region classification
    if config.store_classification_images && mod(t, config.class_image_rate) == 0
        points_above_grid = reshape(pred_above, [height, width]);
        points_below_grid = reshape(pred_below, [height, width]);
        classification_im = zeros(height, width, 3);
        classification_im(:,:,1) = points_above_grid;
        classification_im(:,:,2) = points_below_grid;
        class_images{class_im_index} = classification_im;
        class_im_index = class_im_index + 1;
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
end

% analyze path lengths
path_diff= active_points(2:end,:) - active_points(1:(end-1),:);
path_norm = sqrt(sum(path_diff.^2, 2));
path_lengths = cumsum(path_norm);
times = (1:num_iters);

class_losses = class_losses + config.path_penalty * path_lengths ./ times';

if config.vis_class_im
    figure(3);
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
                     path_penalty, hyp, mean_func, cov_func, lik_func)
    
    % predict function value at points and compute upper confidence bound
    num_points = size(points, 1);
    [mu_pred, sig_pred] = ...
        gp(hyp, @infExact, mean_func, cov_func, lik_func, ...
           active_points, active_y, points);
    ucb_max = mu_pred + sqrt(beta) * sig_pred - h_points;
    ucb_min = h_points - mu_pred + sqrt(beta) * sig_pred;
    ambig = min([ucb_max, ucb_min], [], 2);
    
    % compute and apply distance penalty
    prev_point = active_points(end,:);
    dist_penalty = points - repmat(prev_point, [num_points, 1]);
    dist_penalty = path_penalty * sqrt(sum(dist_penalty.^2, 2));
    scores = ambig - dist_penalty;
end

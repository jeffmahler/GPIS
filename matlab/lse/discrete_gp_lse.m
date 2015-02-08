function [class_results, path_lengths, class_losses, class_images] = discrete_gp_lse(f_grid, config, start_ind)
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
class_results = cell(1, 1);
class_result_index = 1;
class_images = cell(1, 1);
if config.store_classification_images
    class_im_index = 1;
end

% set up buffers for beam search horizon path planning
path_scores = zeros(beam_size, horizon);
path_pointers = zeros(beam_size, horizon);
path_points = zeros(beam_size, horizon);
path_meas = zeros(beam_size, horizon);
path_pred_above = zeros(beam_size, num_points);
path_pred_below = zeros(beam_size, num_points);

next_point_scores = zeros(beam_size, beam_size);
next_point_pointers = zeros(beam_size, beam_size);
next_point_candidates = zeros(beam_size, beam_size);
next_point_meas = zeros(beam_size, beam_size);
next_point_pred_above = zeros(beam_size, num_points);
next_point_pred_below = zeros(beam_size, num_points);

for t = 1:num_iters
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
    
    %use horizon and beam search for planning
    path_pointers(:,1) = ones(beam_size, 1); % point second to last to the first point
    next_point_scores = -1e6 * ones(beam_size, beam_size);
    beam_size_iter = 1;

    for step = 1:horizon
        beam_beta = 2 * log(num_points * (t + step - 1) * pi^2 / (6 * delta));
        
        % for each set of candidate paths
        for k = 1:beam_size_iter
            if step > 1
                % get the data from the next point
                cur_point = points(path_points(k, step-1), :);
                cur_meas = path_meas(k, step-1);
                cur_pointer = path_pointers(k, step-1);
                beam_pred_above = path_pred_above(cur_pointer, :)';
                beam_pred_below = path_pred_below(cur_pointer, :)';

                % get prev point list
                prev_points = cur_point;
                prev_meas = cur_meas;
                pointer = k;
                for j = fliplr(2:(step-1))
                    pointer = path_pointers(pointer, j);
                    prev_points = [prev_points;
                        points(path_points(pointer, j-1), :)];
                    prev_meas = [prev_meas;
                        path_meas(pointer, j-1)];
                end
                beam_active_points = [active_points; prev_points];
                beam_active_y = [active_y; prev_meas];
            else
                beam_active_points = active_points;
                beam_active_y = active_y;
                beam_pred_above = pred_above;
                beam_pred_below = pred_below;
            end

            % predict values over grid
            [beam_path_score, beam_ucb_max, beam_ucb_min, ~, beam_mu, ~] = ...
                next_best_points(beam_active_points, beam_active_y, points, beam_beta, h_points, ...
                                 path_penalty_t, hyp, mean_func, cov_func, lik_func);

            % classify points according to ambiguity score
            beam_pred_below(beam_ucb_max < tol & beam_pred_above == 0) = 1;
            beam_pred_above(beam_ucb_min < tol & beam_pred_below == 0) = 1;

            % reduce score of classified points to prevent selecting
            beam_path_score(beam_pred_above == 1 | beam_pred_below == 1) = -1e6;                 

            % find top values
            [~, sorted_indices] = sort(beam_path_score, 'descend');
            top_K_indices = sorted_indices(1:beam_size);
            top_K_path_scores = beam_path_score(top_K_indices);
            top_K_mean = beam_mu(top_K_indices);

            % next points
            prev_path_score = 0;
            if step > 1
                prev_path_score = path_scores(k,step-1);
            end
            next_point_candidates(k, :) = top_K_indices;
            next_point_scores(k, :) = top_K_path_scores + prev_path_score;
            next_point_pointers(k, :) = k * ones(1, beam_size);
            next_point_meas(k, :) = top_K_mean;
            next_point_pred_above(k, :) = beam_pred_above;
            next_point_pred_below(k, :) = beam_pred_below;

        end
        % compute beam_size best paths from those returned
        [~, sorted_indices] = sort(next_point_scores(:), 'descend');
        top_K_indices = sorted_indices(1:beam_size);
        top_K_point_indices = next_point_candidates(top_K_indices);
        top_K_path_scores = next_point_scores(top_K_indices);
        top_K_pointers = next_point_pointers(top_K_indices);
        top_K_meas = next_point_meas(top_K_indices);

        path_scores(:, step) = top_K_path_scores;
        path_pointers(:, step) = top_K_pointers;
        path_points(:, step) = top_K_point_indices;
        path_meas(:, step) = top_K_meas;

        beam_size_iter = beam_size; %use true beam size after first iter
    end
    
    % go back through path to find best
    best_score_ind  = find(path_scores(:,horizon) == max(path_scores(:,horizon)));
    pointer = best_score_ind(1);
    for j = fliplr(2:horizon)
        pointer = path_pointers(pointer, step);
    end
    next_ind = path_points(pointer,1);
    next_point = points(next_ind, :);
    
    % classify points
    [scores, ucb_max, ucb_min, dist_penalty, ~, ~] = ...
        next_best_points(active_points, active_y, points, beta_t, h_points, ...
                         path_penalty_t, hyp, mean_func, cov_func, lik_func);
                     
    % classify points according to ambiguity score
    pred_below(ucb_max < tol & pred_above == 0) = 1;
    pred_above(ucb_min < tol & pred_below == 0) = 1;

    % reduce score of classified points to prevent selecting
    scores(pred_above == 1 | pred_below == 1) = -1e6;  
    best_ind = find(scores == max(scores));
    if best_ind(1) ~= next_ind
        stop = 1;
    end

    % compute the loss at each timestep
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
    active_points = [active_points; next_point];
    active_y = [active_y; next_y];
    
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

class_losses = class_losses + path_penalty * path_lengths ./ times';

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

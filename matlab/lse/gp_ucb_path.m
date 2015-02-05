% try out some path gp ucb stuff
%if false
% params of grid
height = 60;
width = 60;
num_points = height * width;
[X, Y] = meshgrid(1:width, 1:height);
points = [X(:), Y(:)];

% params of the gp
hyp = struct();
hyp.cov = [2, 0];
hyp.lik = log(0.1);
%hyp.mean = [0];
covfunc = {@covSEiso};
likfunc = {@likGauss};
f_mu = zeros(num_points, 1);
f_Sigma = feval(covfunc{:}, hyp.cov, points);
f_Sigma = 1e-10 + (f_Sigma + f_Sigma') / 2; 

% sample a function from the gp
f = mvnrnd(f_mu, f_Sigma);
f_grid = reshape(f, [height, width]);
f_max = max(f);
max_ind = find(f == f_max);
max_ind = max_ind(1);

% display sampled function
figure(1);
surf(X, Y, f_grid);
hold on;
scatter3(points(max_ind,1), points(max_ind,2), f_max, 100, 'g', 'filled');
%end
%% run gp ucb and path gp ucb
%if false
% set search params
num_iters = 500;%500;
Q = 10e-3;%eye(2);
delta = 0.99;
sigma_noise = exp(hyp.lik);

% get initial point
%rng(100);
start_ind = randsample(num_points, 1);
active_points_ucb = [points(start_ind, :)];
active_y_ucb = [f(start_ind) + normrnd(0, sigma_noise)];
active_points_path = [points(start_ind, :)];
active_y_path = [f(start_ind) + normrnd(0, sigma_noise)];
active_points_context = [points(start_ind, :)];%, sum((points(start_ind, :) - points(start_ind, :)).^2, 2)];
active_y_context = [f(start_ind) + normrnd(0, sigma_noise)];
context = points(start_ind, :);

hyp_context = hyp;
hyp_context.cov = [hyp.cov, hyp.cov];
mask_fn1 = {'covMask', {[1,1,0], 'covSEiso'}};
mask_fn2 = {'covMask', {[0,0,1], 'covSEiso'}};
context_covfunc = {'covProd', ...
    {mask_fn1, mask_fn2}};
%     {@covMask, {[1,1,0,0], @covSEiso}}, ...
%     {@covMask, {[1,1,0,0], @covSEiso}}};
B = 1; % rkhs norm of function

opt_reward_path = zeros(num_iters, 1);
opt_reward_context = zeros(num_iters, 1);

for t = 1:num_iters
    beta = 2 * log(num_points * t * pi^2 / (6 * delta));
    gamma = log(t)^(4);
    beta_context = beta;%B^2 + 10 * gamma * log(t / delta)^3;
    
    % predict next points and compute ucb
    [mu_ucb, sig_ucb] = gp(hyp, @infExact, [], covfunc{1}, likfunc{1}, ...
        active_points_ucb, active_y_ucb, points);
    ucb = mu_ucb + sqrt(beta) * sig_ucb;
    
    % predict next points using path
    [mu_path, sig_path] = gp(hyp, @infExact, [], covfunc{1}, likfunc{1}, ...
        active_points_path, active_y_path, points);
    ucb_path = mu_path + sqrt(beta) * sig_path;
    dist_q_ucb = points - repmat(active_points_path(end,:), [num_points, 1]);
    dist_q_ucb = Q * sqrt(sum(dist_q_ucb.^2, 2));
    path_score = ucb_path - dist_q_ucb;
    path_true_reward = f' - dist_q_ucb;
    opt_reward_path(t) = max(path_true_reward);
    
    % predict next points using context ucb
    mean_context = sum((points - repmat(context, [num_points, 1])).^2, 2);
    points_context = [points, mean_context];
    dist_active_points = Q*sqrt(sum((active_points_context - repmat(context, [t, 1])).^2,2));
    mean_context_func = @(hyp, x) - Q*sqrt(sum((x - repmat(context, [size(x,1), 1])).^2, 2));
    [mu_context, sig_context] = gp(hyp, @infExact, mean_context_func, covfunc, likfunc{1}, ...
        active_points_context, active_y_context - dist_active_points, points);
    ucb_context = mu_context + sqrt(beta_context) * sig_context;
    dist_q_context = points - repmat(context, [num_points, 1]);
    dist_q_context = Q * sqrt(sum(dist_q_context.^2, 2));
    context_true_reward = f' - dist_q_context;
    opt_reward_context(t) = max(context_true_reward);
    
    % add next ucb point to set
    next_ind_ucb = find(ucb == max(ucb));
    next_ind_ucb = next_ind_ucb(1);%randsample(size(next_ind_ucb, 1), 1));
    next_point_ucb = points(next_ind_ucb, :);
    next_y_ucb = f(next_ind_ucb) + normrnd(0, sigma_noise);
    active_points_ucb = [active_points_ucb; next_point_ucb];
    active_y_ucb = [active_y_ucb; next_y_ucb];
    
    % add next path point to set
    noise_val = normrnd(0, sigma_noise);
    next_ind_path = find(path_score == max(path_score));
    next_ind_path = next_ind_path(1);%randsample(size(next_ind_path, 1), 1));
    next_point_path = points(next_ind_path, :);
    next_y_path = f(next_ind_path) + noise_val;
    active_points_path = [active_points_path; next_point_path];
    active_y_path = [active_y_path; next_y_path];
    
    % add next context point to set
    next_ind_context = find(ucb_context == max(ucb_context));
    next_ind_context = next_ind_context(1);%randsample(size(next_ind_ucb, 1), 1));
    next_point_context = points(next_ind_context, :);
    dist_q_penalty = Q * sum((context - next_point_context).^2, 2);
    
    next_y_context = f(next_ind_context) + noise_val;%- dist_q_penalty + normrnd(0, sigma_noise);
    active_points_context = [active_points_context; ...
                             next_point_context];%, sum((next_point_context - context).^2,2)];
    active_y_context = [active_y_context; next_y_context];
    context = next_point_context;
    
    % visualization
    ucb_lin_ind = sub2ind([height, width], active_points_ucb(:,2), active_points_ucb(:,1));
    path_lin_ind = sub2ind([height, width], active_points_path(:,2), active_points_path(:,1));
    context_lin_ind = sub2ind([height, width], active_points_context(:,2), active_points_context(:,1));
    if t > 10
    figure(2);
    clf;
    surf(X, Y, f_grid);
    view(45,70);
    hold on;
    scatter3(points(max_ind,1), points(max_ind,2), f_max, 100, 'g', 'filled');
    plot3(active_points_ucb(end-10:end,1), active_points_ucb(end-10:end,2), f(ucb_lin_ind(end-10:end))+1, 'b', 'LineWidth', 5);
    scatter3(active_points_ucb(end,1), active_points_ucb(end,2), f(ucb_lin_ind(end))+1, 100, 'm', 'filled');
    plot3(active_points_path(end-10:end,1), active_points_path(end-10:end,2), f(path_lin_ind(end-10:end))+1, 'k', 'LineWidth', 5);
    scatter3(active_points_path(end,1), active_points_path(end,2), f(path_lin_ind(end))+1, 100, 'r', 'filled');
    plot3(active_points_context(end-10:end,1), active_points_context(end-10:end,2), f(context_lin_ind(end-10:end))+1, 'c', 'LineWidth', 5);
    scatter3(active_points_context(end,1), active_points_context(end,2), f(context_lin_ind(end))+1, 100, 'k', 'filled');
    end
    pause(0.01);
end

% analyze results
active_ucb_path_diff= active_points_ucb(2:end,:) - active_points_ucb(1:(end-1),:);
active_ucb_path_norm = sqrt(sum(active_ucb_path_diff.^2, 2));
active_ucb_path_length = sum(active_ucb_path_norm);

active_path_diff= active_points_path(2:end,:) - active_points_path(1:(end-1),:);
active_path_norm = sqrt(sum(active_path_diff.^2, 2));
active_path_length = sum(active_path_norm);

active_context_diff= active_points_context(2:end,1:2) - active_points_context(1:(end-1),1:2);
active_context_norm = sqrt(sum(active_context_diff.^2, 2));
active_context_length = sum(active_context_norm);

ucb_cum_reward = sum(f(ucb_lin_ind));
path_cum_reward = sum(f(path_lin_ind));
context_cum_reward = sum(f(context_lin_ind));

ucb_subopt = f_max - f(ucb_lin_ind(end));
path_subopt = f_max - f(path_lin_ind(end));
context_subopt = f_max - f(context_lin_ind(end));

eps = 1e-4;
ucb_time_to_opt = find(f(ucb_lin_ind) > f_max - eps);
if numel(ucb_time_to_opt) == 0
    ucb_time_to_opt = -1;
else
    ucb_time_to_opt = ucb_time_to_opt(1);
end

path_time_to_opt = find(f(path_lin_ind) > f_max - eps);
if numel(path_time_to_opt) == 0
    path_time_to_opt = -1;
else
    path_time_to_opt = path_time_to_opt(1);
end

fprintf('Path lengths\n');
fprintf('UCB:\t\t%f\n', active_ucb_path_length);
fprintf('Path:\t\t%f\n', active_path_length);
fprintf('Context:\t%f\n', active_context_length);

fprintf('Cum reward\n');
fprintf('UCB:\t\t%f\n', ucb_cum_reward);
fprintf('Path:\t\t%f\n', path_cum_reward);
fprintf('Context:\t%f\n', context_cum_reward);

fprintf('Final subopt\n');
fprintf('UCB:\t\t%f\n', ucb_subopt);
fprintf('Path:\t\t%f\n', path_subopt);
fprintf('Context:\t%f\n', context_subopt);
% 
% fprintf('Time to opt\n');
% fprintf('UCB:\t\t%f\n', ucb_time_to_opt);
% fprintf('Path penalty:\t%f\n', path_time_to_opt);

% plot the regret
figure(3);
clf;
plot(repmat(f_max, [num_iters, 1]), ':k', 'LineWidth', 2);
hold on;
plot(f(ucb_lin_ind), 'r', 'LineWidth', 2);
plot(f(path_lin_ind), 'g', 'LineWidth', 2);
plot(f(context_lin_ind), 'b', 'LineWidth', 2);
legend('Optimal', 'UCB', 'Path', 'Context', 'Location', 'Best');
xlabel('Iteration');
ylabel('Simple Reward');
title('Reward versus Iteration');

figure(4);
clf;
plot(cumsum(f_max - f(ucb_lin_ind(2:end)) + Q*active_ucb_path_norm'), 'r', 'LineWidth', 2);
hold on;
plot(cumsum(opt_reward_path' - f(path_lin_ind(2:end)) + Q*active_path_norm'), 'g', 'LineWidth', 2);
plot(cumsum(opt_reward_context' - f(context_lin_ind(2:end)) + Q*active_context_norm'), 'b', 'LineWidth', 2);
legend('UCB', 'Path', 'Context', 'Location', 'Best');
xlabel('Iteration');
ylabel('Cumulative Regret');
title('Cumulative Regret versus Iteration');

figure(5);
clf;
plot(cumsum(active_ucb_path_norm), 'r', 'LineWidth', 2);
hold on;
plot(cumsum(active_path_norm), 'g', 'LineWidth', 2);
plot(cumsum(active_context_norm), 'b', 'LineWidth', 2);
legend('UCB', 'Path', 'Context', 'Location', 'Best');
xlabel('Iteration');
ylabel('Path Length');
title('Path Length versus Iteration');
%end
%% level set versions

% set search params
num_iters = 10;%500;
Q = 1e-2;%eye(2);
delta = 0.99;
sigma_noise = exp(0.1);
h = 0;
h_points = h * ones(num_points, 1);
tol = 1e-2;
 
% get initial point
rng(100);
start_ind = randsample(num_points, 1);
active_points_ucb = [points(start_ind, :)];
active_y_ucb = [f(start_ind) + normrnd(0, sigma_noise)];
active_points_path = [points(start_ind, :)];
active_y_path = [f(start_ind) + normrnd(0, sigma_noise)];
ucb_above= zeros(num_points, 1);
ucb_below = zeros(num_points, 1);
path_above = zeros(num_points, 1);
path_below = zeros(num_points, 1);

for t = 1:num_iters
    beta = 2 * log(num_points * t * pi^2 / (6 * delta));
    
    % predict next points and compute ucb
    [mu_ucb, sig_ucb] = gp(hyp, @infExact, [], covfunc{1}, likfunc{1}, ...
        active_points_ucb, active_y_ucb, points);
    ucb_up = mu_ucb + sqrt(beta) * sig_ucb - h_points;
    ucb_low = h_points - mu_ucb + sqrt(beta) * sig_ucb;
    ambig = min([ucb_up, ucb_low], [], 2);
    ucb_below(ucb_up < tol & ucb_above == 0) = 1;
    ucb_above(ucb_low < tol & ucb_below == 0) = 1;
    ambig(ucb_above == 1) = -1e6;
    ambig(ucb_below == 1) = -1e6;
    
    % predict next points using path
    [mu_path, sig_path] = gp(hyp, @infExact, [], covfunc{1}, likfunc{1}, ...
        active_points_path, active_y_path, points);
    ucb_path_up = mu_ucb + sqrt(beta) * sig_path - h_points;
    ucb_path_low = h_points - mu_path + sqrt(beta) * sig_path;
    ambig_path = min([ucb_path_up, ucb_path_low], [], 2);
    dist_q_ucb = points - repmat(active_points_path(end,:), [num_points, 1]);
    dist_q_ucb = Q * sqrt(sum(dist_q_ucb.^2, 2));
    path_score = ambig_path - dist_q_ucb;
    path_below(ucb_path_up < tol & path_above == 0) = 1;
    path_above(ucb_path_low < tol & path_below == 0) = 1;
    path_score(path_above == 1) = -1e6;
    path_score(path_below == 1) = -1e6;
    
    % add next ucb point to set
    next_ind_ucb = find(ambig == max(ambig));
    next_ind_ucb = next_ind_ucb(1);%randsample(size(next_ind_ucb, 1), 1));
    next_point_ucb = points(next_ind_ucb, :);
    next_y_ucb = f(next_ind_ucb) + normrnd(0, sigma_noise);
    active_points_ucb = [active_points_ucb; next_point_ucb];
    active_y_ucb = [active_y_ucb; next_y_ucb];
    
    % add next path point to set
    next_ind_path = find(path_score == max(path_score));
    next_ind_path = next_ind_path(1);%randsample(size(next_ind_path, 1), 1));
    next_point_path = points(next_ind_path, :);
    next_y_path = f(next_ind_path) + normrnd(0, sigma_noise);
    active_points_path = [active_points_path; next_point_path];
    active_y_path = [active_y_path; next_y_path];
    
    % visualization
    ucb_lin_ind = sub2ind([height, width], active_points_ucb(:,2), active_points_ucb(:,1));
    path_lin_ind = sub2ind([height, width], active_points_path(:,2), active_points_path(:,1));
    figure(2);
    clf;
    surf(X, Y, f_grid);
    view(45,85);
    hold on;
    
    level_set = h * ones(height, width);
    ls_colors = ones(height, width, 3);
    surf(level_set, ls_colors);
    
    scatter3(points(max_ind,1), points(max_ind,2), f_max, 100, 'g', 'filled');
    plot3(active_points_ucb(:,1), active_points_ucb(:,2), f(ucb_lin_ind)+1, 'b', 'LineWidth', 5);
    scatter3(active_points_ucb(end,1), active_points_ucb(end,2), f(ucb_lin_ind(end))+1, 100, 'm', 'filled');
    plot3(active_points_path(:,1), active_points_path(:,2), f(path_lin_ind)+1, 'k', 'LineWidth', 5);
    scatter3(active_points_path(end,1), active_points_path(end,2), f(path_lin_ind(end))+1, 100, 'r', 'filled');
    
    pause(0.01);
end

%% plot classification
lse_above_grid = reshape(ucb_above, [height, width]);
lse_below_grid = reshape(ucb_below, [height, width]);
lse_im = zeros(height, width, 3);
lse_im(:,:,1) = lse_above_grid;
lse_im(:,:,2) = lse_below_grid;

path_above_grid = reshape(path_above, [height, width]);
path_below_grid = reshape(path_below, [height, width]);
path_im = zeros(height, width, 3);
path_im(:,:,1) = path_above_grid;
path_im(:,:,2) = path_below_grid;

f_above = f > h;
f_below = f < h;
f_above_grid = reshape(f_above, [height, width]);
f_below_grid = reshape(f_below, [height, width]);
truth_im = zeros(height, width, 3);
truth_im(:,:,1) = f_above_grid;
truth_im(:,:,2) = f_below_grid;

figure(5);
subplot(1,3,1);
imshow(lse_im);
title('LSE Predicted');
subplot(1,3,2);
imshow(path_im);
title('Path Predicted');
subplot(1,3,3);
imshow(truth_im);
title('Ground Truth');

% analyze results
active_ucb_path_diff= active_points_ucb(2:end,:) - active_points_ucb(1:(end-1),:);
active_ucb_path_norm = sqrt(sum(active_ucb_path_diff.^2, 2));
active_ucb_path_length = sum(active_ucb_path_norm);

active_path_diff= active_points_path(2:end,:) - active_points_path(1:(end-1),:);
active_path_norm = sqrt(sum(active_path_diff.^2, 2));
active_path_length = sum(active_path_norm);

% get misclassification results
lse_fp = sum(sum(lse_above_grid & f_below_grid));
lse_tp = sum(sum(lse_above_grid & f_above_grid));
lse_fn = sum(sum(lse_below_grid & f_above_grid));
lse_tn = sum(sum(lse_below_grid & f_below_grid));
lse_ukn = sum(sum(~lse_above_grid & ~lse_below_grid));
lse_precision = lse_tp / (lse_tp + lse_fp);
lse_recall = lse_tp / (lse_tp + lse_fn);
lse_F1 = 2 * lse_precision * lse_recall / (lse_precision + lse_recall);

path_fp = sum(sum(path_above_grid & f_below_grid));
path_tp = sum(sum(path_above_grid & f_above_grid));
path_fn = sum(sum(path_below_grid & f_above_grid));
path_tn = sum(sum(path_below_grid & f_below_grid));
path_ukn = sum(sum(~path_above_grid & ~path_below_grid));
path_precision = path_tp / (path_tp + path_fp);
path_recall = path_tp / (path_tp + path_fn);
path_F1 = 2 * path_precision * path_recall / (path_precision + path_recall);

% figure(6);
% plot(

fprintf('PATH LENGTHS\n');
fprintf('LSE:\t\t%f\n', active_ucb_path_length);
fprintf('Path:\t\t%f\n', active_path_length);

fprintf('FALSE POSITIVES\n');
fprintf('LSE:\t\t%f\n', lse_fp);
fprintf('Path:\t\t%f\n', path_fp);

fprintf('FALSE NEGATIVES\n');
fprintf('LSE:\t\t%f\n', lse_fn);
fprintf('Path:\t\t%f\n', path_fn);

fprintf('UNCLASSIFIED\n');
fprintf('LSE:\t\t%f\n', lse_ukn);
fprintf('Path:\t\t%f\n', path_ukn);


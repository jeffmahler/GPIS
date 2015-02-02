% try out some path gp ucb stuff
config = struct();
% params of grid
config.height = 50;
config.width = 50;
num_points = config.height * config.width;
[X, Y] = meshgrid(1:config.width, 1:config.height);
config.points = [X(:), Y(:)];
grid_center = [width / 2, height / 2];
config.sdf_trunc = 4.0;

% params of the gp
config.sigma_kernel = 4.5;
config.kernel_scale = 0.1;
config.sigma_noise = 0.1;
config.mean_func = [];
config.cov_func = @covSEiso;
config.lik_func = @likGauss;

% algor params
config.num_iters = 250;
config.path_penalty = 1e-1;
config.beam_size = 1;
config.horizon = 5;
config.use_dec_path_penalty = 1;

% pose prior
config.num_pose_particles = 100;
config.sigma_resample = 1;
config.min_scale = 0.5;
config.max_scale = 1.5;

% results
config.store_classification_images = 1;
config.class_res_rate = 10;
config.class_image_rate = 10;
config.tp_len = 10;
config.vis_path = 1;
config.vis_class_im = 0;

% delta
config.delta = 0.99;
config.h = 0; % explicit level we are trying to estimate
config.tol = 1e-2;
config.f_rkhs_norm = 1;

%% sample function
f_mu = zeros(num_points, 1);
f_Sigma = feval(config.cov_func, ...
    [log(config.sigma_kernel), log(config.kernel_scale)], config.points);
f_Sigma = 1e-10 + (f_Sigma + f_Sigma') / 2; 

% sample a function from the gp
f = mvnrnd(f_mu, f_Sigma);
f_grid = reshape(f, [config.height, config.width]);

%% display sampled function
height = config.height;
width = config.width;
figure(1);
f_colors = zeros(height, width, 3);
f_colors(:,:,1) = ones(height, width);
f_colors(:,:,2) = ones(height, width);
surf(X, Y, f_grid, f_colors);
set(gca, 'FontSize', 15);
xlabel('X Position');
ylabel('Y Position');
zlabel('Value');
hold on;
level_set = zeros(height, width);
ls_colors = 0.5*ones(height, width, 3);
surf(level_set, ls_colors);
view(55,45);

%% warping
yaw = 0;  
pitch = pi/8;  
roll = 0;
dcm = angle2dcm( pitch, roll, yaw);
R = dcm(1:2, 1:2);

tf = struct();
tf.t = [0; 0];
tf.R = R;
tf.s = 0.9;

%%
f_warped = warp_mean_function(tf, f_grid, grid_center);

figure(10);
subplot(1,2,1);
imagesc(f_grid);
subplot(1,2,2);
imagesc(f_warped);

%% try it out
config.mean_func = tsdf;
config.num_pose_particles = 100;
config.r_sigma_resample = 0.01;
config.t_sigma_resample = 0.1;
config.s_sigma_resample = 0.001;
config.min_rot = 0;
config.max_rot = 2 * pi;
config.min_trans = [-config.height / 8, -config.width / 8];
config.max_trans = [config.height / 8, config.width / 8];
config.min_scale = 0.8;
config.max_scale = 1.0;
config.resample_size = config.num_pose_particles;
config.kernel_resample = 0;
config.gaussian_resample = 1;
config.num_f_samples = 100;
config.true_tf = tf;
config.vis_path = 0;
config.kernel_scale = 1;

config.path_penalty = 1e-1;
config.use_dec_path_penalty = 1;
config.class_image_rate = 1;

%% user defined bar function
f_Sigma = feval(config.cov_func, ...
    [log(config.sigma_kernel), log(config.kernel_scale)], config.points);
f_Sigma = 1e-10 + (f_Sigma + f_Sigma') / 2; 

grid = zeros(config.height, config.width);
grid(5:end-5, (config.width/2 - 5):(config.width/2 + 5)) = 1;
tsdf = trunc_signed_distance(grid, config.sdf_trunc);
mu_grid = warp_mean_function(tf, tsdf, grid_center);

f = mvnrnd(mu_grid(:), f_Sigma);
f_grid = reshape(f, [config.height, config.width]);
sdf_surface(f_grid);

%%
rng(100);
[pclass_results, ppath_lengths, plosses, pclass_images, piter_times] = ...
    discrete_gp_lse_pose_prior(f_grid, config);
%% run others with zero mean
config.mean_func = [];
[rclass_results, rpath_lengths, rlosses, rclass_images] = ...
    discrete_gp_ls_random(f_grid, config);
[sclass_results, spath_lengths, slosses, sclass_images] = ...
    discrete_gp_ls_subsample(f_grid, config);
[lclass_results, lpath_lengths, llosses, lclass_images] = ...
    discrete_gp_lse(f_grid, config);

%%
true_above = f > config.h;
true_below = f < config.h;
f_above_grid = reshape(true_above, [config.height, config.width]);
f_below_grid = reshape(true_below, [config.height, config.width]);
truth_im = zeros(config.height, config.width, 3);
truth_im(:,:,1) = f_above_grid; % red above
truth_im(:,:,2) = f_below_grid; % green below

figure(5);
subplot(1,5,1);
imshow(rclass_images{end});
subplot(1,5,2);
imshow(sclass_images{end});
subplot(1,5,3);
imshow(lclass_images{end});
subplot(1,5,4);
imshow(pclass_images{end});
subplot(1,5,5);
imshow(truth_im);
%% test discrete gp lse
rng(100);
% [rclass_results, rpath_lengths, rlosses, rclass_images] = discrete_gp_ls_random(f_grid, config);
% [sclass_results, spath_lengths, slosses, sclass_images] = discrete_gp_ls_subsample(f_grid, config);
% [lclass_results, lpath_lengths, llosses, lclass_images] = discrete_gp_lse(f_grid, config);

%% display class images
% get ground truth values
true_above = f > config.h;
true_below = f < config.h;
f_above_grid = reshape(true_above, [config.height, config.width]);
f_below_grid = reshape(true_below, [config.height, config.width]);
truth_im = zeros(config.height, config.width, 3);
truth_im(:,:,1) = f_above_grid; % red above
truth_im(:,:,2) = f_below_grid; % green below

figure(5);
subplot(1,4,1);
imshow(rclass_images{end});
subplot(1,4,2);
imshow(sclass_images{end});
subplot(1,4,3);
imshow(lclass_images{end});
subplot(1,4,4);
imshow(truth_im);

%% plot results
close all;
num_to_plot = size(rclass_results, 2);
rukns = [];
rF1s = [];
rAcc = [];
riters = [];
sukns = [];
sF1s = [];
sAcc = [];
siters = [];
lukns = [];
lF1s = [];
lAcc = [];
liters = [];
pukns = [];
pF1s = [];
pAcc = [];
piters = [];
for i = 1:num_to_plot
    cr = rclass_results{i};
    rukns = [rukns, cr.ukn_rate];
    rF1s = [rF1s, cr.F1];
    rAcc = [rAcc, (cr.tp + cr.tn) / (cr.tp + cr.tn + cr.fp + cr.fn + cr.ukn)];
    riters = [riters, cr.iteration];
    
    cr = sclass_results{i};
    sukns = [sukns, cr.ukn_rate];
    sF1s = [sF1s, cr.F1];
    sAcc = [sAcc, (cr.tp + cr.tn) / (cr.tp + cr.tn + cr.fp + cr.fn + cr.ukn)];
    siters = [siters, cr.iteration];
    
    cr = lclass_results{i};
    lukns = [lukns, cr.ukn_rate];
    lF1s = [lF1s, cr.F1];
    lAcc = [lAcc, (cr.tp + cr.tn) / (cr.tp + cr.tn + cr.fp + cr.fn + cr.ukn)];
    liters = [liters, cr.iteration];
    
    cr = pclass_results{i};
    pukns = [pukns, cr.ukn_rate];
    pF1s = [pF1s, cr.F1];
    pAcc = [pAcc, (cr.tp + cr.tn) / (cr.tp + cr.tn + cr.fp + cr.fn + cr.ukn)];
    piters = [piters, cr.iteration];
end

figure(1);
set(gca, 'FontSize', 15);
plot(riters, rukns, 'r', 'LineWidth', 2);
hold on;
plot(siters, sukns, 'g', 'LineWidth', 2);
plot(liters, lukns, 'b', 'LineWidth', 2);
plot(piters, pukns, 'c', 'LineWidth', 2);
legend('Random', 'Subsample', 'LSE', 'Pose Filter', 'Location', 'Best');
xlabel('Iteration');
ylabel('Percent Unclassified');
title('Unclassified Rate vs Iteration');

figure(2);
set(gca, 'FontSize', 15);
plot(riters, rF1s, 'r', 'LineWidth', 2);
hold on;
plot(siters, sF1s, 'g', 'LineWidth', 2);
plot(liters, lF1s, 'b', 'LineWidth', 2);
plot(piters, pF1s, 'c', 'LineWidth', 2);
legend('Random', 'Subsample', 'LSE', 'Pose Filter', 'Location', 'Best');
xlabel('Iteration');
ylabel('F1 Score');
title('F1 Score vs Iteration');

figure(3);
set(gca, 'FontSize', 15);
plot(rpath_lengths, 'r', 'LineWidth', 2);
hold on;
plot(spath_lengths, 'g', 'LineWidth', 2);
plot(lpath_lengths, 'b', 'LineWidth', 2);
plot(ppath_lengths, 'c', 'LineWidth', 2);
legend('Random', 'Subsample', 'LSE', 'Pose Filter', 'Location', 'Best');
xlabel('Iteration');
ylabel('Path Length');
title('Path Length vs Iteration');

figure(4);
set(gca, 'FontSize', 15);
plot(riters, rAcc, 'r', 'LineWidth', 2);
hold on;
plot(siters, sAcc, 'g', 'LineWidth', 2);
plot(liters, lAcc, 'b', 'LineWidth', 2);
plot(piters, pAcc, 'c', 'LineWidth', 2);
legend('Random', 'Subsample', 'LSE', 'Pose Filter', 'Location', 'Best');
xlabel('Iteration');
ylabel('Class Accuracy');
title('Classification Accuracy vs Iteration');

figure(5);
set(gca, 'FontSize', 15);
plot(cumsum(rlosses), 'r', 'LineWidth', 2);
hold on;
plot(cumsum(slosses), 'g', 'LineWidth', 2);
plot(cumsum(llosses), 'b', 'LineWidth', 2);
plot(cumsum(plosses), 'c', 'LineWidth', 2);
plot(50*sqrt(1:config.num_iters), 'k', 'LineWidth', 2);
legend('Random', 'Subsample', 'LSE', 'Pose Filter', 'Theoretical', 'Location', 'Best');
xlabel('Iteration');
ylabel('Loss');
title('Loss vs Iteration');


% Set up config and run bandit comparison experiment
% get random shape indices
num_test_shapes = 500;
rng(100);
shape_indices = round(8600 * rand(num_test_shapes, 1) + 1);
shape_indices = [326];

config = struct();
config.arrow_length = 10;
config.scale = 1.0;
config.friction_coef = 0.5;
config.plate_width = 3;
config.grip_scale = 0.4;
config.padding = 5;
config.tsdf_thresh = 10;
config.downsample = 4;
config.snapshot_iter = 1;

config.test_data_file = 'data/caltech/caltech101_silhouettes_28.mat';
config.mat_save_dir = 'data/grasp_transfer_models/test';
config.model_dir = 'data/grasp_transfer_models/brown_dataset';

config.filter_win = 7;
config.filter_sigma = sqrt(2)^2;
config.vis_std = false;

config.quad_min_dim = 2;
config.quad_max_dim = 128;

config.num_shape_samples = 100;
config.image_scale = 4.0;

config.knn = 9;
config.vis_knn = true;
config.num_grasp_samples = 1500;
config.num_knn_grasps = 3;

config.vis_bandits = false;
config.method_names = {'ucb_results', 'thompson_results', 'gittins_results'};
config.num_bins = 100;

%% noise parameters
noise_params = struct();
noise_params.y_thresh1_low = 79;
noise_params.y_thresh1_high = 79;
noise_params.x_thresh1_low = 79;
noise_params.x_thresh1_high = 79;

noise_params.y_thresh2_low = 79;
noise_params.y_thresh2_high = 79;
noise_params.x_thresh2_low = 79;
noise_params.x_thresh2_high = 79;

noise_params.y_thresh3_low = 79;
noise_params.y_thresh3_high = 79;
noise_params.x_thresh3_low = 79;
noise_params.x_thresh3_high = 79;

noise_params.occ_y_thresh1_low = 79;
noise_params.occ_y_thresh1_high = 79;
noise_params.occ_x_thresh1_low = 79;
noise_params.occ_x_thresh1_high = 79;

noise_params.occ_y_thresh2_low = 79;
noise_params.occ_y_thresh2_high = 79;
noise_params.occ_x_thresh2_low = 79;
noise_params.occ_x_thresh2_high = 79;

noise_params.transp_y_thresh1_low = 79;
noise_params.transp_y_thresh1_high = 79;
noise_params.transp_x_thresh1_low = 79;
noise_params.transp_x_thresh1_high = 79;

noise_params.transp_y_thresh2_low = 79;
noise_params.transp_y_thresh2_high = 79;
noise_params.transp_x_thresh2_low = 79;
noise_params.transp_x_thresh2_high = 79;

noise_params.occlusionScale = 1000;
noise_params.transpScale = 4.0;
noise_params.noiseScale = 0.2;
noise_params.interiorRate = 0.1;
noise_params.specularNoise = true;
noise_params.sparsityRate = 0.2;
noise_params.sparseScaling = 1000;
noise_params.edgeWin = 2;

noise_params.noiseGradMode = 'None';
noise_params.horizScale = 1;
noise_params.vertScale = 1;

config.noise_params = noise_params;

%% shape construction parameters
construction_params = struct();
construction_params.activeSetMethod = 'Full';
construction_params.activeSetSize = 1;
construction_params.beta = 10;
construction_params.firstIndex = 150;
construction_params.numIters = 0;
construction_params.eps = 1e-2;
construction_params.delta = 1e-2;
construction_params.levelSet = 0;
construction_params.surfaceThresh = 0.1;
construction_params.scale = 1.0;
construction_params.numSamples = 20;
construction_params.trainHyp = false;
construction_params.hyp = struct();
construction_params.hyp.cov = [log(exp(2)), log(1)];
construction_params.hyp.mean = [0; 0; 0];
construction_params.hyp.lik = log(0.1);
construction_params.useGradients = true;
construction_params.downsample = 2;

config.construction_params = construction_params;

%% run experiment
bandit_comparison_results2 = compare_bandits_for_grasp_transfer(shape_indices, config);




%%
save('results/bandits/bandit_comparison_results.mat', 'bandit_comparison_results');

%% accumulate results
regret_analysis = analyze_final_regret(bandit_comparison_results,...
                                       config.method_names);
save('results/bandits/regret_analysis.mat', 'regret_analysis');

%% average cumulative regret
avg_ucb_cum_regret = mean(cell2mat(regret_analysis{1}.cumulative_regret'), 2);
avg_thomp_cum_regret = mean(cell2mat(regret_analysis{2}.cumulative_regret'), 2);
avg_git_cum_regret = mean(cell2mat(regret_analysis{3}.cumulative_regret'), 2);

std_ucb_cum_regret = std(cell2mat(regret_analysis{1}.cumulative_regret'), 1, 2);
std_thomp_cum_regret = std(cell2mat(regret_analysis{2}.cumulative_regret'), 1, 2);
std_git_cum_regret = std(cell2mat(regret_analysis{3}.cumulative_regret'), 1, 2);

figure(4);
clf;
%errorbar(avg_ucb_cum_regret(1:100:end), std_ucb_cum_regret(1:100:end), 'r', 'LineWidth', 2);
plot(avg_ucb_cum_regret, 'r', 'LineWidth', 2);
hold on;
plot(avg_thomp_cum_regret, 'g', 'LineWidth', 2);
plot(avg_git_cum_regret, 'b', 'LineWidth', 2);
%errorbar(avg_thomp_cum_regret(1:100:end), std_thomp_cum_regret(1:100:end), 'g', 'LineWidth', 2);
%errorbar(avg_git_cum_regret, std_git_cum_regret, 'b', 'LineWidth', 2);
legend('UCB', 'Thompson', 'Gittins', 'Location', 'Best');
xlabel('Iterations', 'FontSize', 15);
ylabel('Cumulative Regret', 'FontSize', 15);
title('Average Cumulative Regret', 'FontSize', 15);

%% simple regret
avg_ucb_simp_regret = mean(cell2mat(regret_analysis{1}.simple_regret'), 2);
avg_thomp_simp_regret = mean(cell2mat(regret_analysis{2}.simple_regret'), 2);
avg_git_simp_regret = mean(cell2mat(regret_analysis{3}.simple_regret'), 2);

figure(5);
clf;
plot((avg_ucb_simp_regret), 'r', 'LineWidth', 2);
hold on;
plot((avg_thomp_simp_regret), 'g', 'LineWidth', 2);
plot((avg_git_simp_regret), 'b', 'LineWidth', 2);
legend('UCB', 'Thompson', 'Gittins', 'Location', 'Best');
xlabel('Iterations', 'FontSize', 15);
ylabel('Simple Regret', 'FontSize', 15);
title('Average Simple Regret', 'FontSize', 15);
% xlim([0, 200]);
% ylim([0, 0.1]);

%% time to find optima
figure(6);
subplot(1,3,1);
hist(regret_analysis{1}.time_to_optimal(regret_analysis{1}.time_to_optimal > 0), ...
     config.num_bins);
title('UCB Time to Optimal');
xlim([0,2100]);
ylim([0,200]);

subplot(1,3,2);
hist(regret_analysis{2}.time_to_optimal(regret_analysis{2}.time_to_optimal > 0), ...
     config.num_bins);
title('Thompson Time to Optimal');
xlim([0,2100]);
ylim([0,200]);

subplot(1,3,3);
hist(regret_analysis{3}.time_to_optimal(regret_analysis{3}.time_to_optimal > 0), ...
     config.num_bins);
title('Gittins Time to Optimal');
xlim([0,2100]);
ylim([0,200]);

%% plot the allocated pulls per grasps
figure(7);

clf;
plot(regret_analysis{1}.pulls_per_grasp, 'r', 'LineWidth', 2);
hold on;
plot(regret_analysis{2}.pulls_per_grasp, 'g', 'LineWidth', 2);
%plot(regret_analysis{3}.pulls_per_grasp, 'b', 'LineWidth', 2);
legend('UCB', 'Thompson', 'Location', 'Best');
xlabel('Grasp Ranking', 'FontSize', 15);
ylabel('Pulls Per Grasp', 'FontSize', 15);
title('Grasp Ranking', 'FontSize', 15);

%% histograms of final regret
figure(7);

xbins = 0:0.00005:0.005;

clf;
subplot(1,2,1);
hist(regret_analysis{1}.final_regret, xbins);
title('UCB');
xlim([xbins(1), xbins(end)]);
xlabel('Final Simple Regret', 'FontSize', 15);
ylabel('# Occurences', 'FontSize', 15);

subplot(1,2,2);
hist(regret_analysis{2}.final_regret, xbins);
title('Thompson');
xlim([xbins(1), xbins(end)]);
xlabel('Final Simple Regret', 'FontSize', 15);
ylabel('# Occurences', 'FontSize', 15);

%% regret with error bars
figure(8);
clf;
errorbar(avg_ucb_cum_regret(1:100:end), std_ucb_cum_regret(1:100:end), 'r', 'LineWidth', 2);
%plot(avg_ucb_cum_regret, 'r', 'LineWidth', 2);
hold on;
%plot(avg_thomp_cum_regret, 'g', 'LineWidth', 2);
%plot(avg_git_cum_regret, 'b', 'LineWidth', 2);
errorbar(avg_thomp_cum_regret(1:100:end), std_thomp_cum_regret(1:100:end), 'g', 'LineWidth', 2);
%errorbar(avg_git_cum_regret, std_git_cum_regret, 'b', 'LineWidth', 2);
legend('UCB', 'Thompson', 'Location', 'Best');
xlabel('Iterations', 'FontSize', 15);
ylabel('Cumulative Regret', 'FontSize', 15);
title('Average Cumulative Regret', 'FontSize', 15);



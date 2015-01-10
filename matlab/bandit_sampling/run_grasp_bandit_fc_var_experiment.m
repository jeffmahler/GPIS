% Set up config and run bandit comparison experiment
% get random shape indices
close all; 
%clear all; 
num_test_shapes = 3;
rng(100);
shape_indices = round(8600 * rand(num_test_shapes, 1) + 1);
%shape_indices = [326];

config = struct();
config.num_shapes = num_test_shapes;
config.num_iters = 8; 
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
config.num_grasp_samples = 1000;
config.num_knn_grasps = 5;
config.num_grasps = 1000; 
config.gittins_in_filename1 = 'matlab/bandit_sampling/indices';
config.gittins_out_filename1 = 'matlab/bandit_sampling/gittins_indices';


%Noise Parameters

config.fric_var = 0.05; 
p_uncertain = eye(3); 
p_uncertain(3,3) = 0.03; 
config.pose_var = p_uncertain; 

config.vis_bandits = false;
config.method_names = ...
    {'random', 'ucb', ...
     'bayes_ucbs', 'thompson', 'gittins98'};
config.num_bins = 100;

%% noise parameters
noise_params = struct();
noise_params.y_thresh1_low = 30;
noise_params.y_thresh1_high = 79;
noise_params.x_thresh1_low = 30;
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
noise_params.noiseScale = 0.4;
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
results_fric_var = cell(config.num_iters, 1); 
fric_var_orig = config.fric_var; 
for i = 1:config.num_iters
    config.fric_var = i*fric_var_orig; 
    bandit_comparison_results = compare_bandits(shape_indices, config);
    results_fric_var{i} = bandit_comparison_results; 
    

    %% accumulate results
    regret_analysis = analyze_final_regret(bandit_comparison_results,...
                                   config.method_names,config);
    save('results/bandits/regret_analysis_fc.mat', 'regret_analysis');

    %% average cumulative regret
    avg_random_cum_regret = mean(cell2mat(regret_analysis{1}.cumulative_regret'), 2);
    avg_ucb_cum_regret = mean(cell2mat(regret_analysis{2}.cumulative_regret'), 2);
    avg_bucb_cum_regret = mean(cell2mat(regret_analysis{3}.cumulative_regret'), 2);
    avg_thomp_cum_regret = mean(cell2mat(regret_analysis{4}.cumulative_regret'), 2);
    avg_git98_cum_regret = mean(cell2mat(regret_analysis{5}.cumulative_regret'), 2);

    % std_ucb_cum_regret = std(cell2mat(regret_analysis{2}.cumulative_regret'), 1, 2);
    % std_bucb_cum_regret = std(cell2mat(regret_analysis{2}.cumulative_regret'), 1, 2);
    % std_thomp_cum_regret = std(cell2mat(regret_analysis{3}.cumulative_regret'), 1, 2);
    % std_git_cum_regret = std(cell2mat(regret_analysis{4}.cumulative_regret'), 1, 2);

    figure(4*i);
    clf;
    %errorbar(avg_ucb_cum_regret(1:100:end), std_ucb_cum_regret(1:100:end), 'r', 'LineWidth', 2);
    plot(avg_random_cum_regret, 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2);
    hold on;

    plot(avg_ucb_cum_regret, 'r', 'LineWidth', 2);
    plot(avg_bucb_cum_regret, 'c', 'LineWidth', 2);
    plot(avg_thomp_cum_regret, 'g', 'LineWidth', 2);
    plot(avg_git98_cum_regret, 'm', 'LineWidth', 2);
    legend('Random', 'UCB', ...
    'Bayes UCB', 'Thompson', ...
    'Gittins (gamma = 0.98)', 'Location', 'Best');
    xlabel('Iterations', 'FontSize', 15);
    ylabel('Cumulative Regret', 'FontSize', 15);
    title('Average Cumulative Regret', 'FontSize', 15);

    %% simple regret
    avg_random_simp_regret = mean(cell2mat(regret_analysis{1}.simple_regret'), 2);
    avg_ucb_simp_regret = mean(cell2mat(regret_analysis{2}.simple_regret'), 2);
    avg_bucb_simp_regret = mean(cell2mat(regret_analysis{3}.simple_regret'), 2);
    avg_thomp_simp_regret = mean(cell2mat(regret_analysis{4}.simple_regret'), 2);
    avg_git98_simp_regret = mean(cell2mat(regret_analysis{5}.simple_regret'), 2);

    %Padding 
    avg_ucb_simp = zeros(size(avg_random_simp_regret)); 
    avg_thom_simp = zeros(size(avg_random_simp_regret));
    avg_git_simp = zeros(size(avg_random_simp_regret)); 

    avg_ucb_simp(1:size(avg_ucb_simp_regret,1),1) = avg_ucb_simp_regret;
    avg_thom_simp(1:size(avg_thomp_simp_regret,1),1) = avg_thomp_simp_regret;
    avg_git_simp(1:size(avg_git98_simp_regret,1),1) = avg_git98_simp_regret;

    figure(5*i);
    clf;
    plot(avg_random_simp_regret, 'Color', [0.5, 0.5, 0.5], 'LineWidth', 3);
    hold on;
    plot(avg_ucb_simp, 'r', 'LineWidth', 3);
    plot(avg_thom_simp, 'g', 'LineWidth', 3);
    plot(avg_git_simp, 'm', 'LineWidth', 3);
    [hleg1, hobj1] = legend('Random', 'UCB', 'Thompson', ...
    'Gittins', 'Location', 'Best');
    textobj = findobj(hobj1, 'type', 'text');
    set(textobj, 'Interpreter', 'latex', 'fontsize', 18);
    xlabel('Iterations', 'FontSize', 15);
    ylabel('Simple Regret', 'FontSize', 15);
    title('Average Simple Regret', 'FontSize', 15);
    axis([1000 size(avg_random_simp_regret,1) 0 1.0]); 
    % xlim([0, 200]);
    % ylim([0, 0.1]);

    %% time to find optima
    % figure(6);
    % subplot(1,3,1);
    % hist(regret_analysis{1}.time_to_optimal(regret_analysis{1}.time_to_optimal > 0), ...
    %      config.num_bins);
    % title('UCB Time to Optimal');
    % xlim([0,2100]);
    % ylim([0,200]);
    % 
    % subplot(1,3,2);
    % hist(regret_analysis{2}.time_to_optimal(regret_analysis{2}.time_to_optimal > 0), ...
    %      config.num_bins);
    % title('Thompson Time to Optimal');
    % xlim([0,2100]);
    % ylim([0,200]);
    % 
    % subplot(1,3,3);
    % hist(regret_analysis{3}.time_to_optimal(regret_analysis{3}.time_to_optimal > 0), ...
    %      config.num_bins);
    % title('Gittins Time to Optimal');
    % xlim([0,2100]);
    % ylim([0,200]);
    % 
    %% plot the allocated pulls per grasps
    figure(7*i);

    clf;
    plot(regret_analysis{1}.pulls_per_grasp, 'Color', [0.5, 0.5, 0.5], 'LineWidth', 3);
    hold on;
    plot(regret_analysis{2}.pulls_per_grasp, 'Color', [1, 0.5, 0.75], 'LineWidth', 3);
    %plot(regret_analysis{3}.pulls_per_grasp, 'Color', [0.5, 0.5, 1], 'LineWidth', 3);
    plot(regret_analysis{4}.pulls_per_grasp, 'r', 'LineWidth', 3);
    plot(regret_analysis{5}.pulls_per_grasp, 'c', 'LineWidth', 3);

    [hleg1, hobj1] = legend('Random', 'UCB', ...
    'Thompson', ...
    'Gittins', 'Location', 'Best');
    textobj = findobj(hobj1, 'type', 'text');
    set(textobj, 'Interpreter', 'latex', 'fontsize', 18);
    xlabel('Grasp Ranking', 'FontSize', 18);
    ylabel('Pulls Per Grasp', 'FontSize', 18);
    title('Grasp Ranking', 'FontSize', 18);

    % %% histograms of final regret
    % figure(8);
    % 
    % xbins = 0:0.00005:0.005;
    % 
    % clf;
    % subplot(1,2,1);
    % hist(regret_analysis{1}.final_regret, xbins);
    % title('UCB');
    % xlim([xbins(1), xbins(end)]);
    % xlabel('Final Simple Regret', 'FontSize', 15);
    % ylabel('# Occurences', 'FontSize', 15);
    % 
    % subplot(1,2,2);
    % hist(regret_analysis{2}.final_regret, xbins);
    % title('Thompson');
    % xlim([xbins(1), xbins(end)]);
    % xlabel('Final Simple Regret', 'FontSize', 15);
    % ylabel('# Occurences', 'FontSize', 15);
    % 
    % %% regret with error bars
    % figure(9);
    % clf;
    % errorbar(avg_ucb_cum_regret(1:100:end), std_ucb_cum_regret(1:100:end), 'r', 'LineWidth', 2);
    % %plot(avg_ucb_cum_regret, 'r', 'LineWidth', 2);
    % hold on;
    % %plot(avg_thomp_cum_regret, 'g', 'LineWidth', 2);
    % %plot(avg_git_cum_regret, 'b', 'LineWidth', 2);
    % errorbar(avg_thomp_cum_regret(1:100:end), std_thomp_cum_regret(1:100:end), 'g', 'LineWidth', 2);
    % %errorbar(avg_git_cum_regret, std_git_cum_regret, 'b', 'LineWidth', 2);
    % legend('UCB', 'Thompson', 'Location', 'Best');
    % xlabel('Iterations', 'FontSize', 15);
    % ylabel('Cumulative Regret', 'FontSize', 15);
    % title('Average Cumulative Regret', 'FontSize', 15);
    % 
    % 
    %% plot a histogram of the grasp samples
    pfc = [];
    for t = 1:size(bandit_comparison_results,2)
        pfc = [pfc; bandit_comparison_results{t}.grasp_values(:,3)];
    end

    n_bins = 100;
    figure(9*i);
    hist(pfc, n_bins);
    xlabel('Probability of Force Closure');
    ylabel('Bin Counts');

end
%%

degs = []; 
for i=1:config.num_iters
    degs = [degs config.fric_var*i]; 
end

regret_sensitive = analyze_final_regret_sensitive(results_fric_var,...
                                   config.method_names,config,degs);
                            
%% covg_plot

avg_random_simp_convg = regret_sensitive{1}.var_time;
avg_ucb_simp_convg =  regret_sensitive{2}.var_time;
avg_bucb_simp_convg =  regret_sensitive{3}.var_time;
avg_thomp_simp_convg =  regret_sensitive{4}.var_time;
avg_git_simp_convg =  regret_sensitive{5}.var_time;
                               
figure(11*i);
clf;
hold on; 
%plot(avg_random_simp_convg, 'b', 'LineWidth', 3);
%plot(avg_ucb_simp_convg, 'r', 'LineWidth', 3);
plot(degs,avg_thomp_simp_convg, 'g', 'LineWidth', 3);
plot(degs,avg_git_simp_convg, 'm', 'LineWidth', 3);
[hleg1, hobj1] = legend('Thompson', ...
'Gittins','Location', 'Best');
textobj = findobj(hobj1, 'type', 'text');
set(textobj, 'Interpreter', 'latex', 'fontsize', 18);
xlabel('Friction Variance', 'FontSize', 15);
ylabel('Grasp Samples', 'FontSize', 15);
title('Sensitivity Analysis of Friction Coefficient', 'FontSize', 15);
    %axis([1000 size(avg_random_simp_regret,1) 0 1.0]);   
                               
hold off; 
save('results/bandits/results_fric_var.mat', 'results_fric_var','-v7.3');

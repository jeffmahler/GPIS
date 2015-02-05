function [avg_unclass_rates, avg_F1_scores, avg_path_lengths, labels] = ...
    analyze_lse_pose_results(experiment_results, experiment_config)
%ANALYZE_LSE_RESULTS Summary of this function goes here
%   Detailed explanation goes here
num_function_samples = size(experiment_results, 2);

% compute averages over all functions
% setup horizons
num_horizons = size(experiment_config.horizons, 2);
num_path_penalties = size(experiment_config.path_penalties, 2);
num_beams = size(experiment_config.beam_sizes, 2);
num_dpp = size(experiment_config.dec_pps, 2);
num_methods = num_horizons * num_path_penalties * num_beams * num_dpp + 2;

num_checkpoints = size(experiment_results{1,1}.class_results, 2);
num_iters = size(experiment_results{1,1}.path_lengths, 1);
avg_unclass_rates = zeros(num_methods, num_checkpoints);
avg_F1_scores = zeros(num_methods, num_checkpoints);
avg_path_lengths = zeros(num_methods, num_iters);
iterations = zeros(num_methods, num_checkpoints);

labels = cell(4, 1);
labels{1} = 'Random';
labels{2} = 'Subsample';

for k = 1:num_function_samples
    cur_index = 3;
    
    % accumulate results for random and average
    for j = 1:num_checkpoints
        avg_unclass_rates(1,j) = avg_unclass_rates(1,j) + experiment_results{1,k}.class_results{j}.ukn_rate;
        avg_F1_scores(1,j) = avg_F1_scores(1,j) + experiment_results{1,k}.class_results{j}.F1;
        if k == 1
            iterations(1,j) = iterations(1,j) + experiment_results{1,k}.class_results{j}.iteration;
        end
        
        avg_unclass_rates(2,j) = avg_unclass_rates(2,j) + experiment_results{2,k}.class_results{j}.ukn_rate;
        avg_F1_scores(2,j) = avg_F1_scores(2,j) + experiment_results{2,k}.class_results{j}.F1;
        if k == 1
            iterations(2,j) = iterations(2,j) + experiment_results{2,k}.class_results{j}.iteration;
        end
    end
    avg_path_lengths(1,:) = avg_path_lengths(1,:) + experiment_results{1,k}.path_lengths';
    avg_path_lengths(2,:) = avg_path_lengths(2,:) + experiment_results{2,k}.path_lengths';

    % loop through param variations
    for j = 1:num_checkpoints
        avg_unclass_rates(cur_index,j) = ...
            avg_unclass_rates(cur_index,j) + ...
            experiment_results{cur_index,k}.class_results{j}.ukn_rate;
        avg_F1_scores(cur_index,j) = ...
            avg_F1_scores(cur_index,j) + ...
            experiment_results{cur_index,k}.class_results{j}.F1;
        if k == 1
            iterations(cur_index,j) = ...
                iterations(cur_index,j) + ...
                experiment_results{cur_index,k}.class_results{j}.iteration;
        end
    end
    avg_path_lengths(cur_index,:) = ...
        avg_path_lengths(cur_index,:) + ...
        experiment_results{cur_index,k}.path_lengths';

    if k == 1
        labels{cur_index} = 'LSE';
    end
    cur_index = cur_index + 1;
    
    % get pose results
    if k == 4
        stop = 1;
    end
    
    num_pose_checks = size(experiment_results{cur_index,k}.class_results, 2);
    for j = 1:num_checkpoints
        if j <= num_pose_checks
            avg_unclass_rates(cur_index,j) = ...
                avg_unclass_rates(cur_index,j) + ...
                experiment_results{cur_index,k}.class_results{j}.ukn_rate;
            avg_F1_scores(cur_index,j) = ...
                avg_F1_scores(cur_index,j) + ...
                experiment_results{cur_index,k}.class_results{j}.F1;
            if k == 1
                iterations(cur_index,j) = ...
                    iterations(cur_index,j) + ...
                    experiment_results{cur_index,k}.class_results{j}.iteration;
            end
        else
            avg_unclass_rates(cur_index,j) = ...
                avg_unclass_rates(cur_index,j) + ...
                experiment_results{cur_index,k}.class_results{num_pose_checks}.ukn_rate;
            avg_F1_scores(cur_index,j) = ...
                avg_F1_scores(cur_index,j) + ...
                experiment_results{cur_index,k}.class_results{num_pose_checks}.F1;
            if k == 1
                iterations(cur_index,j) = ...
                    iterations(cur_index,j) + ...
                    experiment_results{cur_index,k}.class_results{num_pose_checks}.iteration;
            end
        end
    end
    avg_path_lengths(cur_index,:) = ...
        avg_path_lengths(cur_index,:) + ...
        experiment_results{cur_index,k}.path_lengths';

    if k == 1
        labels{cur_index} = 'LSE Pose';
    end
end

avg_unclass_rates = avg_unclass_rates / num_function_samples;
avg_F1_scores = avg_F1_scores / num_function_samples;
avg_path_lengths = avg_path_lengths / num_function_samples;

colors = distinguishable_colors(num_methods);

figure(1);
for i = 1:num_methods
    plot(iterations(i,:), avg_unclass_rates(i,:), 'Color', colors(i,:), 'LineWidth', 2);
    hold on;
end
h_legend = legend(labels{:}, 'Location', 'Best');
set(h_legend,'FontSize',10);
xlabel('Iteration', 'FontSize', 15);
ylabel('Percent Unclassified', 'FontSize', 15);
title('Unclassified Rate vs Iteration', 'FontSize', 15);
savefig(sprintf('%s/avg_unc_rates', experiment_config.output_dir));

figure(2);
set(gca, 'FontSize', 15);
for i = 1:num_methods
    plot(iterations(i,:), avg_F1_scores(i,:), 'Color', colors(i,:), 'LineWidth', 2);
    hold on;
end
h_legend = legend(labels{:}, 'Location', 'Best');
set(h_legend,'FontSize',10);
xlabel('Iteration', 'FontSize', 15);
ylabel('F1 Score', 'FontSize', 15);
title('F1 Score vs Iteration', 'FontSize', 15);
savefig(sprintf('%s/avg_f1_scores', experiment_config.output_dir));

figure(3);
set(gca, 'FontSize', 15);
for i = 1:num_methods
    plot(avg_path_lengths(i,:), 'Color', colors(i,:), 'LineWidth', 2);
    hold on;
end
h_legend = legend(labels{:}, 'Location', 'Best');
set(h_legend,'FontSize',10);
xlabel('Iteration', 'FontSize', 15);
ylabel('Path Length', 'FontSize', 15);
title('Path Length vs Iteration', 'FontSize', 15);
savefig(sprintf('%s/avg_path_lengths', experiment_config.output_dir));

end


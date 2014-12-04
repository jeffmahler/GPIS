function regret_results = ...
    analyze_final_regret(experiment_results, method_names, eps)
%COMPUTE_AVERAGE_REGRET Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    eps = 1e-3;
end

num_trials = size(experiment_results, 2);
num_methods = size(method_names, 2);
num_grasps = size(experiment_results{1}.grasp_candidates, 1);
regret_results = cell(num_methods, 1);

for i = 1:num_methods
    regret_results{i} = struct();
    regret_results{i}.final_regret = zeros(num_trials, 1);
    regret_results{i}.cumulative_regret = cell(num_trials, 1);
    regret_results{i}.simple_regret = cell(num_trials, 1);
    regret_results{i}.pulls_per_grasp = zeros(num_grasps, 1);
    regret_results{i}.time_to_optimal = -1 * ones(num_trials, 1);
    
    for j = 1:num_trials
        trial_results = experiment_results{j};
        method_results = getfield(trial_results, method_names{i});
        regret_results{i}.final_regret(j) = method_results.regret(end-1);
        regret_results{i}.cumulative_regret{j} = cumsum(method_results.regret);
        regret_results{i}.simple_regret{j} = method_results.regret;
        nonzero_regret_ind = find(method_results.regret > eps);
        if size(nonzero_regret_ind,1) > 0
            regret_results{i}.time_to_optimal(j) = nonzero_regret_ind(end);
        end
        
        % get the sorted arms and # pulls
        true_values = experiment_results{j}.grasp_values;
        [sortedX,sortingIndices] = sort(true_values(:,3),'descend');
        pulls_per_grasp = method_results.values(sortingIndices, 2);
        regret_results{i}.pulls_per_grasp = ...
            regret_results{i}.pulls_per_grasp + pulls_per_grasp;
    end
    regret_results{i}.pulls_per_grasp = ...
            regret_results{i}.pulls_per_grasp / num_trials;
    regret_results{i}.mean_final_regret   = mean(regret_results{i}.final_regret);
    regret_results{i}.median_final_regret = median(regret_results{i}.final_regret);
    regret_results{i}.min_final_regret    = min(regret_results{i}.final_regret);
    regret_results{i}.max_final_regret     = max(regret_results{i}.final_regret);
    regret_results{i}.max_final_regret     = max(regret_results{i}.final_regret);
    regret_results{i}.worst_cases = ...
        find(regret_results{i}.final_regret == regret_results{i}.max_final_regret);
end

end


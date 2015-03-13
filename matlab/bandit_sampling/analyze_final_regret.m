function regret_results = ...
    analyze_final_regret(experiment_results, method_names,config, eps)
%COMPUTE_AVERAGE_REGRET Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4
    eps = 1e-3;
end

num_trials = size(experiment_results, 2);
num_methods = size(method_names, 2);
num_grasps = config.num_grasps;
regret_results = cell(num_methods, 1);

for i = 1:num_methods
    regret_results{i} = struct();
    regret_results{i}.final_regret = zeros(num_trials, 1);
    regret_results{i}.cumulative_regret = cell(num_trials, 1);
    regret_results{i}.simple_regret = cell(num_trials, 1);
    regret_results{i}.pulls_per_grasp = zeros(num_grasps, 1);
    regret_results{i}.time_to_optimal = -1 * ones(num_trials, 1);
    regret_results{i}.pfc = cell(num_trials, 1);
    regret_results{i}.upper = cell(num_trials,1); 
    regret_results{i}.lower = cell(num_trials,1); 
    regret_results{i}.opt = cell(num_trials,1); 
    for j = 1:num_trials
        if(j ~= 9)
            trial_results = experiment_results{j};
            method_results = getfield(trial_results, method_names{i});
            regret_results{i}.final_regret(j) = method_results.regret(end-1);
            regret_results{i}.cumulative_regret{j} = cumsum(method_results.regret);
            regret_results{i}.simple_regret{j} = method_results.regret;
            [v,idx] = max(experiment_results{j}.grasp_values(:,3)); 
            
            
            regret_results{i}.opt{j} = zeros(size(method_results.regret))+v; 
            regret_results{i}.pfc{j} = v-method_results.regret;
            
%             idx_u = find(method_results.bounds(:,1) == 0); 
%             method_results.bounds(idx_u,1) = regret_results{i}.pfc{j}(idx_u);
%             
%             idx_l = find(method_results.bounds(:,2) == 0); 
%             method_results.bounds(idx_l,2) = regret_results{i}.pfc{j}(idx_l);
%             
%             
%             
%             upper = abs(method_results.bounds(:,1) - regret_results{i}.pfc{j});
%             lower = abs(regret_results{i}.pfc{j}- method_results.bounds(:,2));
%             
%             
%             zrs = [1:400:size(upper,1)]; 
%             
%             regret_results{i}.upper{j} = zeros(size(upper)); 
%             regret_results{i}.lower{j} = zeros(size(lower));
%             
%             regret_results{i}.upper{j}(zrs) = upper(zrs);
%             regret_results{i}.lower{j}(zrs) = lower(zrs);
            
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
    end
    regret_results{i}.pulls_per_grasp = ...
            regret_results{i}.pulls_per_grasp / num_trials;
    top_10 = sum(regret_results{i}.pulls_per_grasp(1:10))/sum(regret_results{i}.pulls_per_grasp)
    regret_results{i}.mean_final_regret   = mean(regret_results{i}.final_regret);
    regret_results{i}.median_final_regret = median(regret_results{i}.final_regret);
    regret_results{i}.min_final_regret    = min(regret_results{i}.final_regret);
    regret_results{i}.max_final_regret    = max(regret_results{i}.final_regret);
    regret_results{i}.max_final_regret    = max(regret_results{i}.final_regret);
    regret_results{i}.worst_cases = ...
        find(regret_results{i}.final_regret == regret_results{i}.max_final_regret);
end

end


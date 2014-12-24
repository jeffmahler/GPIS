function regret_results = ...
    analyze_final_regret_sensitive(experiment_results, method_names,config,degs, eps)
%COMPUTE_AVERAGE_REGRET Summary of this function goes here
%   Detailed explanation goes here

if nargin < 5
    eps = 1e-3;
end

num_trials = size(experiment_results, 2);
num_methods = size(method_names, 2);
num_iters = config.num_iters; 
num_grasps = config.num_grasps;
regret_results = cell(num_methods, 1);
for i = 1:num_methods
        regret_results{i} = struct();
        regret_results{i}.var_time = zeros(num_iters,1); 
end

figure; 


for t = 1:num_iters
    for i = 1:num_methods
       
     
        regret_results{i}.simple_regret = cell(num_trials, 1);
   
        for j = 1:num_trials
         
            trial_results = experiment_results{t}{j};
            method_results = getfield(trial_results, method_names{i});
           
            regret_results{i}.simple_regret{j} = method_results.regret;
        
        end
        
        mean_simple = mean(cell2mat(regret_results{i}.simple_regret'), 2);
        
        if(i==4)
            subplot(num_iters,1,t); 
            hold on; 
            if(t == 1)
               title('Sensitivity Analysis for Friction'); 
            end
            plot(mean_simple,color_struct(t)); 
            axis([1000 size(mean_simple,1) 0 1.0]); 
            ylabel(strcat('Var ',num2str(degs(t)))); 
        end
        idx = find(mean_simple <= eps); 
        
        if(size(idx,1) == 0)
            regret_results{i}.var_time(t) = size(mean_simple,1); 
        elseif(size(idx,1) == 1)
            regret_results{i}.var_time(t) = idx(1);
        else
            for k = 1:size(idx,1)-1
                test = find(mean_simple(idx(k+1):end) > eps);
                if(size(test,1) == 0)
                    break; 
                end
            end
            regret_results{i}.var_time(t) = idx(k); 
        end
         
           
        end
    end

end


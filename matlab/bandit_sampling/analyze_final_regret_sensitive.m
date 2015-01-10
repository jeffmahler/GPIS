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
num_shapes = config.num_shapes; 
regret_results = cell(num_methods, 1);
for i = 1:num_methods
        regret_results{i} = struct();
        regret_results{i}.var_time = zeros(num_iters,1); 
end


 
for i = 1:num_shapes
    top_grasp = {}; 
    Value = zeros(3,5); 
    count = 1;
    for t =1:3:num_iters
        shape_result = experiment_results{t}{i}; 
        [v,t_grasp] = max(shape_result.grasp_values(:,3)); 
        t_grasp 
        Value(count,:) = shape_result.grasp_values(t_grasp,:);
        
        top_grasp{count} = shape_result.grasp_samples{t_grasp}; 
        count = count + 1; 
    end
    img = shape_result.construction_results.newSurfaceImage; 
    shapeParams = shape_result.construction_results.predGrid; 
    visualize_value( Value,shapeParams,top_grasp,img);
end
figure; 
count = 1; 
for t = 1:num_iters
    for i = 1:num_methods
       
     
        regret_results{i}.simple_regret = cell(num_trials, 1);
   
        for j = 1:num_trials
         
            trial_results = experiment_results{t}{j};
            method_results = getfield(trial_results, method_names{i});
           
            regret_results{i}.simple_regret{j} = method_results.regret;
        
        end
        
        mean_simple = mean(cell2mat(regret_results{i}.simple_regret'), 2);
        
        if(i==5)
            mean_simple_git = mean(cell2mat(regret_results{i}.simple_regret'), 2);
            mean_simple_thom = mean(cell2mat(regret_results{i-1}.simple_regret'), 2);
            subplot(3,1,count); 
            hold on; 
            
            plot(mean_simple_git,'b', 'LineWidth', 3); 
            plot(mean_simple_thom,'r', 'LineWidth', 3); 
            if(count == 1)
               title('Sensitivity Analysis for Translation Variance in Pose','FontSize',18); 
               [hleg1, hobj1] = legend('Gittins', ...
                'Thompson', 'Location', 'Best');
                textobj = findobj(hobj1, 'type', 'text');
                set(textobj, 'Interpreter', 'latex', 'fontsize', 18);
            elseif(count == 3)
                xlabel('Iterations', 'FontSize', 18); 
            elseif(count == 2)
                ylabel('Simple Regret','FontSize',18);
            end
            axis([1000 size(mean_simple_thom,1) 0 1.0]); 
            %ylabel(strcat('Var ',num2str(degs(t)))); 
            count = count+1; 
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
tightfig;

end


function [ ] = visualize_value( Value,shapeParams,grasp_samples,surface_image)

N = size(grasp_samples, 2); 


figure;

 for i=1:N
     cp = grasp_samples{i}.cp;
     visualize_grasp(cp,shapeParams, surface_image, 4, 5,i,N,Value(i,3)); 
     %plot_grasp_arrows( surface_image, cp(1,:)', cp(3,:)', -(cp(1,:)-cp(2,:))', -(cp(3,:)-cp(4,:))', 4,4,i,N,Value(i,3))
 end
tightfig; 
end


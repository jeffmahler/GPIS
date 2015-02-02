function [ best_grasp, regret, Value ] = ...
    ucb(grasp_samples, num_grasps, shapeParams, experimentConfig, ...
        constructionResults, vis_bandits)
%THOMPSON_SAMPLING Summary of this function goes here
%   Detailed explanation goes here

    Total_Iters = 20000; 
    i = 1; 
    ts = true; 
    prune = false; 
    regret = zeros(Total_Iters+num_grasps,1); 

    for interval = 1:1
        Storage = {};
        Value = zeros(num_grasps,4); 
        t = 1;
        for i = 1:num_grasps
     
            [Q] = evaluate_grasp(i,grasp_samples,shapeParams,experimentConfig);

            UCB_part = 1; 
            Value(i,1) =  Q; 
            Value(i,2) = 1; 
            Value(i,3) = Value(i,1)/t; 
            Value(i,4) = UCB_part; 
            
            [v best_grasp] = max(Value(:,3));
            regret(t) =(interval-1)/interval*regret(t) + (1/interval)*compute_regret_pfc(best_grasp);
            t=t+1; 
        end


        i = 1;
        not_sat = true; 
         while(i<Total_Iters && not_sat)
            %i
            if(ts) 
                grasp = get_grasp(Value,t); 
            elseif(prune)
                np_grasp = not_pruned(Value); 
                grasp_idx = randi(length(np_grasp)); 
                grasp = np_grasp(grasp_idx); 
            else
                grasp = randi(num_grasps);  
            end

            [Q, grasp_samples] = evaluate_grasp(grasp,grasp_samples,shapeParams,experimentConfig);
            
            if(Q == -1)
                not_sat = false;
                remaining_time = Total_Iters - i;
                regret(t:end) = regret(t-1);
                Value(grasp,2) = Value(grasp,2) + remaining_time;
                break;
            end
            
            UCB_part = sqrt(1/(Value(grasp,2))); 
            Value(grasp,1) =  Value(grasp,1)+Q; 
            Value(grasp,2) = Value(grasp,2)+1; 
            Value(grasp,3) = Value(grasp,1)/Value(grasp,2); 
            Value(grasp,4) = UCB_part; 
            
            [v, best_grasp] = max(Value(:,3)); 
            regret(t) = (interval-1)/interval*regret(t) + (1/interval)*compute_regret_pfc(best_grasp);
            i = i+1; 
            t=t+1; 

         end
    end
    if(prune)
        np_grasp = not_pruned(Value);
        size(np_grasp);
    end

    if vis_bandits
        figure;
        plot(regret)
        title('Simple Regret over Samples'); 
        xlabel('Samples'); 
        ylabel('Simple Regret'); 

        visualize_value(Value,grasp_samples,constructionResults); 
    end
    if(~ts && ~prune)
        %save('marker_bandit_values_pfc','Value');
        save('regret_marker_pfc_mc_ucb','regret','Value');
    elseif(prune)
        save('regret_marker_pfc_sf_ucb','regret','Value');
    else
        save('regret_marker_pfc_ucb','regret','Value');
    end
end


function [not_pruned_grsp] = not_pruned(Value)
 
 high_low = max(Value(:,4)); 
 not_pruned_grsp = find(high_low < Value(:,5));   

end

function [grasp] = get_grasp(Value,t)    
    sigma = 1; 
    
    Value(:,4) = Value(:,4)*sqrt(6*sigma^2*log(t)); 
   
    [v, grasp] = max(Value(:,3)+Value(:,4));
 
end





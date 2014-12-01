function [ best_grasp ] = gittins_index(grasp_samples,num_grasps,shapeParams,experimentConfig, surface_image)
%THOMPSON_SAMPLING Summary of this function goes here
%   Detailed explanation goes here

    Total_Iters = 4000; 
    i = 1; 
    ts = true; 
    prune = false; 
    regret = zeros(Total_Iters+num_grasps,1); 
    not_sat = true; 
    load('matlab/bandit_sampling/gittins_indices');      

    for interval = 1:1
        Storage = {};
        Value = zeros(num_grasps,5); 
        t = 1;
        for i=1:num_grasps
            grasp_samples{i}.current_iter = 1; 
            [Q] = evaluate_grasp(i,grasp_samples,shapeParams,experimentConfig);
            if(Q == 1)
                Storage{i}.p1 = 2;
                Storage{i}.m1 = 1; 
            else
                Storage{i}.p1 = 1;
                Storage{i}.m1 = 2; 
            end

            
            Value(i,1) = Q;
            Value(i,2) = 1; 
            Value(i,3) = (Value(i,1)+1)/(2+Value(i,2)); 
            Value(i,4) = Value(i,3) - 1.96*(1/Value(i,2)*Value(i,3)*(1-Value(i,3)))^(1/2); 
            Value(i,5) = Value(i,3) + 1.96*(1/Value(i,2)*Value(i,3)*(1-Value(i,3)))^(1/2); 
            
            [v best_grasp] = max(Value(:,3));
            if ts
                regret(t) = (interval-1)/interval*regret(t) + (1/interval)*compute_regret_pfc(best_grasp);
            else
               regret(t) = 0; 
            end
            t=t+1; 
        end


        i = 1
        not_sat = true; 
         while(i<Total_Iters && not_sat)
            %i
            if(ts)
                grasp = get_grasp(Value,indices); 
            elseif(prune)
                np_grasp = not_pruned(Value); 
                grasp_idx = randi(length(np_grasp)); 
                grasp = np_grasp(grasp_idx); 
            else
                grasp = randi(num_grasps);
            end

            [Q, grasp_samples] = evaluate_grasp(grasp,grasp_samples,shapeParams,experimentConfig);

            if(Q == 1)
                Storage{grasp}.p1 = Storage{grasp}.p1+1;  
            elseif( Q == -1)
                not_sat = false; 
                break;
            else
                Storage{grasp}.m1 = Storage{grasp}.m1+1; 
            end

          
            Value(grasp,1) =  Value(grasp,1)+Q; 
            Value(grasp,2) = Value(grasp,2)+1; 
            Value(grasp,3) = (Value(grasp,1)+1)/(Value(grasp,2)+2); 
            Value(grasp,4) = Value(grasp,3) - 1.96*(1/Value(grasp,2)*Value(grasp,3)*(1-Value(grasp,3)))^(1/2); 
            Value(grasp,5) = Value(grasp,3) + 1.96*(1/Value(grasp,2)*Value(grasp,3)*(1-Value(grasp,3)))^(1/2);
            
            [v best_grasp] = max(Value(:,3));
            
            if ts
                regret(t) = (interval-1)/interval*regret(t) + (1/interval)*compute_regret_pfc(best_grasp);
            else
                regret(t) = 0; 
            end
            
            i = i+1; 
            t=t+1; 

         end
    end
    np_grasp = not_pruned(Value);
    size(np_grasp);
    figure;
    plot(regret)
    title('Simple Regret over Samples'); 
    xlabel('Samples'); 
    ylabel('Simple Regret'); 
    
    visualize_value( Value,grasp_samples, surface_image )
    
    if(~ts && ~prune)
        save('marker_bandit_values_pfc','Value');
        %save('regret_marker_pfc_mc','regret','Value');
    elseif(prune)
        save('regret_marker_pfc_sf','regret','Value');
    else
        save('regret_marker_pfc','regret','Value');
    end
end


function [not_pruned_grsp] = not_pruned(Value)
 
 high_low = max(Value(:,4)); 
 not_pruned_grsp = find(high_low < Value(:,5));   

end

function [grasp] = get_grasp(Value,indices)    
   
   A = Value(:,1)+1; 
   B = (Value(:,2)-Value(:,1))+1; 
   
   
   
   [v, grasp] = max(indices(A,B)); 
   
end






function [ best_grasps, qualities, Value] = ...
    monte_carlo_apc(grasp_set, pose_samples, grasp_eval, config)
%THOMPSON_SAMPLING Summary of this function goes here
%   Detailed explanation goes here
    num_grasps = size(grasp_set,2); 
    Value = zeros(num_grasps,3); 
    t = 1;
    for i = 1:num_grasps
        grasp_set{i}.cur_iter = 1; 
        grasp = grasp_set{i};
        sdf = pose_samples{grasp.cur_iter}; 
        [Q] = grasp_eval(grasp, sdf, config);
        
        Value(i,1) = Q;
        Value(i,2) = 1; 
        Value(i,3) = (Value(i,1)+1)/(2+Value(i,2)); 
        
        [v best_grasp] = max(Value(:,3));
        
        alpha = Value(best_grasp,1)+1;
        beta = Value(best_grasp,2) - Value(best_grasp,1)+1; 
        bounds(t,1) = betainv(0.95,alpha,beta); 
        bounds(t,2) = betainv(0.05,alpha,beta); 
        
        t=t+1; 
    end


    i = 1;
    not_sat = true; 
    while i < config.max_iters
        grasp_idx = mod(i, num_grasps) + 1;
        
        %fprintf('Sampling grasp %d\n', grasp_idx);
        grasp = grasp_set{grasp_idx}; 
        
        if(grasp_set{grasp_idx}.cur_iter < max(size(pose_samples)))
            sdf = pose_samples{grasp_set{grasp_idx}.cur_iter}; 

            [Q] = grasp_eval(grasp,sdf,config);
        else 
            break; 
        end
        
        grasp_set{grasp_idx}.cur_iter = grasp_set{grasp_idx}.cur_iter+1;
   
        Value(grasp_idx,1) =  Value(grasp_idx,1)+Q; 
        Value(grasp_idx,2) = Value(grasp_idx,2)+1; 
        Value(grasp_idx,3) = (Value(grasp_idx,1)+1)/(Value(grasp_idx,2)+2); 
       
        [v best_grasp] = max(Value(:,3));
        
        alpha = Value(best_grasp,1)+1;
        beta = Value(best_grasp,2) - Value(best_grasp,1)+1; 
        bounds(t,1) = betainv(0.95,alpha,beta); 
        bounds(t,2) = betainv(0.05,alpha,beta); 
        %abs(bounds(t,1) - bounds(t,2))
        if(abs(bounds(t,1) - bounds(t,2))<config.epsilon)
            break;
        end
        i = i+1; 
        t=t+1; 

    end
    [pfc_sorted, grasp_inds] = sort(Value(:,3), 'descend');
    qualities = pfc_sorted(1:config.num_candidate_grasps);
    best_grasps = cell(1, config.num_candidate_grasps);
    for j = 1:config.num_candidate_grasps
        best_grasps{j} = grasp_set{grasp_inds(j)};
    end

%     [quality best_idx] = max(Value(:,3)); 
%     best_grasp = grasp_set{best_idx}; 

end

function [grasp] = get_grasp(Value)    
   
   A = Value(:,1)+1; 
   B = (Value(:,2)-Value(:,1))+1; 
   
   Sample = betarnd(A,B); 
   
   [v, grasp] = max(Sample); 
   
end

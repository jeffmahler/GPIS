function [ best_grasp quality] = ...
    thompson_apc(grasp_set,pose_samples,config,grasp_eval)
%THOMPSON_SAMPLING Summary of this function goes here
%   Detailed explanation goes here
    num_grasps = size(grasp_set,1); 
    Value = zeros(num_grasps,3); 
    t = 1;
    for i = 1:num_grasps
        grasp_set{i}.cur_iter = 1; 
        
        grasp = grasp_set{1};
        sdf = pose_samples{grasp_set{grasp_idx}.cur_iter}; 
        
        [Q] = grasp_eval(grasp,sdf,config);
        
        
        
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
     while(1 == 1)
        grasp_idx = get_grasp(Value);      
        grasp = grasp_set{grasp_idx}; 
        
        if(grasp_set{grasp_idx}.cur_iter < max(size(pose_samples)))
            sdf = pose_samples{grasp_set{grasp_idx}.cur_iter}; 

            [Q] = grasp_eval(grasp,sdf,config);
        else 
            break; 
        end
        
        grasp_set{grasp_idx}.cur_iter = grasp_set{grasp_idx}.cur_iter+1;
   
        Value(grasp,1) =  Value(grasp,1)+Q; 
        Value(grasp,2) = Value(grasp,2)+1; 
        Value(grasp,3) = (Value(grasp,1)+1)/(Value(grasp,2)+2); 
       
        [v best_grasp] = max(Value(:,3));
        
        alpha = Value(best_grasp,1)+1;
        beta = Value(best_grasp,2) - Value(best_grasp,1)+1; 
        bounds(t,1) = betainv(0.95,alpha,beta); 
        bounds(t,2) = betainv(0.05,alpha,beta); 
        
        if(abs(bounds(t,1) - bounds(t,2))<config.epsilon)
            break;
        end
        i = i+1; 
        t=t+1; 

     end
    [quality best_idx] = max(Value(:,3)); 
    best_grasp = grasp_set{best_idx}; 
    
    np_grasp = not_pruned(Value);
    size(np_grasp);

end

function [grasp] = get_grasp(Value)    
   
   A = Value(:,1)+1; 
   B = (Value(:,2)-Value(:,1))+1; 
   
   Sample = betarnd(A,B); 
   
   [v, grasp] = max(Sample); 
   
end

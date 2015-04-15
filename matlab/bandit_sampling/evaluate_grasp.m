function [Q, grasp_samples] = evaluate_grasp(grasp,grasp_samples,shapeParams,experimentConfig)
    
        
       
        grasp_stor = grasp; 
        num_samples = grasp_samples{grasp_stor}.num_samples; 
        iter = grasp_samples{grasp_stor}.current_iter;
        if(iter > num_samples)
            Q = -1; % same arm forever at this point   -1; 
            return;
        end
       
        Q = grasp_samples{grasp_stor}.Q_samps(iter,1); 
        grasp_samples{grasp_stor}.current_iter = iter + 1; 
end
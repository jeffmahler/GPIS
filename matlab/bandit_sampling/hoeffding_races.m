function [ best_grasp, regret, Value ] = kehoe(grasp_samples, ...
    num_grasps,shapeParams,experimentConfig, surface_image, vis_bandits)
%UGABEB Summary of this function goes here
%   Detailed explanation goes here

    Total_Iters = 25000; 
    M = 0.1;
    I = 1000;
    i = 1; 
    regret = zeros(Total_Iters+num_grasps,1); 
    not_sat = true; 
    B = 1; 
    delta = 0.05; 
    K = num_grasps;
    
    for interval = 1:1
        Storage = {};
        Value = zeros(num_grasps,5); 
        t = 1;
        for i=1:num_grasps
            grasp_samples{i}.current_iter = 1; 
            [Q] = evaluate_grasp(i,grasp_samples,shapeParams,experimentConfig);
      
            
            Value(i,1) = Q;
            Value(i,2) = 1; 
            Value(i,3) = (Value(i,1)+1)/(2+Value(i,2)); 
            Value(i,4) = Value(i,3) - 1.96*(1/Value(i,2)*Value(i,3)*(1-Value(i,3)))^(1/2); 
            Value(i,5) = Value(i,3) + 1.96*(1/Value(i,2)*Value(i,3)*(1-Value(i,3)))^(1/2); 
            
            [v best_grasp] = max(Value(:,3));
            
            regret(t) = (interval-1)/interval*regret(t) + (1/interval)*compute_regret_pfc(best_grasp);
            t=t+1; 
        end


    i = 1
    good_grasps = [1:size(Value,1)]; 
    not_sat = true;
    while(i < Total_Iters && not_sat)
        
        idx = randi(max(size(good_grasps)));
        grasp = good_grasps(idx); 
        
        [Q, grasp_samples] = evaluate_grasp(grasp,grasp_samples,shapeParams,experimentConfig);

        if( Q == -1)
            not_sat = false; 
            remaining_time = Total_Iters - i;
            regret(t:end) = regret(t-1);
            Value(grasp,2) = Value(grasp,2) + remaining_time;
            break;
        end

        Value(grasp,1) =  Value(grasp,1)+Q; 
        Value(grasp,2) = Value(grasp,2)+1; 
        Value(grasp,3) = (Value(grasp,1)+1)/(Value(grasp,2)+2); 
        Value(grasp,4) = Value(grasp,3) - sqrt(B^2*log(2/delta)/Value(grasp,2)); 
        Value(grasp,5) = Value(grasp,3) + sqrt(B^2*log(2/delta)/Value(grasp,2));

        [v best_grasp] = max(Value(:,3));

        regret(t) = (interval-1)/interval*regret(t) + (1/interval)*compute_regret_pfc(best_grasp);

        i = i+1; 
        t=t+1; 
        if(rem(i,I)==0)
            good_grasps = not_pruned(Value,good_grasps,M); 
        end
    end
    end
   
    if vis_bandits
        figure;
        plot(regret)
        title('Simple Regret over Samples'); 
        xlabel('Samples'); 
        ylabel('Simple Regret'); 

        visualize_value( Value,grasp_samples,surface_image )
    end
    
   
end

function [phases] = compute_phases(K,n)
    phases = zeros(K-1,1); 

    log_K = 1/2; 
    
    for i=2:K
        log_K = log_K + 1/i; 
    end
    phases(1) = 0; 
    for i=2:K-1
        phases(i) = 1/log_K*(n-K)/(K+1-(i-1)); 
    end
   
end

function [not_pruned] = not_pruned(Value,good_grasps,M)
     [sX,sInd] = sort(Value(:,3),'descend');
     L = M*size(good_grasps);
     not_pruned = sInd(1:L); 
 


end
   

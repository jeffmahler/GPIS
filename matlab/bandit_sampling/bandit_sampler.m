function [ best_grasp ] = bandit_sampler( grasp_samples,num_grasps,shapeParams,experimentConfig,constructionResults )
%BANDIT_SAMPLER Summary of this function goes here
%   Detailed explanation goes here

    Total_Iters = 30000; 
    i = 1; 
    Value = zeros(num_grasps,6);
    regret = zeros(Total_Iters+2000,1); 
    grasp_samples_orig = grasp_samples; 
    epsilon = 1.0;
    %epsilon = 0.7; 
    Storage = {};
    dif = (epsilon-0.01)/Total_Iters
   
    for interval=1:20
        interval
        [v best_grasp] = max(Value(:,1)); 
        best_grasp = Value(best_grasp,6)
        if(size(Value,1) == 1)
            fill = regret(t-1); 
            regret(t:end) =  (interval-1)/interval*regret(t:end); 
        end
        t=1;
        Value = zeros(num_grasps,6);
        Storage = {}; 
        grasp_samples = grasp_samples_orig;
        for i=1:num_grasps
            Storage{i} = struct();
            Storage{i}.mean = [];
            Storage{i}.conf = [];
            Value(i,6) = i;
            [grasp_samples,Value,Storage] = evaluate_grasp(i,grasp_samples,Value,shapeParams,experimentConfig,Storage);
            [v best_grasp] = max(Value(:,1));

            regret(t) = (interval-1)/interval*regret(t) + (1/interval)*compute_regret(best_grasp);
            t=t+1;
        end

        for i=1:num_grasps
            [grasp_samples,Value,Storage] = evaluate_grasp(i,grasp_samples,Value,shapeParams,experimentConfig,Storage);
            [v best_grasp] = max(Value(:,1));

            regret(t) = (interval-1)/interval*regret(t) + (1/interval)*compute_regret(best_grasp);
            t=t+1;
        end
        i=1;
        
        while(i<Total_Iters && size(Value,1)~=1)

            grasp = get_grasp(Value,epsilon); 

            [grasp_samples,Value,Storage] = evaluate_grasp(grasp,grasp_samples,Value,shapeParams,experimentConfig,Storage);
            %[Value] = filter(Value);

            [v best_grasp] = max(Value(:,1));
            best_grasp = Value(best_grasp,6);
            regret(t) = (interval-1)/interval*regret(t) + (1/interval)*compute_regret(best_grasp);
            t=t+1;

            %epsilon = epsilon - dif; 
            i = i+1; 

        end
    end
    [v best_grasp] = max(Value(:,1)); 
    best_grasp = Value(best_grasp,6)
    plot_bandit(Storage,Value)
    cp = grasp_samples{best_grasp}.cp;
    regret(t) = compute_regret(best_grasp);
    plot_grasp_arrows( constructionResults.newSurfaceImage, cp(1,:)', cp(3,:)', -(cp(1,:)-cp(2,:))', -(cp(3,:)-cp(4,:))', 2,2)
    figure;
    plot(regret)
    title('Simple Regret over Samples'); 
    xlabel('Samples'); 
    ylabel('Simple Regret'); 
    
end


function [] = plot_bandit(Storage,Value)

    top_grasps = 10;
    if(top_grasps > size(Value,1))
        top_grasps = size(Value,1); 
    end
    colors = color_to_num(); 
    num_colors = size(colors,2); 
    
    
    [B,ix] = sort(Value(:,1),'descend');
     B = B(1:top_grasps);
     ix = ix(1:top_grasps);
     c =1;
    figure; 
    font = 18; 
    for i=1:top_grasps
        
        hold on;
        grasp = Value(ix(i),6); 
        errorbar(Storage{grasp}.mean,Storage{grasp}.conf,colors{c});
        axis([0 1000 0.04 0.2]);
        title('Bandit Sampling','FontSize',font); 
        xlabel('Samples','FontSize',font); 
        ylabel('Grasp Quality','FontSize',font); 
        c = c+1; 
        if(c>num_colors)
            c = 1; 
        end
    end


end




function [grasp] = get_grasp(Value,epsilon)

    [v, i] = max(Value(:,1)); 
    
    r = rand;
  
    if(r > epsilon)
        grasp = i
    else
        
        grasp  = randi(size(Value(:,1),1));

    end
    
end


function [grasp_samples,Value,Storage] = evaluate_grasp(grasp,grasp_samples,Value,shapeParams,experimentConfig,Storage)
        
        cm = shapeParams.com;
        ca = atan(experimentConfig.frictionCoef);
        
        c = zeros(3,2);
        
        grasp_stor = Value(grasp,6); 
       
        iter = grasp_samples{grasp_stor}.current_iter;
        if(iter > 1500)
            return;
        end
        n1 = grasp_samples{grasp_stor}.n1_emps(iter,:);
        %n1 = n1/norm(n1,2);
        
        n2 = grasp_samples{grasp_stor}.n2_emps(iter,:);
        %n2 = n2/norm(n2,2);
        
        norms(:,1) = n1';
        norms(:,2) = n2'; 
        
        c1 = grasp_samples{grasp_stor}.c1_emps(iter,:);
        c2 = grasp_samples{grasp_stor}.c2_emps(iter,:);
        
        c(1:2,1) = grasp_samples{grasp_stor}.loa_1(c1,:)';
        c(1:2,2) = grasp_samples{grasp_stor}.loa_2(c2,:)'; 
        
        if(abs(c(1,1)- c(1,2))<0.001 && abs(c(2,1)-c(2,2))<0.001)
            return;
        end
        
        for k = 1:2
            if(k ==1)
                forces = forces_on_friction_cone(norms(:,k),ca);
            else
                forces = [forces forces_on_friction_cone(norms(:,k),ca)];
            end
        end
        
        Q = ferrari_canny( [cm 0]',c,forces );
        
        grasp_samples{grasp_stor}.q = [grasp_samples{grasp_stor}.q; Q];
        
        Value(grasp,1) = mean(grasp_samples{grasp_stor}.q);
        
        std_q = std(grasp_samples{grasp_stor}.q);
        N = size(grasp_samples{grasp_stor}.q,1);
        Value(grasp,2) = 1.96*std_q/(sqrt(N));
        
        Value(grasp,3) = Value(grasp,1)+Value(grasp,2);
        
        grasp_samples{grasp_stor}.current_iter = iter+1;
        
        
        Value(grasp,5) = Value(grasp,1)-Value(grasp,2);
        
        
        
        Storage{grasp_stor}.mean = [Storage{grasp_stor}.mean; Value(grasp,1)];
        Storage{grasp_stor}.conf = [Storage{grasp_stor}.conf; Value(grasp,2)];
        Value(grasp,4) = iter+1;
        
        
        

end

function [Value] = filter(Value)

    %Filter out lower bounds
    max_min = max(Value(:,5));

    n_value = find(Value(:,3) >= max_min);
   
    if(min(Value(:,4)) > 2)
        Value = Value(n_value,:);
    end
        
end
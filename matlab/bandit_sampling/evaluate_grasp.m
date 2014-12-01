function [Q, grasp_samples] = evaluate_grasp(grasp,grasp_samples,shapeParams,experimentConfig)
    
        
        c = zeros(3,2);
        grasp_stor = grasp; 
        
        iter = grasp_samples{grasp_stor}.current_iter;
        if(iter > 1300)
            Q = -1; 
            return;
        end
       
        cm = grasp_samples{grasp_stor}.com(iter,:);
        
        ca = atan(grasp_samples{grasp_stor}.fc(iter));
        
        if(size(grasp_samples{grasp_stor}.n1_emps,1) < 1000 || size(grasp_samples{grasp_stor}.n2_emps,1) < 1000)
            Q = 0; 
            return 
        end
        
        n1 = grasp_samples{grasp_stor}.n1_emps(iter,:);
   
        
        n2 = grasp_samples{grasp_stor}.n2_emps(iter,:);
       
        
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
        grasp_samples{grasp_stor}.current_iter = iter +1;
        Q = ferrari_canny( [cm 0]',c,forces );
        
        Q = Q>0;
end
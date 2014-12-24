function [Q] = evaluate_grasp_lite(n1,n2,c1,c2,cm,fc)
    
        ca = atan(fc);
       
      
        c = zeros(3,2); 
        norms(:,1) = n1;
        norms(:,2) = n2; 
        
        c(1:2,1) = c1;
        c(1:2,2) = c2; 
        
        if(abs(c(1,1)- c(1,2))<0.001 && abs(c(2,1)-c(2,2))<0.001)
            Q = 0; 
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
        
        Q = Q>0;
end
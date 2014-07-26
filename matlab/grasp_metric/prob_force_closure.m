function [ P_fc,E_Q ] = probs_on_Qg( loa_1,loa_2,Norms,pc_1,pn_1,pc_2,pn_2,cm,ca)
%PROB_FORCE_CLOSURE Takes distribution on grasp parameters and returns
%probability of achieving force closure. 

size_c1 = size(loa_1,1); 
size_n1 = size(pn_1,1); 

size_c2 = size(loa_2,1); 
size_n2 = size(pn_2,1); 

    for i = 1:size_c1
        for j = 1:size_c2
            for theta1 = 1:size_n1
                for theta2 = 1:size_n2
                    %Get Parameters
                    norm(:,1) = Norms(theta1,:);
                    norm(:,2) = Norms(theta2,:); 
                    c(:,1) = loa_1(i,:)';
                    c(:,2) = loa_2(i,:)'; 
                    for i = 1:2
                        if(i ==1)
                            forces = forces_on_friction_cone(norm,ca);
                        else
                            forces = [forces forces_on_friction_cone(norm,ca)];
                        end
                    end
                    Q = ferrari_canny( cm,c,f );
                    
                    p_g(index,1) = pc_1(i)*pc_2(j)*pn_1(theta1)*pn_2(theta2);
                    p_g(index,2) = Q; 
                    p_g(index,3) = Q >0; 
                    index = index+1; 
                    
                end
            end
        end
    end
    
    p_g(:,1) = p_g(:,1)/norm(p_g(:,1),1);
    
    E_Q = sum(p_g(:,1).*p_g(:,2));
    
    stable = p_g(:,3); 
    
    P_fc = sum(p_g(:,stable));

end

function [ forces ] = forces_on_friction_cone(norm,ca)
        f = norm; 
        f = f/norm(f,1)*(1/2);
       
        % get extrema of friction cone, negative to point into object
        opp_len = tan(ca) * norm(f);
        opp_dir = [-1 / f(1,1); 1 / f(2,1)];
        opp = opp_len * opp_dir / norm(opp_dir);
        f_r = -(f + opp);
        f_l = -(f - opp);

        % normalize (can only provide as much force as the normal force)
        f_r =  f_r * norm(f) / norm(f_r);
        f_l =  f_l * norm(f) / norm(f_l);

        forces(:,1) = [f_r; 0]; 
       
        forces(:,2) = [f_l; 0];
end
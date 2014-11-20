function [ E_Q,lb ] = compute_lb( loa_1,loa_2,Norms,pc_1,pn_1,pc_2,pn_2,cm,cone_angle,fc,gpModel,Grasp_Data)
%COMPUTE_LB Given a distribution on grasps returns a lower bound on
%expected grasp 

    nc =2; 
    zeta = 0.95; 
    
    [E_C,E_N,sc,sn ] = expected_grasp( loa_1,loa_2,Norms,pc_1,pn_1,pc_2,pn_2);
      
    forces = forces_on_friction_cone(E_N,cone_angle);
    
    E_C = [E_C; zeros(1,nc)]
    
    E_Q = ferrari_canny( [cm 0]',E_C,forces);
    
    [b(1) r1] = find_b(pc_1,pn_1,sc(1),sn(1),Norms,loa_1,gpModel,zeta,cm,fc);
        
    [b(2) r2] = find_b(pc_2,pn_2,sc(2),sn(2),Norms,loa_2,gpModel,zeta,cm,fc);
    
    
    max_r = max(r1,r2); 
    num_samples =1; 
    
    for i=1:num_samples
        [v,ic1] = max(pc_1); 
        grasp(1:2) = loa_1(ic1,:)-cm; 
        
        [v,ic2] = max(pc_2);
        grasp(3:4) = loa_2(ic2,:)-cm; 
        
        [v,in1] = max(pn_1);
        grasp(5:6) = Norms(in1,:); 
        
        [v,in2] = max(pn_2);
        grasp(7:8) = Norms(in2,:); 
        
        
        [vl vh] = compute_closest_grasp( Grasp_Data,grasp,fc,max_r);
    end
    
    
        
    lb = max(b); 

end

function [b,r] =  find_b(p_c,pn,sc,sn,Norms,loa,gpModel,zeta,cm,fc)

    cov_loa = gp_cov(gpModel,loa, [], true);

    mean_loa = gp_mean(gpModel,loa,true);
    
    [d_c,first,last] = integrate_contact(p_c,sc,zeta,loa);
    
    [p_n,X_circ,Y_circ] = normal_distribution(loa,cov_loa,mean_loa,p_c,first,last);
    
    dn = integrate_normal(p_n,sn,zeta,Norms);
    
    r = norm(loa(sc,:)-cm,2); 
    
    b = dn*(1+r)+d_c+fc*(dn*(1+r)+d_c); 


end

function [d_n] = integrate_normal(p_n,sn,zeta,Norms)

    i=sn; 
    j=sn;
    sum = 0; 
    f_norm = 0; 
    l_norm = 0; 
    while(sum <= zeta)
        if(i ~= 1)
            sum = p_n(i)+sum; 
            f_norm = f_norm+norm(Norms(i,:)-Norms(sn,:),2)*p_n(i);
            i = i-1; 
        elseif(i == 1)
            sum = p_n(i)+sum; 
            i = size(p_n(i),1); 
            f_norm = f_norm+norm(Norms(i,:)-Norms(sn,:),2)*p_n(i);
        end
        
        if(j ~= size(p_n,1))
            sum = p_n(j)+sum;
            j = j+1; 
            l_norm = l_norm+ norm(Norms(j,:)-Norms(sn,:),2)*p_n(j);
        elseif(j == size(p_n,1))
            sum = p_n(j)+sum; 
            j = 1; 
            l_norm = l_norm+ norm(Norms(j,:)-Norms(sn,:),2)*p_n(j);
        end
    end
       
    d_n = max(f_norm,l_norm);

end

function [d_c,first,last] = integrate_contact(pc,sc,zeta,loa)

    i = sc; 
    j = sc; 
    sum = 0;
    f_norm = 0; 
    l_norm = 0;
    while(sum <= zeta)
        if(i >= 1)
            sum = pc(i)+sum; 
            f_norm = f_norm+norm(loa(i,:)-loa(sc,:),2)*pc(i);
            i = i-1; 
        end
            
        
        if(j <= size(pc,1))
            sum = pc(j)+sum;
            l_norm = l_norm+norm(loa(j,:)-loa(sc,:),2)*pc(j);
            j = j+1; 
        end
    end
    
    if(i < 1)
        i = 1; 
    end
    
    if(j > size(pc,1))
        j = size(pc,1); 
    end
    
    
    first = i; 
    last = j; 
     
    
    d_c = max(f_norm,l_norm);

end


function [ E_C,E_N,sc,sn ] = expected_grasp( loa_1,loa_2,Norms,pc_1,pn_1,pc_2,pn_2)
%EXPECTED_GRASP returns the expected grasp on the distribution 

E_C(:,1) = [sum(loa_1(:,1).*pc_1) sum(loa_1(:,2).*pc_1)]'; 

sc(1) = find_closest_index(E_C(:,1)',loa_1);

E_C(:,2) = [sum(loa_2(:,1).*pc_2) sum(loa_2(:,2).*pc_2)]';

sc(2) = find_closest_index(E_C(:,2)',loa_2);

E_N(:,1) = [sum(Norms(:,1).*pn_1) sum(Norms(:,2).*pn_1)]'; 
E_N(:,2) = [sum(Norms(:,1).*pn_2) sum(Norms(:,2).*pn_2)]';  

for i=1:2
    E_N(:,i) = E_N(:,i)/norm(E_N(:,i),2);
end

sn(1) = find_closest_index(E_N(:,1)',Norms);
sn(2) = find_closest_index(E_N(:,2)',Norms);

end

function [idx] = find_closest_index(v_array,table)

sub = [table(:,1)-v_array(1) table(:,2)-v_array(2)];

for i=1:size(sub,1)
    tmp(i) = norm(sub(i,:),2); 
end

[C idx] = min(tmp); 

end



function [ forces ] = forces_on_friction_cone(nrm,cone_angle)
        index = 1;
        for i=1:size(nrm,2); 
            f = nrm(:,i)
            f = f/norm(f,1)*(1/2);


            %get extrema of friction cone, negative to point into object
            opp_len = tan(cone_angle) * norm(f)
            opp_dir = [-f(2,1); f(1,1)]
            opp = opp_len * opp_dir / norm(opp_dir)
            f_r = -(f + opp);
            f_l = -(f - opp);

            % normalize (can only provide as much force as the normal force)
            f_r =  f_r * norm(f) / norm(f_r);
            f_l =  f_l * norm(f) / norm(f_l);

            forces(:,index) = [f_r; 0]; 
            index = index+1;
            forces(:,index) = [f_l; 0];
            index = index+1;
        end
end
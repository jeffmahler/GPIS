function [ avg_q,var_q ] = sample_known_dist( loa_1,loa_2,ca,cm,c1_emp,c2_emp,n1_emp,n2_emp)
%SAMPLE_KNOWN_DIST Summary of this function goes here
%   Detailed explanation goes here
    avg_q = [];
    q = [];
    var_q = [];
    
 
    
    for i =1:min(length(c1_emp),length(c2_emp))
        c = zeros(3,2); 
       
%         norms(:,1) = Norms(samples_n1(i),:)';
%         norms(:,2) = Norms(samples_n2(i),:)'; 
%         c(1:2,1) = loa_1(samples_c1(i),:)';
%         c(1:2,2) = loa_2(samples_c2(i),:)'; 
        n1 = n1_emp(i,:);
        %n1 = n1/norm(n1,2);
        
        n2 = n2_emp(i,:);
        %n2 = n2/norm(n2,2);
        
        norms(:,1) = n1';
        norms(:,2) = n2'; 
        c(1:2,1) = loa_1(c1_emp(i),:)';
        c(1:2,2) = loa_2(c2_emp(i),:)'; 

        for k = 1:2
            if(k ==1)
                forces = forces_on_friction_cone(norms(:,k),ca);
            else
                forces = [forces forces_on_friction_cone(norms(:,k),ca)];
            end
        end
        Q = ferrari_canny( [cm 0]',c,forces );
        q = [q; Q];
        avg_q = [avg_q; mean(q)];
        std_q = std(q); 
        var_q = [var_q; 1.96*std_q/(sqrt(i))]; 
    end


end


function [p_c] = convert_emp_c_to_dist(loa,contact_emp)

     dist = zeros(size(loa,1),1); 


    for i=1:size(contact_emp,1)
        t =contact_emp(i);

        dist(t) = dist(t)+1; 
    end


    p_c = dist/norm(dist,1); 
    
end

function [p_n] = convert_emp_n_to_dist(Norms,normals_emp,p_n)

    dist = zeros(size(p_n)); 


    for i=1:size(normals_emp,1)
        n =normals_emp(i,:);
        n = n/norm(n,2);
        idx = find_closest_index(n,Norms);
        dist(idx) = dist(idx)+1; 
    end


    p_n = dist/norm(dist,1); 
    

end

function [idx] = find_closest_index(v_array,table)

sub = [table(:,1)-v_array(1) table(:,2)-v_array(2)];

for i=1:size(sub,1)
    tmp(i) = norm(sub(i,:),2); 
end

[C idx] = min(tmp); 

end

% 
% function [ forces ] = forces_on_friction_cone(nrm,cone_angle)
%         f = nrm;
%         f = f/norm(f,1)*(1/2);
%        
%         
%         %get extrema of friction cone, negative to point into object
%         opp_len = tan(cone_angle) * norm(f);
%         opp_dir = [-f(2,1); f(1,1)];
%         opp = opp_len * opp_dir / norm(opp_dir);
%         f_r = -(f + opp);
%         f_l = -(f - opp);
% 
%         % normalize (can only provide as much force as the normal force)
%         f_r =  f_r * norm(f) / norm(f_r);
%         f_l =  f_l * norm(f) / norm(f_l);
% 
%         forces(:,1) = [f_r; 0]; 
%        
%         forces(:,2) = [f_l; 0];
% end



function [sample] = sampler(p,num_samples)
    p = p';
    n = length(p);
   
    uni=rand(1,num_samples);
    cumprob=[0 cumsum(p)];
    sample=zeros(1,num_samples);
    for j=1:n
      ind=find((uni>cumprob(j)) & (uni<=cumprob(j+1)));
      sample(ind)=j;
    end

end
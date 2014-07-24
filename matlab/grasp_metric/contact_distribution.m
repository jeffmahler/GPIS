function [p_c] = contact_distribution(loa,cov_loa,mean_loa)

    %Calculate Contact Point Distribution 
    %TODO: VISUALIZE AGAINST OBJECT
    p_c = zeros(size(loa(:,1))); 

    for i = 1:size(loa,1)
        
        p0 = mvnpdf(0,mean_loa(i),cov_loa(i,i)); 
        if(i ~= 1)
            [mean_con,cov_con] = gauss_condition(mean_loa(1:i),cov_loa(1:i,1:i)); 
            ll = zeros(size(mean_con));
            ul = zeros(size(mean_con))+10000; 
            
            P = mvncdf(ll,ul,mean_con,cov_con);
            p_c(i) = p0*P;
        else
            p_c(i) = p0; 
        end
        
        
    end
    
  
    p_c = p_c/norm(p_c,1);
   
end

function [nu_mean,nu_cov] = gauss_condition(mean,cov)
    
    mu_b = mean(end);
    mu_a = mean(1:end-1); 
    
    sigma_b = cov(end,end); 
    sigma_a = cov(1:end-1,1:end-1); 
    sigma_c = cov(1:end-1,end); 
    
    nu_mean = mu_a + sigma_c/sigma_b*(0-mu_b); 
    
    nu_cov = sigma_a-sigma_c/sigma_b*sigma_c';
   
end


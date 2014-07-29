function [p_c] = contact_distribution(loa,cov_loa,mean_loa)

    %Calculate Contact Point Distribution 
    %TODO: VISUALIZE AGAINST OBJECT
    p_c = zeros(size(loa(:,1))); 
    trail = 19; 
    options = optimset('TolFun',1e-2); 
    P(1) = 1; 

    for i = 1:size(loa,1)/2
        
        p0 = mvnpdf(0,mean_loa(i),cov_loa(i,i)); 
        if(i ~= 1)
            [mean_con,cov_con] = gauss_condition(mean_loa(1:i),cov_loa(1:i,1:i)); 
            
            if(i >=trail+1)
                ll = zeros(trail,1);
                ul = zeros(trail,1)+5;
                dim = size(mean_con,1);
              
                
                A = make_psd(cov_con(i-trail:end,i-trail:end)); 
            
                try
                    P(i) = mvncdf(ll,ul,mean_con(i-trail:end,:),A,options);
                catch
                    A
                end
                    
            else
                ll = zeros(size(mean_con));
                ul = zeros(size(mean_con))+5; 
                cov_con = make_psd(cov_con); 
               
                try
                    P(i) = mvncdf(ll,ul,mean_con,cov_con,options);
                catch 
                    cov_con
                    eig(cov_con)
                end
                
                if(isnan(P(i)))
                    P(i) =1;
                end
            end
           
            
            p_c(i) = p0*P(i);
        else
            p_c(i) = p0; 
        end
        
        
    end
    
  
    p_c = p_c/norm(p_c,1);
   
end

function [Apsd] = make_psd(A)

Apsd = A + eye(size(A))*1e-12; 



end

function [nu_mean,nu_cov] = gauss_condition(mean,cov)
    
    mu_b = mean(end);
    mu_a = mean(1:end-1); 
    
    sigma_b = cov(end,end); 
    sigma_a = cov(1:end-1,1:end-1); 
    sigma_c = cov(1:end-1,end); 
    
    nu_mean = mu_a + (sigma_c/sigma_b)*(0-mu_b); 
    C = (sigma_c/sigma_b)*sigma_c';
    nu_cov = sigma_a-(C+C')/2;
   
end


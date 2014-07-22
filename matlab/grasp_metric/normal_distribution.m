function [p_n,x] = normal_distribution(loa,cov_loa,mean_loa,p_c)

    step_size = 0.002; 
    
        
    x = [-0.4:0.002:0.4-step_size]'; 
    X = [zeros(size(x))+x(1) x]; 
    for i=2:size(x,1)
        X = [X;[zeros(size(x))+x(i) x]];
    end

    p_n = zeros(size(X,1),1); 

    for t = 1:size(loa,1)
        [marg_cov,marg_mean] = marg_normals(cov_loa,mean_loa,t);
        size(p_n)
        size(x)
        p_n = p_n+mvnpdf(X,marg_mean',marg_cov)*p_c(t);      
    end
    
    p_n = reshape(p_n,sqrt(size(X,1)),sqrt(size(X,1)));
   
    
end

function [dist] = project_to_sphere(cov,mean)






end

function [marg_cov,marg_mean] = marg_normals(cov_loa,mean_loa,i)

sz = size(mean_loa,1)/3;
dx = sz+i; 
dy = 2*sz+i;

marg_mean = [mean_loa(dx); mean_loa(dy)];

marg_cov = [cov_loa(dx,dx) cov_loa(dx,dy); 
            cov_loa(dx,dy) cov_loa(dy,dy)];

end
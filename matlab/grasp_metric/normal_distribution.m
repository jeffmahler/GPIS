function [p_n,X_circ,Y_circ] = normal_distribution(loa,cov_loa,mean_loa,p_c,first,last)

    step_size = 0.002; 
    
    if nargin < 5
        first = 1; 
        last = size(loa,1); 
    end
        
   
    for t = first:last
        [marg_cov,marg_mean] = marg_normals(cov_loa,mean_loa,t);
        
        
        [dist,X_circ,Y_circ] = project_to_sphere(marg_cov,marg_mean);
      
        
        if (t == first)
            p_n = dist*p_c(t); 
        end
        
        p_n = p_n+dist*p_c(t); 
    end
    
  
   
    
end

function plot_circle(dist,X,Y)

    Dist = diag(dist); 
    h = surf(X,Y,Dist); 
    
    set(h,'edgecolor','interp')
    axis([-1 1 -1 1]); 


end


function [dist,X,Y] = project_to_sphere(cov,mean)
%dt = 0.04; 
dt = 0.01; 
gridDM = ceil(2*pi/dt); 

dist = zeros(gridDM,1); 
index =1; 
    for theta = 0:dt:2*pi
       
       
        x = cos(theta); 
        y = sin(theta); 
        
        X(index) = x; 
        Y(index) = y; 
        
        z = (-1+x^2+y^2)/(1+x^2+y^2);
        dist(index) = mvnpdf([x/(1-z); y/(1-z)], mean,cov); 
        if(isnan(dist(index)))
            x
            y
        end
        index = index+1; 
       
    end
    %Scale up for numerical percision issues
  
    
    if(norm(dist,1) ~= 0)
       dist = dist/norm(dist,1);
    end
    
end

function [marg_cov,marg_mean] = marg_normals(cov_loa,mean_loa,i)

sz = size(mean_loa,1)/3;
dx = sz+i; 
dy = 2*sz+i;

marg_mean = [mean_loa(dx); mean_loa(dy)];

marg_cov = [cov_loa(dx,dx) cov_loa(dx,dy); 
            cov_loa(dx,dy) cov_loa(dy,dy)];

end
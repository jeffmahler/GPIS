function [ p_c,p_n,p_nc] = Compute_Distributions(  gpModel,grip_point,testImage)
%COMPUTE_DISTRIBUTIONS Summary of this function goes here
%   Detailed explanation goes here

    loa = compute_loa(grip_point); 

    %Calculate Distribution Along Line 
    cov_loa = gp_cov(gpModel,loa, [], true);

    mean_loa = gp_mean(gpModel,loa,true);


    p_c = contact_distribution(loa,cov_loa,mean_loa);
    plot_contact(p_c,grip_point,loa,testImage);
    [p_n, x] = normal_distribution(loa,cov_loa,mean_loa,p_c);
    plot_normal(p_n,grip_point,x,testImage)
    %plot(loa(:,1),mean_loa(1:size(loa,1)));

end

function [p] = plot_contact(dist,point,loa,testImage)
    
    figure; 
    subplot(1,2,1)
    imshow(testImage);
 
    hold on     
    plot(point(:,1),point(:,2))
    title('Mean Function of GPIS'); 
    hold off
    
    subplot(1,2,2)
    plot(loa(:,1),dist); 
    title('Distribution on Contact Points'); 
    xlabel('x-axis'); 
    ylabel('pdf'); 
    
    
end

function [p] = plot_normal(dist,point,x,testImage)
    
    figure; 
    subplot(1,2,1)
    imshow(testImage);
 
    hold on     
    plot(point(:,1),point(:,2))
    title('Mean Function of GPIS'); 
    axis([1,25,1,25])
    hold off
    
    subplot(1,2,2)
    h = surf(x,x,dist); 
    set(h,'edgecolor','interp')
    colormap
    title('Distribution on Surface Normals'); 
    xlabel('x-axis'); 
    ylabel('y-axis'); 
    zlabel('pdf');
    
    
end

function [loa] = compute_loa(grip_point)
%Calculate Line of Action given start and end point

    step_size = 2; 

    start_point = grip_point(1,:); 
    end_p = grip_point(2,:); 

    grad = end_p-start_point; 
    end_time = norm(grad, 2);
    grad = grad/end_time; 
    i=1; 
    time = 0;

    while(time < end_time)
        point = start_point + grad*time;
        loa(i,:) = point;
        time = time + step_size; 
        i = i + 1;
    end
    loa
end

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
    surf(x,x,p_n);
    
end



function [marg_cov,marg_mean] = marg_normals(cov_loa,mean_loa,i)

sz = size(mean_loa,1)/3;
dx = sz+i; 
dy = 2*sz+i;

marg_mean = [mean_loa(dx); mean_loa(dy)];

marg_cov = [cov_loa(dx,dx) cov_loa(dx,dy); 
            cov_loa(dx,dy) cov_loa(dy,dy)];

end

function [p_c] = contact_distribution(loa,cov_loa,mean_loa)

    %Calculate Contact Point Distribution 
    %TODO: VISUALIZE AGAINST OBJECT
    p_c = loa(1,:); 
   
    for i = 1:size(loa,1)   
        p0 = mvnpdf(0,mean_loa(i),cov_loa(i,i)); 
        [mean_con,cov_con] = gauss_condition(mean_loa(1:i),cov_loa(1:i,1:i)); 
        
        P = 1-mvncdf(zeros(size(mean_con)),mean_con,cov_con);
        p_c(i) = p0*P;
        
    end
    
  
    p_c = p_c/norm(p_c,1);
    plot(loa(:,1),p_c); 
end

function [nu_mean,nu_cov] = gauss_condition(mean,cov)
    
    mu_b = mean(end);
    mu_a = mean(1:end-1); 
    
    sigma_b = cov(end,end); 
    sigma_a = cov(1:end-1,1:end-1); 
    sigma_c = cov(1:end-1,end); 
    
    nu_mean = mu_b + sigma_c/sigma_b*(0-mu_b); 
    
    nu_cov = sigma_a-sigma_c/sigma_b*sigma_c';
   
end
function [loa,norms,p_c,p_n] = Compute_Distributions(  gpModel,shapeParams,grip_point,img)
%COMPUTE_DISTRIBUTIONS Summary of this function goes here
%   Detailed explanation goes here
    
    loa = compute_loa(grip_point); 

    %Calculate Distribution Along Line 
    cov_loa = gp_cov(gpModel,loa, [], true);

    mean_loa = gp_mean(gpModel,loa,true);
   
    %Calculate Distribution Along In Workspace
    cov = gp_cov(gpModel,shapeParams.all_points,[],true); 
    mean = gp_mean(gpModel,shapeParams.all_points,true); 
    
    cov_wksp = cov(1:shapeParams.gridDim,1:shapeParams.gridDim); 
    mean_wksp = mean(1:shapeParams.gridDim); 
    
    
%   %Compute Center of Mass and Plot 
%   p_com = center_of_mass(mean,cov,shapeParams.gridDim);
%   plot_com(p_com,shapeParams.com,shapeParams.gridDim,img.mean)
    
    %Compute Contact Distribution and Plot 
    p_c = contact_distribution(loa,cov_loa,mean_loa);
    plot_contact(p_c,grip_point,loa,img.mean);
    
    %Compute Normals Distribution and Plot 
    [p_n, x,y] = normal_distribution(loa,cov_loa,mean_loa,p_c);
    norms = [x' y']; 
    plot_normal(p_n,grip_point,x,y,img.mean,loa)
    %plot(loa(:,1),mean_loa(1:size(loa,1)));

end

function [dist] = center_of_mass(mean,cov,gridDim)
    
    for i=1:gridDim
        for j=1:gridDim
            dist(i,j) = mvncdf(0,mean(gridDim*(i-1)+j,:),cov(gridDim*(i-1)+j,gridDim*(i-1)+j));
        end
    end

    %Normalize the grid 
    sm = 0; 
    for i=1:gridDim
        sm = sm+sum(dist(i,:)); 
    end
    
    dist = dist/sm; 
    
end



function [p] = plot_com(dist,com,gridDim,testImage)
    
    figure; 
   
    imshow(testImage);
    axis on;
    
    hold on     
    plot(2*com(:,1),2*com(:,2),'x','MarkerSize',10)
    title('Mean Function of GPIS'); 
    hold off
    figure;
    h = surfc([1:gridDim],[1:gridDim],dist,dist); 
    %set(h,'edgecolor','interp')
    
    title('Distribution on Density'); 
    colorbar
    xlabel('x-axis'); 
    ylabel('y-axis'); 
    zlabel('pdf');
    
    
end

function [p] = plot_contact(dist,point,loa,testImage)
    
    figure; 
   
    imshow(testImage);
    set(gca,'YDir','normal');
    axis on;
    hold on     
  
    scatter(2*loa(:,1),2*loa(:,2),10,dist);
   
    plot(2*loa(1,1),2*loa(1,2),'xg','MarkerSize',10);
  
    
    title('Mean Function of GPIS'); 
    xlabel('x-axis (2X)'); 
    ylabel('y-axis (2X)'); 
    hold off
    
    figure;
    scatter(1:size(dist,1),dist);
    plot(dist);
    title('Distribution on Contact Points'); 
    xlabel('t'); 
    ylabel('pdf'); 
    
 
    
    
    
end

function [p] = plot_normal(dist,point,x,y,testImage,loa)
    
 

%     imshow(testImage);
%     set(gca,'YDir','normal');
%     axis on
%     hold on     
%     plot(2*loa(:,1),2*loa(:,2));
%     plot(2*loa(1,1),2*loa(1,2),'xg','MarkerSize',10);
%     
%     title('Mean Function of GPIS');
%     xlabel('x-axis (2X)'); 
%     ylabel('y-axis (2X)'); 
%     hold off
    
    figure;
    %h = surf(x,y,dist); 
    %set(h,'edgecolor','flat')
    scatter(-x,-y,10,dist);
    axis([-1 1 -1 1]); 
    axis equal
    colorbar
    title('Distribution on Surface Normals'); 
    xlabel('x-direction'); 
    ylabel('y-direction'); 
    zlabel('pdf');
    
    
end

function [loa] = compute_loa(grip_point)
%Calculate Line of Action given start and end point

    %step_size = 0.5; 
    step_size = 0.3; 
    
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

end


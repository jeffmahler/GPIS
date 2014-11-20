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
   
    
    %Compute Contact Distribution and Plot 
    p_c = contact_distribution(loa,cov_loa,mean_loa);
    plot_contact(p_c,grip_point,loa,img.mean);
    
    %Compute Normals Distribution and Plot 
    [p_n, x,y] = normal_distribution(loa,cov_loa,mean_loa,p_c);
    norms = [x' y']; 
    plot_normal(p_n,grip_point,x,y,img.mean,loa,shapeParams.com)
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
    plot(dist,'LineWidth', 4); 
    %set(gca, 'TickLength', [0.025, 0.0])
    set(gca,'FontSize',12)
    font = 18;
    title('Distribution on Contact Points','FontSize', font); 
    xlabel('t','FontSize', font); 
    ylabel('pdf','FontSize', font); 
    
    
end

function [p] = plot_normal(dist,point,x,y,testImage,loa,com)
    
    figure; 

    imshow(testImage);
    set(gca,'XDir','normal');
    axis on
    hold on     
    plot(2*loa(:,1),2*loa(:,2));
    plot(2*loa(1,1),2*loa(1,2),'xg','MarkerSize',10);
    %plot(2*com(:,1),2*com(:,2),'x','MarkerSize',10)
   
    title('Mean Function of GPIS');
    xlabel('x-axis (2X)'); 
    ylabel('y-axis (2X)'); 
    hold off
    
    
    dist = dist*50; 
    dt = 0.01; 
    x = x';
    y = -y';
    
    Values = []; 
    for i=1:size(x,1)
        Values = [Values; [x(i,:) y(i,:)]]; 
        if(dist(i) ~=0)
            grad = [x(i,:) y(i,:)]
            point = grad; 
            while abs(point) < abs(grad+dist(i)*grad)
                Values = [Values; point]; 
                point = point + grad*dt;
            end
        end
    end
    
    figure;
    %h = surf(x,y,dist); 
    %set(h,'edgecolor','flat')
    
    hold on; 
    scatter(Values(:,1),Values(:,2),40);
    scatter(x,y,40,'k');
    hold off; 
    
    [vx idx] = max(abs(Values(:,1)));
    [vy idx] = max(abs(Values(:,2))); 
    axis([-vx-0.25 vx+0.25 -vy-0.25 vy+0.25]); 
    axis equal
  
    font = 18
    set(gca,'FontSize',12)
    title('Distribution on Surface Normals','FontSize', font); 
    xlabel('x-direction','FontSize', font); 
    ylabel('y-direction','FontSize', font); 
    zlabel('pdf');
    
    
end




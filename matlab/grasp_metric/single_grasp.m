function [ output_args ] = single_grasp(  gpModel,shapeParams,cp,img,experimentConfig)
%SINGLE_GRASP Summary of this function goes here
%   Detailed explanation goes here
% 
[loa_1,Norms,pc_1,pn_1] = Compute_Distributions(gpModel,shapeParams,cp(1:2,:),img);

% [loa_2,Norms,pc_2,pn_2] = Compute_Distributions(gpModel,shapeParams,cp(3:4,:),img);
% 
% ca  = atan(experimentConfig.frictionCoef);
% fc = experimentConfig.frictionCoef;
% plot_grasp(loa_1,loa_2,img.mean,pc_1,pc_2)
    
  
%Calculate Distribution Along In Workspace
cov = gp_cov(gpModel,shapeParams.all_points,[],true); 
mean = gp_mean(gpModel,shapeParams.all_points,true); 

%Compute Center of Mass and Plot 
p_com = center_of_mass(mean,cov,shapeParams.gridDim);
plot_com(p_com,shapeParams.com,shapeParams.gridDim,img.mean)


end

function [] = plot_hist(q_vals)
    qScale = 1; 
    figure;
    nbins = 100;
    [hst,centers] = hist(qScale*q_vals);
    hist(qScale*q_vals, qScale*(-0.1:0.0025:0.1));

    title('Histogram of Grasp Quality'); 
    xlim(qScale*[-0.1, 0.1]);
    xlabel('Grasp Quality'); 
    ylabel('Count');
end


function [] = plot_grasp(loa_1,loa_2,testImage,pc_1,pc_2)
    
    figure;
    testImage = cat(3, testImage, testImage, testImage);
    imshow(testImage);
    set(gca,'XDir','normal');
    axis on;
    hold on     
  
    scatter(2*loa_2(:,1),2*loa_2(:,2),10,pc_2); 
    plot(2*loa_2(1,1),2*loa_2(1,2),'xg','MarkerSize',10);
  
    scatter(2*loa_1(:,1),2*loa_1(:,2),10,pc_1);
    plot(2*loa_1(1,1),2*loa_1(1,2),'xr','MarkerSize',10);
    
    colorbar
    
    title('Mean Function of GPIS'); 
    xlabel('x-axis (2X)'); 
    ylabel('y-axis (2X)'); 
    hold off

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
    set(h,'edgecolor','interp')
    font = 18; 
    set(gca,'FontSize',12)
    title('Distribution on Density','FontSize', font);  
    colorbar
    xlabel('x-axis','FontSize', font); 
    ylabel('y-axis','FontSize', font);  
    zlabel('pdf','FontSize', font); 
    set(gcf,'color',[1 1 1]);
  
    
end
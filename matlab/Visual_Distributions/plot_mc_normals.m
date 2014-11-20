function [divg] = plot_mc_normals( Norms,normals_grasp,normals_shape)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    dist_grasp = zeros(size(Norms,1),1); 
    dist_shape = zeros(size(Norms,1),1);

    for i=1:size(normals_grasp,1)
        n =normals_grasp(i,:);
        n = n/norm(n,2);
        idx = find_closest_index(n,Norms);
        dist_grasp(idx) = dist_grasp(idx)+1; 
        
        n =normals_shape(i,:);
        n = n/norm(n,2);
        idx = find_closest_index(n,Norms);
        dist_shape(idx) = dist_shape(idx)+1; 
    end


    dist_grasp = dist_grasp/norm(dist_grasp,1); 
    dist_shape = dist_shape/norm(dist_shape,1); 
    
    
   
    plot_normal(dist_grasp,Norms(:,1),Norms(:,2));
    plot_normal(dist_shape,Norms(:,1),Norms(:,2));
    
    divg = KL_divg(dist_grasp,dist_shape);

end


function [idx] = find_closest_index(v_array,table)

sub = [table(:,1)-v_array(1) table(:,2)-v_array(2)];

for i=1:size(sub,1)
    tmp(i) = norm(sub(i,:),2); 
end

[C idx] = min(tmp); 

end

function [d] =  KL_divg(h1,h2)
    %# you may want to do some input testing, such as whether h1 and h2 are
    %# of the same size

    %# create an index of the "good" data points
    goodIdx = h1>0 & h2>0; %# bin counts <0 are not good, either

    d1 = sum(h1(goodIdx) .* log(h1(goodIdx) ./h2(goodIdx)));
    d2 = sum(h2(goodIdx) .* log(h2(goodIdx) ./h1(goodIdx)));

    %# overwrite d only where we have actual data
    %# the rest remains zero
    d = (d1+d2)/2; 
end

function [] = plot_normal(dist,x,y)
   
    figure;
   
    dist = dist*50; 
    dt = 0.01; 
    x = x;
    y = -y;
    
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
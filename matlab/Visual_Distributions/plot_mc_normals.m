function [divg] = plot_mc_normals( loa,Norms,normals_emp,p_n )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    dist = zeros(size(p_n)); 


    for i=1:size(normals_emp,1)
        n =normals_emp(i,:);
        n = n/norm(n,2);
        idx = find_closest_index(n,Norms);
        dist(idx) = dist(idx)+1; 
    end


    dist = dist/norm(dist,1); 
    
    plot(dist); 
    
    divg = KL_divg(dist,p_n);
    plot_normal(dist,Norms(:,1),Norms(:,2));

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
   
    figure(5);
   
    scatter(-x,-y,10,dist);
    axis([-1 1 -1 1]); 
    axis equal
    colorbar
    title('Empirical Distribution on Surface Normals'); 
    xlabel('x-direction'); 
    ylabel('y-direction'); 
    zlabel('pdf');
    
    
end
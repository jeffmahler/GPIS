function [divg] = plot_mc_contact( loa,contact_emp,p_c )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    dist = zeros(size(loa,1),1); 


    for i=1:size(contact_emp,1)
        t = find_closest_index(loa,contact_emp(:,i);
        
        dist(t) = dist(t)+1; 
    end


    dist = dist/norm(dist,1); 
    
    plot(dist); 
    
    divg = KL_divg(dist,p_c);


end


function [idx] = find_closest_index(v_array,table)

sub = [table(:,1)-v_array(1) table(:,2)-v_array(2)];

for i=1:size(sub,1)
    tmp(i) = norm(sub(i,:),2); 
end

[C idx] = min(tmp); 

end

function KL_divg(h1,h2)
    %# you may want to do some input testing, such as whether h1 and h2 are
    %# of the same size

   

    %# create an index of the "good" data points
    goodIdx = h1>0 & h2>0; %# bin counts <0 are not good, either

    d1 = sum(h1(goodIdx) .* log(h1(goodIdx) . /h2(goodIdx)));
    d2 = sum(h2(goodIdx) .* log(h2(goodIdx) . /h1(goodIdx)));

    %# overwrite d only where we have actual data
    %# the rest remains zero
    d = d1+d2; 
end


function [ contacts,norm ] = find_contact_points(contact_points,nc,allPoints,allTsdf,allNorm,COM,thresh )
%Given sampled Tsdf finds the contact points, which are the intersection of
%lines of actions with the 0-level crossing 
gridDM = max(allPoints(:,1)); 



for i=1:nc
    loa = compute_loa(contact_points(i:i+1,:)); 
   
    for t =1:size(loa,2)
        if(thresh <= allTsdf(gridDM*(loa(i,1)-1)+loa(i,2)))
            contacts(:,i) = loa(i,:);
            norm(:,i) = allNorm(gridDM*(loa(i,1)-1)+loa(i,2),:)'; 
            break;
        end
    end

end
end

function [loa] = compute_loa(grip_point)
%Calculate Line of Action given start and end point

    step_size = 2; 

    start_point = grip_point(1,:); 
    end_p = grip_point(2,:); 

    grad = end_p-start_point; 
    end_time = norm(grad, 2)
    grad = grad/end_time; 
    i=1; 
    time = 0;

    while(time < end_time)
        point = start_point + grad*time;
        loa(i,:) = ceil(point);
        time = time + step_size; 
        i = i + 1;
    end
  
end
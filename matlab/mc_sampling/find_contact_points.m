function [ contacts,norm, bad ] = ...
    find_contact_points(contact_points, nc, allPoints, allTsdf, allNorm, ...
        COM, thresh, vis)
%Given sampled Tsdf finds the contact points, which are the intersection of
%lines of actions with the 0-level crossing

if nargin < 8
    vis = true;
end

gridDM = max(allPoints(:,1)); 
contacts = zeros(2, nc);
norm = zeros(2, nc);
dim = uint16(sqrt(size(allPoints,1)));

for i=1:nc
    index = 2*(i-1) + 1;
    loa = compute_loa(contact_points(index:index+1,:));
   
    tsdfVal = 10;
    if loa(1,1) > 0 && loa(1,2) > 0 && loa(1,1) <= dim && loa(1,2) <= dim
        tsdfVal = allTsdf(gridDM*(loa(1,1)-1)+loa(1,2)); 
    end
    for t =1:size(loa,1)
        prevTsdfVal = tsdfVal;
        if loa(t,1) > 0 && loa(t,2) > 0 && loa(t,1) <= dim && loa(t,2) <= dim
            tsdfVal = allTsdf(gridDM*(loa(t,1)-1)+loa(t,2));
        end
        if vis
            figure(10);
            scale = 5;
            scatter(scale*loa(t,1), scale*loa(t,2), 50.0, 'x', 'LineWidth', 1.5);
            hold on;
            scatter(scale*COM(1), scale*COM(2), 50.0, '+', 'LineWidth', 2);
        end
        if(abs(tsdfVal) < thresh || (sign(prevTsdfVal) ~= sign(tsdfVal)) )
            contacts(:,i) = loa(t,:)';
            norm(:,i) = allNorm(gridDM*(loa(t,1)-1)+loa(t,2), :)'; 
            break;
        end
        
    end

end

bad = false;
if sum(contacts(:,1) == zeros(2,1)) == 2
    %fprintf('Bad contacts!\n');
    bad = true;
end
if sum(contacts(:,2) == zeros(2,1)) == 2
    %fprintf('Bad contacts!\n');
    bad = true;
end
end

function [loa] = compute_loa(grip_point)
%Calculate Line of Action given start and end point

    step_size = 1; 

    start_point = grip_point(1,:); 
    end_p = grip_point(2,:); 

    grad = end_p-start_point; 
    end_time = norm(grad, 2);
    grad = grad/end_time; 
    i=1; 
    time = 0;

    while(time < end_time)
        point = start_point + grad*time;
        loa(i,:) = round(point);
        time = time + step_size; 
        i = i + 1;
    end
  
end
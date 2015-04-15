function [ contacts, normal, bad ] = ...
    find_contact_points(contactPoints, nc, allPoints, allTsdf, allNorm, ...
        COM, thresh, vis, plateWidth, scale)
%Given sampled Tsdf finds the contact points, which are the intersection of
%lines of actions with the 0-level crossing

if nargin < 8
    vis = true;
end
if nargin < 9
   plateWidth = 1; 
end
if nargin < 10
   scale = 5; 
end

gridDim = max(allPoints(:,1)); 
contacts = zeros(2, nc);
normal = zeros(2, nc);
dim = uint16(sqrt(size(allPoints,1)));
tsdfGrid = reshape(allTsdf, gridDim, gridDim);
xNormGrid = reshape(allNorm(:,1), gridDim, gridDim);
yNormGrid = reshape(allNorm(:,2), gridDim, gridDim);

% find the line tangent to the loa
loaDir = contactPoints(1,:)' - contactPoints(2,:)';
tangentDir = [-loaDir(2,1); loaDir(1,1)];
tangentDir = tangentDir / norm(tangentDir);

% for each direction in the loa compute the contact location
for i=1:nc
    index = 2*(i-1) + 1;
    loa = compute_loa(contactPoints(index:index+1,:));
   
    tsdfVal = 10;
    if loa(1,1) > 0 && loa(1,2) > 0 && loa(1,1) <= dim && loa(1,2) <= dim
        tsdfVal = tsdfGrid(loa(1,2), loa(1,1)); 
    end
    for t = 1:size(loa,1)
        % visualize the region
        if vis
            figure(10);
            scatter(scale*loa(t,1), scale*loa(t,2), 50.0, 'x', 'LineWidth', 1.5);
            hold on;
            scatter(scale*COM(1), scale*COM(2), 50.0, '+', 'LineWidth', 2);
        end
        
        % tsdf grid 
        prevTsdfVal = tsdfVal;
        if loa(t,1) > 0 && loa(t,2) > 0 && loa(t,1) <= dim && loa(t,2) <= dim
            tsdfVal = tsdfGrid(loa(t,2), loa(t,1));
        end
        
        % check whether the gripper has contacted the surface
        if(abs(tsdfVal) < thresh || (sign(prevTsdfVal) ~= sign(tsdfVal)) )
            contacts(:,i) = loa(t,:)';
            normal(:,i) = [xNormGrid(loa(t,2), loa(t,1));...
                         yNormGrid(loa(t,2), loa(t,1))]; 
            break;
        end
        
        % check other points along the width of the gripper
        exitLoop = false;
        for k = 1:plateWidth
            % positive along tangent dir
            gripPoint = loa(t,:)' + double(k)*tangentDir;
            gripPoint = round(gripPoint);
            if vis
                figure(10);
                scatter(scale*gripPoint(1), scale*gripPoint(2), 50.0, 'o', 'LineWidth', 1.5);
            end
            
            if gripPoint(1) > 0 && gripPoint(2) > 0 && gripPoint(1) <= dim && gripPoint(2) <= dim
                plateTsdfVal = tsdfGrid(gripPoint(2), gripPoint(1));
            
                % check contact
                if(abs(plateTsdfVal) < thresh || (sign(prevTsdfVal) ~= sign(plateTsdfVal)) )
                    contacts(:,i) = gripPoint;
                    normal(:,i) = [xNormGrid(gripPoint(2), gripPoint(1));...
                                yNormGrid(gripPoint(2), gripPoint(1))]; 
                    exitLoop = true;
                    break;
                end
            end
            
            % negative along tangent dir
            gripPoint = loa(t,:)' - double(k)*tangentDir;
            gripPoint = round(gripPoint);
            if vis
                figure(10);
                scatter(scale*gripPoint(1), scale*gripPoint(2), 50.0, 'o', 'LineWidth', 1.5);
            end
            
            if gripPoint(1) > 0 && gripPoint(2) > 0 && gripPoint(1) <= dim && gripPoint(2) <= dim
                plateTsdfVal = tsdfGrid(gripPoint(2), gripPoint(1));
            
                % check contact
                if(abs(plateTsdfVal) < thresh || (sign(prevTsdfVal) ~= sign(plateTsdfVal)) )
                    contacts(:,i) = gripPoint;
                    normal(:,i) = [xNormGrid(gripPoint(2), gripPoint(1));...
                                yNormGrid(gripPoint(2), gripPoint(1))]; 
                    exitLoop = true;
                    break;
                end
            end
        end
        
        % use grip width
        if exitLoop
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

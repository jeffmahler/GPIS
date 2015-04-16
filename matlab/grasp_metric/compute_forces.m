function [forces, normal, forces_failed] = compute_forces(contacts, Gx, Gy, Gz, ...
    friction_coef, n_cone_faces)
%COMUTE_FORCES Summary of this function goes here
%   Detailed explanation goes here

if nargin < 6    
    n_cone_faces = 10;
end

n_contacts = size(contacts, 1);
delta_face = 1.0 / n_cone_faces;
forces_failed = false;
cone_angle = atan(friction_coef);

forces = cell(1, n_contacts);

for i = 1:n_contacts
    index = 1;
    forces{i} = zeros(3,1);
    contact = round(contacts(i,:));
    grad = [Gy(contact(1), contact(2), contact(3)), ...
        Gx(contact(1), contact(2), contact(3)), ...
        Gz(contact(1), contact(2), contact(3))];
    
    % find the unit direction of the normal force 
    if sum(abs(grad)) == 0
        forces_failed = true;
        break;
    end
    force = grad' / norm(grad);
    normal = -force;

    % get extrema of friction cone, negative to point into object
    [U, S, V] = svd(force); % used to get tangent plane to object
    tan_plane = U(:,2:3);
    tan_len = tan(cone_angle);
    
    for t = 0:delta_face:1
        tan_dir = t * tan_plane(:,1) + (1 - t) * tan_plane(:,2);
        tan_vec = tan_len * tan_dir / norm(tan_dir);
            
        forces{i}(:,index) = -(force + tan_vec); 
        index = index+1;
        forces{i}(:,index) = -(force - tan_vec);
        index = index+1;
    end

    for t = (0 + delta_face):delta_face:(1 - delta_face)
        tan_dir = t * tan_plane(:,1) - (1 - t) * tan_plane(:,2);
        tan_vec = tan_len * tan_dir / norm(tan_dir);
 
        forces{i}(:,index) = -(force + tan_vec); 
        index = index+1;
        forces{i}(:,index) = -(force - tan_vec);
        index = index+1;
    end
end

end


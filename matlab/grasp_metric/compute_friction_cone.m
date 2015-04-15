function [cone_support, n, forces_failed] = compute_friction_cone(x_contact, Gx, Gy, Gz, ...
    friction_coef, n_cone_faces)
% COMPUTE_FORCE

delta_face = 1.0 / n_cone_faces;
forces_failed = false;
cone_angle = atan(friction_coef);

% set up gradients, etc
index = 1;
cone_support = zeros(3,1);
contact = round(x_contact);
grad = [Gy(contact(1), contact(2), contact(3)), ...
        Gx(contact(1), contact(2), contact(3)), ...
        Gz(contact(1), contact(2), contact(3))];
n = grad' / norm(grad);

% find the unit direction of the normal force 
if sum(abs(grad)) == 0
    forces_failed = true;
    return;
end
force = grad' / norm(grad);

% get extrema of friction cone, negative to point into object
[U, S, V] = svd(force); % used to get tangent plane to object
tan_plane = U(:,2:3);
tan_len = tan(cone_angle);

for t = 0:delta_face:1
    tan_dir = t * tan_plane(:,1) + (1 - t) * tan_plane(:,2);
    tan_vec = tan_len * tan_dir / norm(tan_dir);

    cone_support(:,index) = -(force + tan_vec); 
    index = index+1;
    cone_support(:,index) = -(force - tan_vec);
    index = index+1;
end

for t = (0 + delta_face):delta_face:(1 - delta_face)
    tan_dir = t * tan_plane(:,1) - (1 - t) * tan_plane(:,2);
    tan_vec = tan_len * tan_dir / norm(tan_dir);

    cone_support(:,index) = -(force + tan_vec); 
    index = index+1;
    cone_support(:,index) = -(force - tan_vec);
    index = index+1;
end

end


function [contacts, contact_found] = ...
    antipodal_grasp_contacts(x1, grasp_axis, sdf, grasp_width, surface_thresh)

% set up variables
sdf_dims = size(sdf);
v = - grasp_axis / norm(grasp_axis);
c = x1 + grasp_width * grasp_axis / norm(grasp_axis);

d = size(x1, 2);
c = max(ones(1, d), min(c, sdf_dims));
prev_c = c;

cur_point = round(c);
if d == 3
    cur_sdf_val = sdf(cur_point(1), cur_point(2), cur_point(3));
else
    cur_sdf_val = sdf(cur_point(1), cur_point(2));
end
contact_found = false;
t = 1;

contacts = [x1; zeros(1,ndims(sdf))];

while sum(cur_point < 1) == 0 && sum(cur_point > sdf_dims) == 0 && ~contact_found
    prev_sdf_val = cur_sdf_val;
    
%     figure(2);
%     scatter3(c(1), c(2), c(3), 50, 'c', 'MarkerFaceColor', 'c');
%     hold on;
    
    % get new sdf val
    if d == 3
        cur_sdf_val = sdf(cur_point(1), cur_point(2), cur_point(3));
    else
        cur_sdf_val = sdf(cur_point(1), cur_point(2));
    end

    % look for sign change
    % sign(cur_sdf_val) ~= sign(prev_sdf_val)
    if abs(cur_sdf_val) <= surface_thresh && norm(c - x1) < grasp_width
        % compute the zero crossing (assuming linear sdf locally)
        estimated_sdf_val = prev_sdf_val;
        estimated_c = prev_c;
        delta = 0.01;
        t = delta;
        while abs(estimated_sdf_val) > 0.001 && norm(c - estimated_c) > 0.001 
            estimated_c = (1 - t) * prev_c + t * c;
            estimated_sdf_val = (1 - t) * prev_sdf_val + t * cur_sdf_val;
            t = t + delta;
        end
        
        contacts(2,:) = estimated_c;
        contact_found = true;
    end

    prev_c = c;
    c = c + v;
    cur_point = round(c);

    t = t+1;
end

end


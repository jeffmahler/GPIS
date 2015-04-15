function [contacts, contact_found] = ...
    antipodal_grasp_contacts(x1, grasp_axis, sdf, grasp_width)

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

    % get new sdf val
    if d == 3
        cur_sdf_val = sdf(cur_point(1), cur_point(2), cur_point(3));
    else
        cur_sdf_val = sdf(cur_point(1), cur_point(2));
    end

    % look for sign change
    if sign(cur_sdf_val) ~= sign(prev_sdf_val) && norm(c - x1) < grasp_width
        contacts(2,:) = prev_c;
        contact_found = true;
    end

    prev_c = c;
    c = c + v;
    cur_point = round(c);

    t = t+1;
end

end


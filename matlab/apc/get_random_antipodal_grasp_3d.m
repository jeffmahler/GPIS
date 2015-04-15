function [contacts, contact_found] = ...
    get_random_antipodal_grasp_3d(sdf, grasp_width, perturb_scale)

sdf_dims = size(sdf);

if nargin < 2
    grasp_width = max(sdf_dims);
end
if nargin < 3
    perturb_scale = 0.1;
end

[~, surf_points, ~] = compute_tsdf_surface(sdf);
[Gx, Gy, Gz] = gradient(sdf);
centroid = mean(surf_points);

ind = randsample(size(surf_points,1) ,1);
x1 = surf_points(ind,:);

% get opposite gradient
n1 = [Gy(x1(1), x1(2), x1(3)), ...
      Gx(x1(1), x1(2), x1(3)), ...
      Gz(x1(1), x1(2), x1(3))];
n1 = -n1 / norm(n1);

% get dir to com
z1 = centroid - x1;
z1 = z1 / norm(z1);

% search dir (unit norm)
if rand() < 0.8
    v = n1;
else
    v = z1;
end
v = v + perturb_scale * rand(1, 3); % add small perturbations to the direction
v = v / norm(v);

c = x1 + v;
cur_point = round(c);
cur_sdf_val = sdf(cur_point(1), cur_point(2), cur_point(3));
contact_found = false;
t = 1;

contacts = [x1; zeros(1,ndims(sdf))];
    
while sum(cur_point < 1) == 0 && sum(cur_point > sdf_dims) == 0 && ~contact_found
    prev_sdf_val = cur_sdf_val;
    
    % get new sdf val
    if size(sdf, 3) > 1
        cur_sdf_val = sdf(cur_point(1), cur_point(2), cur_point(3));
    else
        cur_sdf_val = sdf(cur_point(2), cur_point(1));
    end

    % look for sign change
    if sign(cur_sdf_val) ~= sign(prev_sdf_val) && norm(c - x1) < grasp_width
        contacts(2,:) = c;
        contact_found = true;
    end
    
    c = c + v;
    cur_point = round(c);
    
    t = t+1;
end

end


function [in_cone, dir_angle] = within_cone(cone_support, n, v)

in_cone = false;
v = v / norm(v);
angles = acos(n' * cone_support);
max_angle = max(angles);
dir_angle = acos(n' * v);
if dir_angle <= max_angle
    in_cone = true;
end

end


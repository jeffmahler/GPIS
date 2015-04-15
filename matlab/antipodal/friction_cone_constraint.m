function val = friction_cone_constraint(x, gpModel, frictionCoef, shapeParams, ...
    gripWidth, plateWidth, com, comTol)

use_com = true;
if nargin < 7
    use_com = false;
end
if nargin < 8
    comTol = 1;
end

% right now just the surface values
%d = size(x,1) / 2;
val = zeros(3,1);
%xp = [x(1:d,1)'; x(d+1:2*d,1)'];

nc = 2;
loa = create_ap_loa(x, gripWidth);
fakeCom = [0,0];
vis = false;
[xp, ~, ~ ] = ...
    find_contact_points(loa, nc, shapeParams.points, ...
        shapeParams.tsdf, shapeParams.normals, ...
        fakeCom, shapeParams.surfaceThresh, vis, plateWidth);
xp = xp';

% lies on surface
[mu, Mx, Kxxp] = gp_mean(gpModel, xp, true);    

% antipodality
% opposite normals
grad_1 = [mu(3); mu(5)];
grad_2 = [mu(4); mu(6)];

% check line between points is in friction cone (gives force closure)
diff = xp(1,:)' - xp(2,:)';
if norm(diff) > 0 && norm(grad_1) > 0
    val(1) = cos(atan(frictionCoef)) - diff' * grad_1 / (norm(diff) * norm(grad_1));
else
    val(1) = cos(atan(frictionCoef));
end
if norm(diff) > 0 && norm(grad_2) > 0
    val(2) = cos(atan(frictionCoef)) - (-diff)' * grad_2 / (norm(diff) * norm(grad_2));
else
    val(2) = cos(atan(frictionCoef));
end
val = 1.75*val;
% soft center of mass constraint (since the hard constraint is very hard to
% satisfy) but only use when flagged
if use_com
    n = diff / norm(diff);
    v = xp(2,:)' - com';
 %   val(3) = norm(v - (v'*n) * n) - comTol; % abs(n'*com' + n'*xp(1,:)') / norm(n);
end

%val

end


function val = friction_cone_constraint(x, gpModel, frictionCoef, com, comTol)

use_com = true;
if nargin < 4
    use_com = false;
end

% right now just the surface values
d = size(x,1) / 2;
val = zeros(3,1);
xp = [x(1:d,1)'; x(d+1:2*d,1)'];

% lies on surface
[mu, Mx, Kxxp] = gp_mean(gpModel, xp, true);    

% antipodality
% opposite normals
grad_1 = [mu(3); mu(5)];
grad_2 = [mu(4); mu(6)];

% check line between points is in friction cone (gives force closure)
diff = xp(1,:)' - xp(2,:)';
val(1) = cos(atan(frictionCoef)) - diff' * grad_1 / (norm(diff) * norm(grad_1));
val(2) = cos(atan(frictionCoef)) - (-diff)' * grad_2 / (norm(diff) * norm(grad_2));

% soft center of mass constraint (since the hard constraint is very hard to
% satisfy)
if use_com
    n = diff / norm(diff);
    v = xp(2,:)' - com';
 %   val(3) = norm(v - (v'*n) * n) - comTol; % abs(n'*com' + n'*xp(1,:)') / norm(n);
end

%val

end


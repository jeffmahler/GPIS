function val = surface_and_antipodality_functions(x, gpModel, com)
% 5 constraints evaluated here:
%   1,2: Points on surface
%   3,4: Opposite normals
%   5: Center of mass


use_com = true;
if nargin < 3
    use_com = false;
end

% right now just the surface values
d = size(x,1) / 2;
val = zeros(5,1);
xp = [x(1:d,1)'; x(d+1:2*d,1)'];

% lies on surface
[mu, Mx, Kxxp] = gp_mean(gpModel, xp, true);    
val(1:2) = 2*mu(1:2);

% antipodality
% opposite normals
grad_1 = [mu(3); mu(5)];
grad_2 = [mu(4); mu(6)];
val(3:4) = (grad_1 + (norm(grad_2) / norm(grad_1)) * grad_2);

% tangents (enforced softly right now)
diff = xp(1,:)' - xp(2,:)';
% val(5) = diff' * grad_1 - norm(diff) * norm(grad_1);
% val(6) = -diff' * grad_2 - norm(diff) * norm(grad_2);

% center of mass alignment
if use_com
    n = diff / norm(diff);
    v = xp(2,:)' - com';
%    val(5) = norm(v - (v'*n) * n); % abs(n'*com' + n'*xp(1,:)') / norm(n);
end

% xp
% val
% n
% com

end


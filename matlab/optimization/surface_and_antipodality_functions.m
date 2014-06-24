function val = surface_and_antipodality_functions(x, gpModel)

% right now just the surface values
d = size(x,1) / 2;
val = zeros(3,1);
xp = [x(1:d,1)'; x(d+1:2*d,1)'];

% lies on surface
[mu, Mx, Kxxp] = gp_mean(gpModel, xp, true);    
val(1:2) = mu(1:2);

% antipodality
% opposite normals
grad_1 = [mu(3); mu(5)];
grad_2 = [mu(4); mu(6)];
val(3:4) = grad_1 + (norm(grad_2) / norm(grad_1)) * grad_2;

% tangents
diff = xp(1,:)' - xp(2,:)';
val(5) = diff' * grad_1 - norm(diff) * norm(grad_1);
val(6) = -diff' * grad_2 - norm(diff) * norm(grad_2);
val

end


function val = surface_and_antipodality_functions(x, gpModel)

% right now just the surface values
d = size(x,1) / 2;
xp = [x(1:d,1)'; x(d+1:2*d,1)'];
Kxxp = feval(gpModel.covFunc{:}, gpModel.hyp.cov, gpModel.training_x, xp);    
val = Kxxp' * gpModel.alpha;

% TODO antipodality

end


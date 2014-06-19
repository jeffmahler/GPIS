function grad = numerical_jac(f,x)
% numerical gradient and diagonal hessian

y = f(x);

grad = zeros(length(y), length(x));

eps = 1e-5;
xp = x;


for i=1:length(x)
	xp(i) = x(i) + eps/2;
	yhi = f(xp);
	xp(i) = x(i) - eps/2;
	ylo = f(xp);
	xp(i) = x(i);
	grad(:,i) = (yhi - ylo) / eps;
end

end
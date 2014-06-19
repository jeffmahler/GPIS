function mean = linear_mean_derivative(meanfunc, hyp, x)
%LINEAR_MEAN_DERIVATIVE Linear mean function w derivatives

M = size(x, 1);
D = size(x, 2);

mean = zeros(M + D*M, 1);
mean(1:M, 1) = feval(meanfunc{:}, hyp, x);

start_I = M+1;
end_I = 2*M;
for d = 1:D
    mean(start_I:end_I, 1) = hyp(d);
    start_I = end_I + 1;
    end_I = end_I + M;
end


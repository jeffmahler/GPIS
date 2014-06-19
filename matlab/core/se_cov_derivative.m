function K = se_cov_derivative(covfunc, hyp, beta, x, y)
%SE_COV_DERIVATIVE Squared exponential covariance function derivatives

if nargin < 5
    y = x;
end

% get sizes of arrays, initialize kernel matrix
M = size(x, 1);
N = size(y, 1);
D = size(x, 2);
P = M + D*M;
Q = N + D*N;
K = zeros(P, Q);

% regular ol covariance
tic;
K(1:M, 1:N) = feval(covfunc{:}, hyp, x, y) + beta * eye(M,N);

% disp('Cov done');
% toc

% dervative wrt first arg
start_I = M+1;
end_I = 2*M;
joint = [x; y];
for d = 1:D
    joint_mesh = meshgrid(joint(:,d));
    diff = joint_mesh - joint_mesh';
    K(start_I:end_I, 1:N) = diff(1:M,M+1:M+N) .* K(1:M, 1:N) / (exp(hyp(1))^2);
    start_I = end_I + 1;
    end_I = end_I + M;
%     disp('Grad 1 segment done');
%     toc
end

% disp('Grad 1 done');
% toc

% dervative wrt second arg
start_I = N+1;
end_I = 2*N;
for d = 1:D
    joint_mesh = meshgrid(joint(:,d));
    diff = (joint_mesh - joint_mesh')';
    K(1:M, start_I:end_I) = diff(1:M,M+1:M+N) .* K(1:M, 1:N) / (exp(hyp(1))^2);
    start_I = end_I + 1;
    end_I = end_I + N;
end

% disp('Grad 2 done');
% toc

% second dervative
start_I_N = N+1;
end_I_N = 2*N;
for di = 1:D
    start_I_M = M+1;
    end_I_M = 2*M;
    for dj = 1:D
        joint_mesh_i = meshgrid(joint(:,di));
        joint_mesh_j = meshgrid(joint(:,dj));
        diff_i = joint_mesh_i - joint_mesh_i';
        diff_j = joint_mesh_j - joint_mesh_j';
        sig_w_sq = exp(hyp(1))^2;
        K(start_I_M:end_I_M, start_I_N:end_I_N) = ...
            (sig_w_sq * (di == dj) * ones(M,N) - diff_i(1:M,M+1:M+N) .* diff_j(1:M,M+1:M+N)) ...
            .* K(1:M, 1:N) / sig_w_sq^2;
        start_I_M = end_I_M + 1;
        end_I_M = end_I_M + M;
    end
    start_I_N = end_I_N + 1;
    end_I_N = end_I_N + N;
end
% disp('Hess done');
% toc

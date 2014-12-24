function T = sample_pose_2d(num_samples,sigma, mu)
% SAMPLE_POSE sample pose with variance sigma and mean mu
% mu organized as follows:
%   [tx ty wx]

% default to unitary var, zero mean
if nargin < 1
    num_samples = 1; 
end

if nargin < 2
   sigma = eye(3); 
end
if nargin < 3
   mu = zeros(3,1); 
end
if size(sigma, 1) == 1
    sigma = sigma * eye(3);
end

T = cell(1, num_samples);
for i = 1:num_samples
    xi = mvnrnd(mu, sigma);

    t = xi(1:2)';
    phi = xi(3)';

    % compute skew symmetric form
    phi_ss = [      0, -phi(1);
               phi(1),       0];

    M = [    phi_ss, t;
         zeros(1,2), 0];

    T{i} = [t; phi];
end

if num_samples == 1
    T = T{1};
end
 
end


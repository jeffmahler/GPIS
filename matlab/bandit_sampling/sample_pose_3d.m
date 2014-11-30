function T = sample_pose_3d(sigma, mu)
% SAMPLE_POSE sample pose with variance sigma and mean mu
% mu organized as follows:
%   [tx ty tz wx wy wz]

% default to unitary var, zero mean
if nargin < 1
   sigma = eye(6); 
end
if nargin < 2
   mu = zeros(6,1); 
end

if size(sigma, 1) == 1
    sigma = sigma * eye(6);
end

xi = mvnrnd(mu, sigma);

t = xi(1:3)';
phi = xi(4:6)';

% compute skew symmetric form
phi_ss = [      0, -phi(3),  phi(2);
           phi(3),       0, -phi(1);
          -phi(2),  phi(1),      0];
 
M = [    phi_ss, t;
     zeros(1,3), 0];

T = expm(M);
 
end


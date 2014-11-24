function registration = register_2d_affine_unknown_corrs(source_tsdf, target_tsdf, ...
    penalties, stop_criteria, update_params, initial_params)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[height, width] = size(source_tsdf);
num_points = height*width;

[X, Y] = meshgrid(1:height, 1:width);
points = [X(:), Y(:)];

if nargin < 6
    initial_params = struct();
    initial_params.initial_temp = 10;
    
    initial_params.initial_tf = struct();
    initial_params.initial_tf.R = eye(2);
    initial_params.initial_tf.t = zeros(2,1);

    initial_params.initial_corrs = eye(num_points);
    
    initial_params.initial_delta = 1e10;
end

registration = struct();
registration.R = initial_params.initial_tf.R;
registration.t = initial_params.initial_tf.t;
registration.M = initial_params.initial_corrs;

T = initial_params.initial_temp;
delta = initial_params.initial_delta;

while T > stop_criteria.T
   while delta > stop_criteria.eps
       f = @(x) (registration.R * points' + registration.t);
       
       registration.M = update_corrs(points, f, penalties.Q);
       [registration.R, registration.t] = ...
           update_rigid_transform(points, registration.M);
       
   end
   T = shrink_temp * T;
end


    
end


function registration = register_2d_rigid_unknown_corrs(source_tsdf, target_tsdf, ...
    penalties, stop_criteria, update_params, initial_params)

[height, width] = size(source_tsdf);
num_points = height*width;

[X, Y] = meshgrid(1:height, 1:width);
points = [X(:), Y(:)];

if nargin < 6
    initial_params = struct();
    initial_params.initial_temp = 10.0;
    
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

figure(2);
subplot(1,2,1);
sdf_surface(source_tsdf, 1.0);
subplot(1,2,2);
sdf_surface(target_tsdf, 1.0);

figure(3);
subplot(1,2,1);
imagesc(source_tsdf);
subplot(1,2,2);
imagesc(target_tsdf);

T = initial_params.initial_temp;
k = 1;

while T > stop_criteria.T
   delta = initial_params.initial_delta;
   while delta > stop_criteria.eps && k < stop_criteria.max_iter
       f = @(x) (registration.R * x + registration.t);
       
       M_prev = registration.M;
       R_prev = registration.R;
       t_prev = registration.t;
       
       [registration.M, source_outliers, target_outliers] = ...
           update_corrs_sdf(points, source_tsdf, target_tsdf, ...
           f, T, penalties.Q);
       
       % plot the corespondences
       diff = zeros(height, width, 2);
       for j = 1:num_points
            if ~source_outliers(j)
                x_j = int16(points(j,:)');
                if x_j(2) == 2 && x_j(1) == 8
                    test = 1;
                end
                
                best_corr = find(registration.M(:,j) == max(registration.M(:,j)));
                y_i = int16(points(best_corr,:)');
                diff(x_j(2), x_j(1), :) = y_i - x_j;
            end
       end
       
       figure(3);
       subplot(1,2,1);
       imagesc(source_tsdf);
       hold on;
       quiver(diff(:,:,1), diff(:,:,2), 'LineWidth', 2.0);
       %quiver(ones(height, width), -ones(height,width), 'LineWidth', 2.0);
       subplot(1,2,2);
       imagesc(target_tsdf);
       
       [registration.R, registration.t] = ...
          solve_rigid_transform(points, registration.M, source_outliers, ...
            target_outliers);
       
        delta = norm(registration.t - t_prev);
        k = k+1;
   end
   T = update_params.shrink_temp * T;
end

% apply the rigid transformation?


end


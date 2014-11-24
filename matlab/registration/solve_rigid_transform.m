function [R, t] = solve_rigid_transform(points, M, source_outliers, target_outliers)

num_points = size(points,1);

y_hat = points';
x = points';

% for i = 1:num_points
%    row_sum = sum(M(i,:));
%    if ~target_outliers(i)
%        row_val = repmat(M(i,:), 2, 1) .* y_hat;
%        y_hat(:,i) = sum(row_val, 2) / row_sum;
%    end
% end

% compute centers
y_center = zeros(2,1);
y_count = 0;
x_center = zeros(2,1);
x_count = 0;
for i = 1:num_points
    if ~target_outliers(i)
        y_center = y_center + y_hat(:,i);
        y_count = y_count + 1;
    end
    if ~source_outliers(i)
        x_center = x_center + x(:,i);
        x_count = x_count + 1;
    end
end

y_center = y_center / y_count;
x_center = x_center / x_count;

y_center = repmat(y_center, 1, num_points);
x_center = repmat(x_center, 1, num_points);

% compute covariance 
H = zeros(2,2);
y_hat = y_hat - y_center;
x = x - x_center;

for j = 1:num_points
    if ~source_outliers(j)
        indices = find(M(:,j) > 1e-3);
        if size(indices,2) > 0
            for i = 1:size(indices,1)
                %if ~target_outliers(indices(i))
                    H = H + M(indices(i),j) * y_hat(:,indices(i)) * x(:,j)';
                %end
            end
        end
    end
end

[U, S, V] = svd(H);
R = V * U';
if det(R) < 0
    V(:,2) = -V(:,2);   
    R = V * U';
end

t = y_center(:,1) - R * x_center(:,1);


end


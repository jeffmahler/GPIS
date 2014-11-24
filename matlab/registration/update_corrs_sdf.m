function [M, source_outliers, target_outliers] = ...
    update_corrs_sdf(points, source_tsdf, target_tsdf, f, T, Q, trunc)

if nargin < 7
   trunc = 10; 
end

[height, width] = size(source_tsdf);
num_points = height*width;
M = zeros(num_points);
win = 2;
eps = 1e-2;

% not really that fast
for j = 1:num_points
    x_j = points(j,:)';
    x_j_target = f(x_j);
        
    k = uint16(x_j_target(1));
    l = uint16(x_j_target(2));
    
    k_low = max(k - win, 1);
    k_high = min(k + win, width);
    l_low = max(l - win, 1);
    l_high = min(l + win, height);
    
    [Xn, Yn] = meshgrid(k_low:k_high, l_low:l_high);
    neighbors = [Xn(:), Yn(:)]; 
    
    if j == 212
       test = 1; 
    end
    
    penalties = zeros(size(neighbors,1),1);
    indices = zeros(size(neighbors,1),1);
    for z = 1:size(neighbors,1)
        u = neighbors(z,:);
        i = u(2) + ((u(1) - 1) * height);
        
        indices(z) = i;
        y_i = points(i,:)';
       
        t_val = target_tsdf(y_i(2), y_i(1));
        s_val = source_tsdf(x_j(2), x_j(1));
        
        if abs(s_val) < trunc - eps % && abs(t_val) < trunc - eps
            point_diff = norm(y_i - x_j_target);
            tsdf_diff = t_val - s_val;
            point_penalty = exp((-1.0/(2*T)) * point_diff^2);
            tsdf_penalty = exp((-Q/(2*T)) * tsdf_diff^2);
            penalty = point_penalty * tsdf_penalty;
            penalties(z) = penalty;
            %M(i,j) = penalty;
        end
    end
    
    if max(penalties) > 0
        best_corr_ind = find(penalties == max(penalties));
        M(indices(best_corr_ind(1)),j) = 1.0;
    end
end

target_outliers = sum(M,2) == 0;
source_outliers = sum(M,1) == 0;
source_outliers = source_outliers';

% M_col_sum = sum(M,1);
% M_col_sum = repmat(M_col_sum, num_points, 1);
% M = M ./ M_col_sum;
% 
% M(isnan(M)) = 0;

return;

% iterated column and row normalization until convergence
eps = 1e-2;
k = 1;
max_iter = 100;

delta_col_max = abs(max(sum(M,1)) - 1);
delta_col_min = abs(min(sum(M,1)) - 1);
delta_col = max(delta_col_max, delta_col_min);

delta_row_max = abs(max(sum(M,2)) - 1);
delta_row_min = abs(min(sum(M,2)) - 1);
delta_row = max(delta_row_max, delta_row_min);

a = M(:,41);

% force M stricly positive to ensure convergence
target_outliers = sum(M,2) == 0;
source_outliers = sum(M,1) == 0;
source_outliers = source_outliers';
M(M < 1e-30) = 1e-30;

while (delta_row > eps || delta_col > eps) && k < max_iter
    % norm cols
    M_col_sum = sum(M,1);
    M_col_sum = repmat(M_col_sum, num_points, 1);
    M = M ./ M_col_sum;
    
    %M(isnan(M)) = 0;
    
%     % norm rows
%     M_row_sum = sum(M,2);
%     M_row_sum = repmat(M_row_sum, 1, num_points);
%     M = M ./ M_row_sum;
%     
    %M(isnan(M)) = 0;
    
    col_sum = sum(M,1);
    delta_col_max = abs(max(col_sum(col_sum > 0)) - 1);
    delta_col_min = abs(min(col_sum(col_sum > 0)) - 1);
    delta_col = max(delta_col_max, delta_col_min);

    row_sum = sum(M,2);
    delta_row_max = abs(max(row_sum(row_sum > 0)) - 1);
    delta_row_min = abs(min(row_sum(row_sum > 0)) - 1);
    delta_row = 0;%max(delta_row_max, delta_row_min);
    
    k = k + 1;
end

b = M(:,41);
% M(bad_rows,:) = 0;
% M(:,bad_cols) = 0;

end


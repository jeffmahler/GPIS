function [corrs, outliers ] = ...
    tsdf_tps_corrs(points, source_tsdf, target_tsdf, ...
        source_grads, target_grads, tsdf_thresh, alpha, beta, win)

% alpha = weighting for tsdf
% beta = weighting for normals

if nargin < 6
   tsdf_thresh = 10; 
end
if nargin < 7
   alpha = 1.0; 
end
if nargin < 8
   beta = 1.0; 
end
if nargin < 9
   win = 2; 
end

[height, width] = size(source_tsdf);
num_points = height * width;
corrs = -1 * ones(num_points,1);
eps = 1e-2;

s_grad = zeros(2,1);
t_grad = zeros(2,1);

% not really that fast
for j = 1:num_points
    x_j = points(j,:)';
    x_j_target = x_j;
        
    k = uint16(x_j_target(1));
    l = uint16(x_j_target(2));
    
    k_low = max(k - win, 1);
    k_high = min(k + win, width);
    l_low = max(l - win, 1);
    l_high = min(l + win, height);
    
    [Xn, Yn] = meshgrid(k_low:k_high, l_low:l_high);
    neighbors = [Xn(:), Yn(:)]; 
    
    penalties = flintmax * ones(size(neighbors,1),1);
    indices = zeros(size(neighbors,1),1);
    for z = 1:size(neighbors,1)
        u = neighbors(z,:);
        i = u(2) + ((u(1) - 1) * height);
        
        indices(z) = i;
        y_i = points(i,:)';
       
        % get tsdf and normal parameters
        t_val = target_tsdf(y_i(2), y_i(1));
        s_val = source_tsdf(x_j(2), x_j(1));
        
        t_grad(:) = target_grads(y_i(2), y_i(1), :);
        s_grad(:) = source_grads(x_j(2), x_j(1), :);
        
        if abs(s_val) < tsdf_thresh - eps % && abs(t_val) < trunc - eps
            point_diff = norm(y_i - x_j_target);
            tsdf_diff = t_val - s_val;
            grad_diff = t_grad' * s_grad / (norm(t_grad) * norm(s_grad));
            
            point_penalty = point_diff^2;
            tsdf_penalty = tsdf_diff^2;
            grad_penalty = abs(grad_diff);
            
            penalty = point_penalty + alpha*tsdf_penalty + beta*grad_penalty;
            penalties(z) = penalty;
        end
    end
    
    if min(penalties) < flintmax
        best_corr_ind = find(penalties == min(penalties));
        corrs(j) = indices(best_corr_ind(1));
    end
end

outliers = (corrs == -1);
  
end


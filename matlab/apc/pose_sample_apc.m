function sdf_samples = ...
    pose_sample_apc(num_perturbations, sdf, centroid, ...
        sigma_trans, sigma_rot, only_2d, vis)

if nargin < 6
    only_2d = false;
end
    
if nargin < 7
    vis = false;
end

sdf_dims = size(sdf);

% generate pose samples to evaluate probability of force closure subject to pose samples
t_perturb = normrnd(0, sigma_trans, num_perturbations, 3);
rot_perturb = normrnd(0, sigma_rot, num_perturbations, 3);
R_sdf = cell(1, num_perturbations);
for i = 1:num_perturbations
    if only_2d
        rot_ind = mnrnd(1, 1.0 / 3.0 * ones(1,3));
        if rot_ind == 1
            R_sdf{i} = angle2dcm(rot_perturb(i,1), 0, 0);
        elseif rot_ind == 2
            R_sdf{i} = angle2dcm(0, rot_perturb(i,2), 0);
        else
            R_sdf{i} = angle2dcm(0, 0, rot_perturb(i,3));
        end
           
    else
        R_sdf{i} = angle2dcm(rot_perturb(i,1), rot_perturb(i,2), rot_perturb(i,3));
    end
end

sdf_samples = cell(1, num_perturbations);
for i = 1:num_perturbations
    if mod(i, 10) == 0
        fprintf('Warping sdf %d\n', i); 
    end
    tf = struct();
    tf.R = R_sdf{i};
    tf.t = t_perturb(i,:)';
    tf.s_center = 1;
    tf.s_trans = 1;
    
    sdf_sample = warp_grid_3d(tf, sdf, centroid);
    [Gx_sample, Gy_sample, Gz_sample] = gradient(sdf_sample);
    shape_params = struct();
    shape_params.sdf = sdf_sample;
    shape_params.Gx = Gx_sample;
    shape_params.Gy = Gy_sample;
    shape_params.Gz = Gz_sample;
    
    [~, surf_points_samp, ~] = compute_tsdf_surface(sdf_sample);
    shape_params.centroid = mean(surf_points_samp);
    
    sdf_samples{i} = shape_params;
    
    if vis
        sdf_x_samp = surf_points_samp(:,1);
        sdf_y_samp  = surf_points_samp(:,2);
        sdf_z_samp  = surf_points_samp(:,3);

        figure(2);
        scatter3(sdf_x_samp , sdf_y_samp , sdf_z_samp);
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        xlim([1, sdf_dims(1)]);
        ylim([1, sdf_dims(2)]);
        zlim([1, sdf_dims(3)]);
    end
end

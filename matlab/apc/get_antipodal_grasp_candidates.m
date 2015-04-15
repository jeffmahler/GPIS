function grasps = get_antipodal_grasp_candidates(sdf, config)

% parse params
grasp_width = config.grasp_width;
friction_coef = config.friction_coef;
n_cone_faces = config.n_cone_faces;
num_samples = config.num_samples;
dir_prior = config.dir_prior;
alpha_thresh = config.alpha_thresh;
rho_thresh = config.rho_thresh;
sdf_res = config.grid_res;
constrain_2d = config.constrain_2d;
vis = config.vis;
  
% tsdf surface points, gradients
[~, surf_points, ~] = compute_tsdf_surface(sdf);
[Gx, Gy, Gz] = gradient(sdf);
sdf_centroid = mean(surf_points);
num_surf = size(surf_points,1);

grasps = {};
index = 1;

for i = 1:num_surf
    fprintf('Checking point %d\n', i);
    
    % get first point on surface
    x1 = surf_points(i,:);
    
    % compute the force cone faces
    [cone_support_x1, n1, failed] = compute_friction_cone(x1, Gx, Gy, Gz, ...
        friction_coef, n_cone_faces);
    if failed
        continue;
    end
    
    % for visualization only
    if vis
        num_faces = size(cone_support_x1, 2);
        contact = [0, 0, 0];
        figure(2);
        for j = 1:num_faces
            line_f = [contact(1), contact(2), contact(3);
                      cone_support_x1(:,j)' / norm(cone_support_x1(:,j))];
            plot3(line_f(:,1), line_f(:,2), line_f(:,3), 'r', 'LineWidth', 5);
            hold on;
        end
    end
    
    % sample dirichlet
    v_samples = sample_friction_cone_3d(cone_support_x1, num_samples, dir_prior);
    
    % iterate through directions
    for j = 1:num_samples
        
        num_dirs = 1;
        if constrain_2d
            num_dirs = 3; % need to consider 3 possible slice dirs
        end
        
        % loop through dirs (so that 2d also works)
        for k = 1:num_dirs
            
            % get new grasp direction
            v = v_samples(:,j)';
            if constrain_2d
                v(k) = 0;
                v = v  / norm(v); % renormalize
            end

            % for visualization only
            if vis
                line_f = [contact(1), contact(2), contact(3);
                    v / norm(v)];
                plot3(line_f(:,1), line_f(:,2), line_f(:,3), 'b', 'LineWidth', 5);
            end

            % start searching for contacts
            [contacts, contact_found] = ...
                antipodal_grasp_contacts(x1, v, sdf, grasp_width);
            if ~contact_found
                continue;
            end
            x2 = contacts(2,:);

            % get the friction cone support at contact 2
            v_true = x2 - x1; % may change due to numeric precision
            v_true = v_true / norm(v_true);
            [cone_support_x2, n2, failed] = compute_friction_cone(x2, Gx, Gy, Gz, ...
                friction_coef, n_cone_faces);
            if failed
                continue;
            end

            % check friction cone
            [in_cone1, alpha1] = within_cone(cone_support_x1, n1, -v_true');
            [in_cone2, alpha2] = within_cone(cone_support_x2, n2, v_true');
            if in_cone1 && in_cone2
                % compute moment arms
                rho1 = norm(x1 - sdf_centroid);
                rho2 = norm(x2 - sdf_centroid);

                fprintf('Grasp candidate with alpha = %f %f rho = %f %f\n', ...
                    alpha1, alpha2, rho1, rho2);

                if alpha1 < alpha_thresh && alpha2 < alpha_thresh && ...
                        rho1 < rho_thresh && rho2 < rho_thresh
                    g_center = (x1 + x2) / 2;

                    grasp = struct();
                    grasp.g1 = x1;
                    grasp.g2 = x2;
                    grasp.g1_open = g_center - (grasp_width / 2) * v_true;
                    grasp.g2_open = g_center + (grasp_width / 2) * v_true;
                    grasp.alpha1 = alpha1;
                    grasp.alpha2 = alpha2;
                    grasp.rho1 = rho1;
                    grasp.rho2 = rho2;
                    grasp.constrained_2d = constrain_2d;
                    grasp.slice = k;

                    if index == 100
                        stop = 1;
                    end
                    [R_g_obj_list, t_g_obj_list] = ...
                        grasp_points_to_poses(grasp, sdf_centroid, sdf_res, config);
                    grasp.R_g_obj_list = R_g_obj_list;
                    grasp.t_g_obj_list = t_g_obj_list;

                    grasps{index} = grasp;
                    index = index + 1;
                end
            end
        end
    end
end


function Q = grasp_quality_apc(grasp, sdf_params, config, step_size)

% set up line of actions
g1_open = grasp.g1_open;
g2_open = grasp.g2_open;

g1_gp = [g1_open; g2_open];
g1_loa = compute_loa(g1_gp, step_size);

g2_gp = [g2_open; g1_open];
g2_loa = compute_loa(g2_gp, step_size);

loas = {g1_loa, g2_loa};

% find contact points
[contacts, success] = find_contacts(loas, sdf_params.sdf);

Q = 0; % if failure then set to 0
if success
    % compute forces and quality
    [forces, normal, failed] = compute_forces(contacts, sdf_params.Gx, ...
        sdf_params.Gy, sdf_params.Gz, config.friction_coef, config.n_cone_faces);
    if ~failed
        Q = ferrari_canny_3d(sdf_params.centroid', contacts', forces, normal);
    end
end

end


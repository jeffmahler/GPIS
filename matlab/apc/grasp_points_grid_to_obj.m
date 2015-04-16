function [g1_obj, g2_obj] = ...
    grasp_points_grid_to_obj(g1, g2, sdf_centroid, sdf_res)
%GRASP_POINTS_TO_POSES 

% centroid in volume frame of ref
centroid_m_vol = grid_to_m(sdf_centroid', sdf_res);
t_vol_obj = -centroid_m_vol;

% convert grasp to obj frame in meters
g1_vol = grid_to_m(g1', sdf_res);
g2_vol = grid_to_m(g2', sdf_res);

% center, (don't rotate here)
g1_obj = (g1_vol + t_vol_obj);
g2_obj = (g2_vol + t_vol_obj);

end

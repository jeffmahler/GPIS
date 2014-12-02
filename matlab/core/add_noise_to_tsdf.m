function [measured_tsdf, noise_grid] = add_noise_to_tsdf(tsdf, points, noise_params)

[height, width] = size(tsdf);
grid_dim = height;
point_inds_linear = sub2ind(size(tsdf), points(:,2), points(:,1));

num_points = size(points, 1);
measured_tsdf = tsdf(point_inds_linear);
noise_grid = zeros(num_points, 1);

for k = 1:num_points
    i = points(k,2);
    j = points(k,1);
    i_low = max(1,i-noise_params.edgeWin);
    i_high = min(grid_dim,i+noise_params.edgeWin);
    j_low = max(1,j-noise_params.edgeWin);
    j_high = min(grid_dim,j+noise_params.edgeWin);
    tsdf_win = tsdf(i_low:i_high, j_low:j_high);
  
    % add in transparency, occlusions
    if ((i > noise_params.transp_y_thresh1_low && i <= noise_params.transp_y_thresh1_high && ...
          j > noise_params.transp_x_thresh1_low && j <= noise_params.transp_x_thresh1_high) || ...
          (i > noise_params.transp_y_thresh2_low && i <= noise_params.transp_y_thresh2_high && ...
          j > noise_params.transp_x_thresh2_low && j <= noise_params.transp_x_thresh2_high) )
        % occluded regions
        if tsdf(i,j) < 0.6 % only add noise to ones that were actually in the shape
            measured_tsdf(k) = 0.5; % set outside shape
            noise_grid(k) = noise_params.transpScale; 
        end

    elseif min(min(tsdf_win)) < 0.6 && ((i > noise_params.y_thresh1_low && i <= noise_params.y_thresh1_high && ...
            j > noise_params.x_thresh1_low && j <= noise_params.x_thresh1_high) || ...
            (i > noise_params.y_thresh2_low && i <= noise_params.y_thresh2_high && ... 
            j > noise_params.x_thresh2_low && j <= noise_params.x_thresh2_high) || ...
            (i > noise_params.y_thresh3_low && i <= noise_params.y_thresh3_high && ... 
            j > noise_params.x_thresh3_low && j <= noise_params.x_thresh3_high))

        noise_grid(k) = noise_params.occlusionScale;
    elseif ((i > noise_params.occ_y_thresh1_low && i <= noise_params.occ_y_thresh1_high && ...
            j > noise_params.occ_x_thresh1_low && j <= noise_params.occ_x_thresh1_high) || ... 
            (i > noise_params.occ_y_thresh2_low && i <= noise_params.occ_y_thresh2_high && ...
            j > noise_params.occ_x_thresh2_low && j <= noise_params.occ_x_thresh2_high) )
        % occluded regions
        noise_grid(k) = noise_params.occlusionScale;

    elseif tsdf(i,j) < -0.5 % only use a few interior points (since realistically we wouldn't measure them)
        if rand() > (1-noise_params.interiorRate)
           noise_grid(k) = noise_params.noiseScale;
        else
           noise_grid(k) = noise_params.occlusionScale; 
        end
    else
        noise_val = 1; % scaling for noise

        % add specularity to surface
        if noise_params.specularNoise && min(min(abs(tsdf_win))) < 0.6
            noise_val = rand();

            if rand() > (1-noise_params.sparsityRate)
                noise_val = noise_params.occlusionScale / noise_params.noiseScale; % missing data not super noisy data
            end
        end
        noise_grid(k) = noise_val * noise_params.noiseScale;
    end
end

end


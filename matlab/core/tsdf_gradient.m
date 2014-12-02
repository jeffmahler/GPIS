function tsdf_grad = tsdf_gradient(tsdf)

[height, width] = size(tsdf);
[Gx, Gy] = imgradientxy(tsdf, 'CentralDifference');
tsdf_grad = zeros(height, width, 2);
tsdf_grad(:,:,1) = Gx;
tsdf_grad(:,:,2) = Gy;

end


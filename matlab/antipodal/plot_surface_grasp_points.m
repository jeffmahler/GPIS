function stop = plot_surface_grasp_points(x, info)
%% PLOT_PATH_2D plots solution path and trust region
% Call with no arguments to clear out the stored solution
persistent h xs1 xs2
if nargin == 0,
%     if exist('h','var') && ishandle(h), delete(h); end
    h = [];
    xs1 = [];
    xs2 = [];
    return;
end
if ~isempty(h)
    delete(h); 
end
fprintf('x: %s\n',mat2str(x));

d = size(x,1) / 2;
scale = info.cfg.scale;
length = info.cfg.arrow_length;
com = info.cfg.com;
plateWidth = info.cfg.plate_width;

[m,n] = size(info.cfg.surfaceImage);
if m >  1
    xs1 = [x(1:d,1)];
    xs2 = [x(d+1:2*d,:)];
    diff = xs2 - xs1;
    
    grad1 = diff;
    grad2 = -diff;
    
    figure(11);
    plot_grasp_arrows( info.cfg.surfaceImage, xs1, xs2, grad1, grad2, ...
        scale, length, com, plateWidth);
%     imshow(info.cfg.surfaceImage);
%     hold on;
%     plot(scale*xs1(1,:),scale*xs1(2,:),'rx-', 'LineWidth', 3, 'MarkerSize', 10);
%     plot(scale*xs2(1,:),scale*xs2(2,:),'rx-', 'LineWidth', 3, 'MarkerSize', 10);
%     hold off;
%     pause(.01);
end
stop = 0;
end

function plot_surface_grasp_points(x, info)
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
[m,n] = size(info.cfg.surfaceImage);
xs1 = [xs1, x(1:d,1)];
xs2 = [xs2, x(d+1:2*d,:)];
imshow(info.cfg.surfaceImage);
hold on;
plot(2*xs1(1,:),2*xs1(2,:),'rx-');
plot(2*xs2(1,:),2*xs2(2,:),'gx-');
hold off;
pause(.01);
end

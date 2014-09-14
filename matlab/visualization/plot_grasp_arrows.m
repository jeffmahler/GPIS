function [] = plot_grasp_arrows( shapeImage, x1, x2, grad1, grad2, scale, length, com)

use_com = true;
if nargin < 8
    use_com = false;
end

% normalize directions
grad1 = grad1 / norm(grad1);
grad2 = grad2 / norm(grad2);
tan1 = [-grad1(2); grad1(1)];
tan2 = [-grad2(2); grad2(1)];

%figure(45);
imshow(shapeImage);
hold on;
% plot arrows
start1 = x1 - length*grad1;
start2 = x2 - length*grad2;

arrow(scale*start1, scale*x1, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 5, 'Width', 2, 'TipAngle', 30);
arrow(scale*start2, scale*x2, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 5, 'Width', 2, 'TipAngle', 30);

% plot small parallel jaws
start1 = x1 + length*tan1/4;
start2 = x2 + length*tan2/4;
end1 = x1 - length*tan1/4;
end2 = x2 - length*tan2/4;

plot(scale*[start1(1); end1(1)], scale*[start1(2); end1(2)], 'r', 'LineWidth', 2);
plot(scale*[start2(1); end2(1)], scale*[start2(2); end2(2)], 'r', 'LineWidth', 2);

if use_com
    % plot the center of mass
    scatter(scale*com(1), scale*com(2), 100.0, '+', 'LineWidth', 2);
end

hold off;

end


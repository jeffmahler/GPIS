function [] = plot_grasp_arrows( shapeImage, x1, x2, grad1, grad2, scale, length)

% normalize directions
grad1 = grad1 / norm(grad1);
grad2 = grad2 / norm(grad2);

imshow(shapeImage);
hold on;
start1 = x1 - length*grad1;
start2 = x2 - length*grad2;

arrow(scale*start1, scale*x1, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 3, 'Width', 2, 'TipAngle', 45);
arrow(scale*start2, scale*x2, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 3, 'Width', 2, 'TipAngle', 45);

hold off;

end


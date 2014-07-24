function [] = plot_grasp_lines( shapeImage, x1, x2, grad1, grad2, scale, length)

% normalize directions
grad1 = grad1 / norm(grad1);
grad2 = grad2 / norm(grad2);

imshow(shapeImage);
hold on;

plot(scale*x1(1,:), scale*x1(2,:), 'rx-', 'MarkerSize', 20, 'LineWidth', 1.5);
plot(scale*x2(1,:), scale*x2(2,:), 'rx-', 'MarkerSize', 20, 'LineWidth', 1.5);
    
hold off;

end


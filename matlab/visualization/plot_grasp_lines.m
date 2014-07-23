function [] = plot_grasp_lines( shapeImage, x1, x2, grad1, grad2, scale, length)

% normalize directions
grad1 = grad1 / norm(grad1);
grad2 = grad2 / norm(grad2);

imshow(shapeImage);
hold on;
start1 = x1 - length*grad1;
start2 = x2 - length*grad2;

plot(scale*[start1(1); x1(1)], scale*[start1(2); x1(2)], 'r', 'LineWidth', 2);
plot(scale*[start2(1); x2(1)], scale*[start2(2); x2(2)], 'r', 'LineWidth', 2);

hold off;

end


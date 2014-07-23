function [] = plot_grasp_lines( shapeImage, x1, x2, grad1, grad2, scale, length)

% normalize directions
grad1 = grad1 / norm(grad1);
grad2 = grad2 / norm(grad2);
tan1 = [-grad1(2); grad1(1)];
tan2 = [-grad2(2); grad2(1)];

imshow(shapeImage);
hold on;
start1 = x1 + length*tan1/2;
start2 = x2 + length*tan2/2;
end1 = x1 - length*tan1/2;
end2 = x2 - length*tan2/2;


plot(scale*[start1(1); end1(1)], scale*[start1(2); end1(2)], 'r', 'LineWidth', 4);
plot(scale*[start2(1); end2(1)], scale*[start2(2); end2(2)], 'r', 'LineWidth', 4);

hold off;

end


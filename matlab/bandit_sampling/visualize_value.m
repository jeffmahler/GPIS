function [ ] = visualize_value( Value,grasp_samples,surface_image)

N = 16; 


A =  load('marker_bandit_values_pfc.mat'); 

true_value = A.Value; 
true_value
[sortedX,sortingIndices] = sort(Value(:,3),'descend');

figure; 
plot(true_value(sortingIndices,3),Value(sortingIndices,2)); 
title('Samples per Grasp Quality'); 
xlabel('Probability of Force Closure'); 
ylabel('Samples'); 
figure;

 for i=1:N
     cp = grasp_samples{sortingIndices(i)}.cp;
     
     plot_grasp_arrows2( surface_image, cp(1,:)', cp(3,:)', -(cp(1,:)-cp(2,:))', -(cp(3,:)-cp(4,:))', 1,10,i,N,true_value(sortingIndices(i),3))
 end

end

function [] = plot_grasp_arrows2( shapeImage, x1, x2, grad1, grad2, scale, length,i,N,val)

% normalize directions
grad1 = grad1 / norm(grad1);
grad2 = grad2 / norm(grad2);
tan1 = [-grad1(2); grad1(1)];
tan2 = [-grad2(2); grad2(1)];


subplot(sqrt(N),sqrt(N),i), subimage(shapeImage);
hold on;
% plot arrows
start1 = x1 - length*grad1;
start2 = x2 - length*grad2;

arrow(scale*start1, scale*x1, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 5, 'Width', 1, 'TipAngle', 30);
arrow(scale*start2, scale*x2, 'FaceColor', 'r', 'EdgeColor', 'r', 'Length', 5, 'Width', 1, 'TipAngle', 30);

% plot small parallel jaws
start1 = x1 + length*tan1/4;
start2 = x2 + length*tan2/4;
end1 = x1 - length*tan1/4;
end2 = x2 - length*tan2/4;

plot(scale*[start1(1); end1(1)], scale*[start1(2); end1(2)], 'r', 'LineWidth', 2);
plot(scale*[start2(1); end2(1)], scale*[start2(2); end2(2)], 'r', 'LineWidth', 2);

xlabel(num2str(val)); 

hold off;

end
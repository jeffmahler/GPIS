load('Grasp.mat')

grasp_set = Grasp;
grasp = Grasp(1,:); 


r1_p = grasp_set(:,2:3);
r2_p = grasp_set(:,4:5); 
n1_p = grasp_set(:,6:7); 
n2_p = grasp_set(:,8:9); 

Dif = zeros(size(r1_p(:,1)));

r1 = grasp(:,2:3); 
r2 = grasp(:,4:5); 
n1 = grasp(:,6:7); 
n2 = grasp(:,8:9); 


for i = 1:size(r1_p(:,1))
    sum = norm(r1_p(i,:)-r1)+norm(r2_p(i,:)-r2)+norm(n1_p(i,:)-n1)+norm(n2_p(i,:)-n2);
    Dif(i,1) = sum;
end

Q = grasp_set(1:3000,1)
Q = Q - zeros(3000,1)+grasp(:,1)

scatter(Dif(1:3000,:),Q)
title('Change in Lipschitz'); 
xlabel('d(g_p,g)'); 
ylabel('Q(g_p)-Q(g)'); 
%plot(Q)
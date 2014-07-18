function [ Q ] = mc_sample( gpModel,allPoints,cone_angle,cp,num_contacts,cm)
%Takes a line not a single contact, add points in the order of [first1;
%last1; first2; last2]; 

%look at Run_Comp.m for a more detail of how to input everything 

%Parameters for Sampling
num_sample = 100; 
thresh = 0.01; 


mn_cp = zeros(2,num_contacts); 
sum = mn_cp; 
sum_sq = sum; 

mn_Q = 0; 
sum_Q = mn_Q; 
sum_Q_sq = sum_Q; 

for k=1:num_sample
    
    [Tsdf, Norm] = sample_shape(gpModel,allPoints);
    
    [contactPts,norms] = find_contact_points(cp,num_contacts,allPoints,Tsdf,Norm,cm,thresh);
    
    fprintf('Mean Contact Points')
    sum = sum + contactPts; 
    mn_cp = sum/k
    
    fprintf('Variance of Contact Points')
    sum_sq = sum_sq + contactPts.*contactPts; 
    v_cp = (sum_sq-(sum.*sum)/k)/k
    
    index = 1; 
    
    %TODO:DOUBLE CHECK ALL FORCES POINT INTO OBJECT
    
    for i=1:num_contacts
        %Find the normalized direction of the normal force 
        f = norms(:,i);
        f = f/norm(f,1)*(1/num_contacts);
        %Compute the extrema of the friction cone 
        y_d = tan(cone_angle);
        x_d = 1/sin(cone_angle);

        f_r = [f(1,1) + x_d; f(2,1)+y_d; 0]; 
        f_l = [f(1,1) - x_d; f(2,1)+y_d; 0]; 

       

        forces(:,index) = f_r; 
        index = index+1;
        forces(:,index) = f_l; 
        index = index+1;
    end

    CP = [contactPts; zeros(1,num_contacts)]; 
    CM = [cm'; 0];
    Q = ferrari_canny(CM,CP,forces);
    data(k) = Q; 
    fprintf('Mean of Grasp Quality')
    sum_Q = sum_Q + Q; 
    mn_Q = sum_Q/k
    
    fprintf('Variance of Grasp Quality')
    sum_Q_sq = sum_Q_sq + Q.*Q; 
    v_Q = (sum_Q_sq-(sum_Q.*sum_Q)/k)/k
    
end
[hst,centers] = hist(data); 
plot(centers,hst)
title('Histogram of Grasp Quality'); 
xlabel('Grasp Quality'); 
ylabel('Count'); 
    
end

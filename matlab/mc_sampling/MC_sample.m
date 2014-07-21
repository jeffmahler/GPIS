function [ mn_Q, v_Q, success] = mc_sample(gpModel, allPoints, cone_angle, cp, ...
                                    num_contacts, cm, num_sample, surf_thresh, ...
                                    bad_contact_thresh)
%Takes a line not a single contact, add points in the order of [first1;
%last1; first2; last2]; 

%look at Run_Comp.m for a more detail of how to input everything 

%Parameters for Sampling
if nargin < 7
    num_sample = 100;
end
if nargin < 8
    surf_thresh = 0.05;
end
if nargin < 9
    bad_contact_thresh = 10;
end

success = true;
mn_cp = zeros(2,num_contacts); 
sum = mn_cp; 
sum_sq = sum; 

mn_Q = 0;
v_Q = 0;
sum_Q = mn_Q; 
sum_Q_sq = sum_Q; 

k = 1;
num_bad = 0;
gridDim = uint16(sqrt(size(allPoints,1)));

while k <= num_sample
    
    [Tsdf, Norm] = sample_shape(gpModel, allPoints);
    sampleCom = mean(allPoints(Tsdf < 0, :), 1);

    tsdfGrid = reshape(Tsdf, 25, 25);
    scale = 5;
    figure(10);
    tsdfGridBig = imresize(tsdfGrid, scale);
    imshow(tsdfGridBig);
    hold on;
    [contactPts, norms, badContact] = ...
        find_contact_points(cp,num_contacts,allPoints,Tsdf,Norm,sampleCom,surf_thresh);

    if badContact
        num_bad = num_bad+1;

        % check if we've had enough failed contacts to give up
        if num_bad > bad_contact_thresh
            disp('Too many failed contacts. Aborting...');
            success = false;
            return;
        end
        continue;
    end

    %%fprintf('Mean Contact Points\n')
    sum = sum + contactPts; 
    mn_cp = sum/k;
    
    %fprintf('Variance of Contact Points\n')
    sum_sq = sum_sq + contactPts.*contactPts; 
    v_cp = (sum_sq-(sum.*sum)/k)/k;
    
    index = 1; 
    for i=1:num_contacts
        % find the unit direction of the normal force 
        f = norms(:,i);
        f = f/norm(f,1)*(1/num_contacts);
       
        % get extrema of friction cone, negative to point into object
        opp_len = tan(cone_angle) * norm(f);
        opp_dir = [-1 / f(1,1); 1 / f(2,1)];
        opp = opp_len * opp_dir / norm(opp_dir);
        f_r = -(f + opp);
        f_l = -(f - opp);

        % normalize (can only provide as much force as the normal force)
        f_r =  f_r * norm(f) / norm(f_r);
        f_l =  f_l * norm(f) / norm(f_l);

        forces(:,index) = [f_r; 0]; 
        index = index+1;
        forces(:,index) = [f_l; 0]; 
        index = index+1;
    end

    fprintf('Assessing grasp sample %d\n', k);
    CP = [contactPts; zeros(1,num_contacts)]; 
    CM = [sampleCom'; 0];
    Q = ferrari_canny(CM,CP,forces);
    data(k) = Q; 
    sum_Q = sum_Q + Q; 
    mn_Q = sum_Q/k;
    fprintf('Mean of Grasp Quality: %f\n', mn_Q);
    
    sum_Q_sq = sum_Q_sq + Q.*Q; 
    v_Q = (sum_Q_sq-(sum_Q.*sum_Q)/k)/k;
    fprintf('Variance of Grasp Quality: %f\n', v_Q);


    k = k+1; % increment loop counter 
end
figure;
nbins = 100;
[hst,centers] = hist(data);
hist(data, nbins);
%figure;
%plot(centers,hst)
title('Histogram of Grasp Quality'); 
xlabel('Grasp Quality'); 
ylabel('Count'); 
    
end

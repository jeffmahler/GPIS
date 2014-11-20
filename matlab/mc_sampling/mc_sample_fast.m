

function [ mn_Q, v_Q, success, varargout] = mc_sample_fast(allPoints, cone_angle, cp, ...
                                    num_contacts, samples, gridDim,com,resolution, surf_thresh, ...
                                    bad_contact_thresh, vis, showHist, qScale)
%Takes a line not a single contact, add points in the order of [first1;
%last1; first2; last2]; 

%look at Run_Comp.m for a more detail of how to input everything 

%Parameters for Sampling
if nargin < 9
    surf_thresh = 0.05;
end
if nargin < 10
    bad_contact_thresh = 100000;
end
if nargin < 11
    vis = false;
end
if nargin < 12
    showHist = false;
end
if nargin < 13
    qScale = 1;
end


success = true;
mn_cp = zeros(2,num_contacts); 

sum_contact = mn_cp; 
sum_sq = sum_contact; 

mn_Q = 0;
v_Q = 0;
sum_Q = mn_Q; 
sum_Q_sq = sum_Q; 

k = 1;
num_bad = 0;
num_sample = size(samples,2);
q_vals = [];
contacts_emp = []; 
norms_emp = []; 
COMS = []; 
avg_q = [];
var_q = [];

while k <= num_sample
    
    shapeSample = samples{k};
    Tsdf = shapeSample.tsdf;
    Norm = shapeSample.normals;
    sampleCom = com;%mean(allPoints(Tsdf < 0, :), 1);
    
    
    tsdfGrid = reshape(Tsdf, sqrt(size(Tsdf,1)), sqrt(size(Tsdf,1)));
    
    if vis
        scale = 5;
        figure(10);
        tsdfGridBig = imresize(tsdfGrid, scale);
        imshow(tsdfGridBig);
        hold on;
    end
    [contactPts, norms, badContact] = ...
        find_contact_points(cp, num_contacts, allPoints, Tsdf, Norm, ...
            sampleCom, surf_thresh, vis);
        
    contactPts = contactPts *resolution; 
    
    if badContact
        num_bad = num_bad+1;

        % check if we've had enough failed contacts to give up
        if num_bad > bad_contact_thresh
            disp('Too many failed contacts. Aborting...');
            success = false;
            p_fc = 0;
            varargout{1} = p_fc;
            return;
        end
        k = k+1;
        continue;
    end

    %%fprintf('Mean Contact Points\n')
    sum_contact = sum_contact + contactPts; 
    mn_cp = sum_contact/k;
    
    %fprintf('Variance of Contact Points\n')
    sum_sq = sum_sq + contactPts.*contactPts; 
    v_cp = (sum_sq-(sum_contact.*sum_contact)/k)/k;
    
    index = 1; 
    for i=1:num_contacts
        % find the unit direction of the normal force 
        f = norms(:,i);
        f = f/norm(f);
       
        % get extrema of friction cone, negative to point into object
        opp_len = tan(cone_angle) * norm(f);
        opp_dir = [-f(2,1); f(1,1)];
        opp = opp_len * opp_dir / norm(opp_dir);
        f_r = -(f + opp);
        f_l = -(f - opp);

        % normalize (can only provide as much force as the normal force)
%         f_r =  f_r * norm(f) / norm(f_r);
%         f_l =  f_l * norm(f) / norm(f_l);

        forces(:,index) = [f_r; 0]; 
        index = index+1;
        forces(:,index) = [f_l; 0]; 
        index = index+1;
    end
    
    
    CP = [contactPts; zeros(1,num_contacts)]; 
    CM = [sampleCom'; 0];
    [Q, qValid] = ferrari_canny(CM,CP,forces);

    if ~qValid
        num_bad = num_bad+1;
        
        % check if we've had enough failed contacts to give up
        if num_bad > bad_contact_thresh
            disp('Too many failed contacts. Aborting...');
            success = false;
            p_fc = 0;
            varargout{1} = p_fc;
            return;
        end
        k = k+1;
        continue;
        
    end

    q_vals = [q_vals; Q];
    mn_Q = mean(q_vals);
    avg_q = [avg_q; mn_Q];
     
    v_Q = std(q_vals);
    var_q = [var_q; 1.96*v_Q/(sqrt(k))];
    contacts_emp = [contacts_emp; [contactPts(:,1)' contactPts(:,2)']]; 
    norms_emp = [norms_emp; [(norms(:,1)/norm(norms(:,1)))' (norms(:,2)/norm(norms(:,2)))']]; 
    
    COMS = [COMS; sampleCom];
    
    if vis
        fprintf('Assessing grasp sample %d\n', k);
        fprintf('Mean of Grasp Quality: %f\n', mn_Q);
        fprintf('Variance of Grasp Quality: %f\n', v_Q);
    end
    
    k = k+1; % increment loop counter 
end


if showHist
    figure;
    nbins = 100;
    [hst,centers] = hist(qScale*q_vals);
    hist(qScale*q_vals, qScale*(-0.1:0.0025:0.1));
    %figure;
    %plot(centers,hst)
    title('Histogram of Grasp Quality'); 
    xlim(qScale*[-0.1, 0.1]);
    xlabel('Grasp Quality'); 
    ylabel('Count');
end


prob_fc = sum(q_vals > 0) / num_sample;
varargout{1} = prob_fc;
varargout{2} = q_vals;
varargout{3} = contacts_emp; 
varargout{4} = norms_emp; 
varargout{5} = COMS;
varargout{6} = avg_q;
varargout{7} = var_q;

end


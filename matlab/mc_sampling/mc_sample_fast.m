function [ mn_Q, v_Q, success, varargout] = mc_sample_fast(allPoints, coneAngle, cpSamples, ...
                                    numContacts, shapeSamples, gridDim, surfThresh, ...
                                    badContactThresh, plateWidth, ...
                                    vis, showHist, qScale)
%Takes a line not a single contact, add points in the order of [first1;
%last1; first2; last2]; 

%look at Run_Comp.m for a more detail of how to input everything 

%Parameters for Sampling
if nargin < 7
    surfThresh = 0.05;
end
if nargin < 8
    badContactThresh = 10;
end
if nargin < 9
    plateWidth = 1;
end
if nargin < 10
    vis = false;
end
if nargin < 11
    showHist = false;
end
if nargin < 12
    qScale = 1;
end

success = true;
mn_cp = zeros(2,numContacts); 
sum_contact = mn_cp; 
sum_sq = sum_contact; 

mn_Q = 0;
v_Q = 0;

k = 1;
num_bad = 0;
num_sample = size(shapeSamples,2);
q_vals = [];

while k <= num_sample
    
    % get next shape and line of action sample
    cp = cpSamples{k};
    shapeSample = shapeSamples{k};
    Tsdf = shapeSample.tsdf;
    Norm = shapeSample.normals;
    sampleCom = mean(allPoints(Tsdf < 0, :), 1);

    % create grid of tsdf (since sampling returns a vector)
    tsdfGrid = reshape(Tsdf, gridDim, gridDim);
    
    scale = 5;
    if vis
        figure(10);
        tsdfGridBig = imresize(tsdfGrid, scale);
        imshow(tsdfGridBig);
        hold on;
    end
    
    % find the contact points given this grasp
    [contactPts, norms, badContact] = ...
        find_contact_points(cp, numContacts, allPoints, Tsdf, Norm, ...
            sampleCom, surfThresh, vis, plateWidth, scale);

    if badContact
        num_bad = num_bad+1;

        % check if we've had enough failed contacts to give up
        if num_bad > badContactThresh
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
    
    norms_degenerate = false;
    for i = 1:numContacts
        for j = i+1:numContacts
            if norm(norms(:,i) - norms(:,j)) < 1e-2
                num_bad = num_bad+1;
                k = k+1;
                norms_degenerate = true;
                break;
            end
        end
    end
    if norms_degenerate
       continue; 
    end
    
    index = 1; 
    forces_failed = false;
    
    for i = 1:numContacts
        % find the unit direction of the normal force 
        f = norms(:,i);
        if sum(abs(f)) == 0
            num_bad = num_bad+1;
            k = k+1;
            forces_failed = true;
            break;
        end
        f = f/norm(f,1)*(1/numContacts);
        
        % get extrema of friction cone, negative to point into object
        opp_len = tan(coneAngle) * norm(f);
        opp_dir = [-f(2,1); f(1,1)];
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
    
    % check for equal forces, which fucks up the convex hull calculation
    for i = 1:2
       if ~forces_failed && sum(abs(forces(i,:) - forces(i,1)), 2) < 1e-2
           num_bad = num_bad+1;
           k = k+1;
           forces_failed = true;
       end
    end
    
    if forces_failed
       continue; 
    end
    
    CP = [contactPts; zeros(1,numContacts)]; 
    CM = [sampleCom'; 0];
    [Q, qValid] = ferrari_canny(CM,CP,forces);

    if ~qValid
        num_bad = num_bad+1;
        
        % check if we've had enough failed contacts to give up
        if num_bad > badContactThresh
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
    v_Q = std(q_vals);
    
    if vis
        fprintf('Assessing grasp sample %d\n', k);
        fprintf('Mean of Grasp Quality: %f\n', mn_Q);
        fprintf('Variance of Grasp Quality: %f\n', v_Q);
    end
    
    k = k+1; % increment loop counter 
end
if showHist
    figure(99);
    [hst,centers] = hist(qScale*q_vals);
    hist(qScale*q_vals, qScale*(-0.2:0.005:0.2));
    %figure;
    %plot(centers,hst)
    title('Histogram of Grasp Quality', 'FontSize', 10); 
    xlim(qScale*[-0.2, 0.2]);
    xlabel('Grasp Quality', 'FontSize', 10); 
    ylabel('Count', 'FontSize', 10);
end

tolerance = 0.001; % how close to zero we're willing to allow the probability to be
prob_fc = sum(q_vals > tolerance) / num_sample;
varargout{1} = prob_fc;
   
end


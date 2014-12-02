function init_grasp = get_initial_antipodal_grasp(predShape, useNormalDirection)

    predGrid = reshape(predShape.tsdf, predShape.gridDim, predShape.gridDim);
    varGrid = reshape(predShape.noise, predShape.gridDim, predShape.gridDim);
    xNormGrid = reshape(predShape.normals(:,1), predShape.gridDim, predShape.gridDim);
    yNormGrid = reshape(predShape.normals(:,2), predShape.gridDim, predShape.gridDim);
    [tsdf_surface, surf_points, ~] = compute_tsdf_surface(predGrid);
    numSurf = size(surf_points, 1);
    
    % choose random point on surface for point 1
    ind1 = uint16(rand * (numSurf-1) + 1);
    x1 = surf_points(ind1,:)' + (0.5 - rand); % small random perturbation
        
    % get direction
    n1 = [xNormGrid(round(x1(2)),round(x1(1))) + 20*(rand - 0.5); ...
          yNormGrid(round(x1(2)),round(x1(1))) + 20*(rand - 0.5)];
    v = -n1 / norm(n1); % go in normal opposite direction
    if ~useNormalDirection
        v = predShape.com' + 4*[rand - 0.5; rand - 0.5] - x1;
        v = v / norm(v);
    end
    
    
    alpha = 1;
    noiseThresh = 5.0;
    firstBigStep = 2;
    cur = x1 + firstBigStep*alpha * v;
    foundInd2 = false;
    hitBorder = false;
    prevTsdfSign = 0;
    latestCrossing = x1;
    k = 1;

    while ~hitBorder
        if round(cur(1)) < 1 || round(cur(2)) < 1 || ...
                round(cur(1)) > predShape.gridDim || round(cur(2)) > predShape.gridDim
            hitBorder = true; % give up if we hit the border
            %disp('Hit border');
            
            % assign last zero crossing before hitting the border
            if foundInd2
                x2 = latestCrossing;
                %disp('Found');
            else
                x2 = cur;
                %disp('Not found');
            end
            continue;
        end
        
%         plot(2*cur(1,:), 2*cur(2,:), 'gx-', 'MarkerSize', 20, 'LineWidth', 1.5);
%         
        tsdfVal = predGrid(round(cur(2)), round(cur(1)));
        varVal = varGrid(round(cur(2)), round(cur(1)));
        if k == 1
           prevTsdfSign = sign(tsdfVal);
        end
        
%         fprintf('Tsdf: %f, Var: %f, Sign %d Prev %d Cur %d %d\n', ...
%             tsdfVal, varVal, sign(tsdfVal), prevTsdfSign, round(cur));
        if varVal < noiseThresh && (abs(tsdfVal) < 0.1 || ...
                                    sign(tsdfVal) ~= prevTsdfSign)
            latestCrossing = cur;
            foundInd2 = true;
            %disp('Hit surface');
            %plot(2*cur(1,:), 2*cur(2,:), 'cx-', 'MarkerSize', 20, 'LineWidth', 1.5);
        
        end
        prevTsdfSign = sign(tsdfVal);
        cur = cur + alpha*v;
        k = k+1;
    end
    
    % now search backwards from initial point
    v = -v;
    cur = x1 + firstBigStep*alpha * v;
    hitBorder = false;
    prevTsdfSign = 0;
    latestCrossing = x1;
    k = 1;

    while ~hitBorder
        if round(cur(1)) < 1 || round(cur(2)) < 1 || ...
                round(cur(1)) > predShape.gridDim || round(cur(2)) > predShape.gridDim
            hitBorder = true; % give up if we hit the border
            %disp('Hit border');
            
            % assign last zero crossing before hitting the border
            x1 = latestCrossing;
            continue;
        end
        
        %plot(2*cur(1,:), 2*cur(2,:), 'rx-', 'MarkerSize', 20, 'LineWidth', 1.5);
        
        tsdfVal = predGrid(round(cur(2)), round(cur(1)));
        varVal = varGrid(round(cur(2)), round(cur(1)));
        if k == 1
           prevTsdfSign = sign(tsdfVal);
        end
        
        %fprintf('Tsdf: %f, Var: %f, Sign %d Prev %d\n', tsdfVal, varVal, sign(tsdfVal), prevTsdfSign);
        if varVal < noiseThresh && (abs(tsdfVal) < 0.1 || ...
                                    sign(tsdfVal) ~= prevTsdfSign)
            latestCrossing = cur;
            %disp('Hit surface');
        end
        prevTsdfSign = sign(tsdfVal);
        cur = cur + alpha*v;
        k = k+1;
    end
%     plot(2*x1(1,:), 2*x1(2,:), 'bx-', 'MarkerSize', 20, 'LineWidth', 1.5);
%     plot(2*x2(1,:), 2*x2(2,:), 'bx-', 'MarkerSize', 20, 'LineWidth', 1.5);
%         

    init_grasp = [x1(2); x1(1); x2(2); x2(1)];
    
%     figure(1);
%     imshow(tsdf_surface);
%     hold on;
%     scatter(x1(2), x1(1), 50, '+r');
%     scatter(x2(2), x2(1), 50, '+g');
end


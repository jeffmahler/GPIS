function [ forces ] = forces_on_friction_cone(nrm,cone_angle)
        f = nrm;
        f = f/norm(f);
       
        
        %get extrema of friction cone, negative to point into object
        opp_len = tan(cone_angle) * norm(f);
        opp_dir = [-f(2,1); f(1,1)];
        opp = opp_len * opp_dir / norm(opp_dir);
        f_r = -(f + opp);
        f_l = -(f - opp);

        % normalize (can only provide as much force as the normal force)
%         f_r =  f_r * norm(f) / norm(f_r);
%         f_l =  f_l * norm(f) / norm(f_l);

        forces(:,1) = [f_r; 0]; 
       
        forces(:,2) = [f_l; 0];
end
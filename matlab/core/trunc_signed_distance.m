function truncSignedDist = trunc_signed_distance(occupancy, thresh)
%SIGNED_DISTANCE compute the signed distance for a signed distance grid

if nargin < 2 || thresh > 10
   thresh = 10;
end

posSignedDist = bwdist(occupancy);
negSignedDist = -bwdist(~occupancy);
truncSignedDist = min(posSignedDist, thresh) + max(negSignedDist, -thresh);

end


function dcm = angle2dcm( r1, r2, r3, varargin )

%  ANGLE2DCM Create direction cosine matrix from rotation angles.

%   N = ANGLE2DCM( R1, R2, R3 ) calculates the direction cosine matrix, N,

%   for a given set of rotation angles, R1, R2, R3.   R1 is an M array of

%   first rotation angles.  R2 is an M array of second rotation angles.  R3

%   is an M array of third rotation angles.  N returns an 3-by-3-by-M

%   matrix containing M direction cosine matrices.  Rotation angles are

%   input in radians.  

%

%   N = ANGLE2DCM( R1, R2, R3, S ) calculates the direction cosine matrix,

%   N, for a given set of rotation angles, R1, R2, R3, and a specified

%   rotation sequence, S. 

%

%   The default rotation sequence is 'ZYX' where the order of rotation

%   angles for the default rotation are R1 = Z Axis Rotation, R2 = Y Axis

%   Rotation, and R3 = X Axis Rotation. 

%

%   All rotation sequences, S, are supported: 'ZYX', 'ZYZ', 'ZXY', 'ZXZ',

%   'YXZ', 'YXY', 'YZX', 'YZY', 'XYZ', 'XYX', 'XZY', and 'XZX'.

%

%   Examples:

%

%   Determine the direction cosine matrix from rotation angles:

%      yaw = 0.7854; 

%      pitch = 0.1; 

%      roll = 0;

%      dcm = angle2dcm( yaw, pitch, roll )

%

%   Determine the direction cosine matrix from multiple rotation angles:

%      yaw = [0.7854 0.5]; 

%      pitch = [0.1 0.3]; 

%      roll = [0 0.1];

%      dcm = angle2dcm( pitch, roll, yaw, 'YXZ' )

%

%   See also DCM2ANGLE, DCM2QUAT, QUAT2DCM, QUAT2ANGLE.

 

%   Copyright 2000-2011 The MathWorks, Inc.

 

narginchk(3,4)

 

if any(~isreal(r1) || ~isnumeric(r1))

    error(message('aero:angle2dcm:isNotReal1'));

end

 

if any(~isreal(r2) || ~isnumeric(r2))

    error(message('aero:angle2dcm:isNotReal2'));

end

 

if any(~isreal(r3) || ~isnumeric(r3))

    error(message('aero:angle2dcm:isNotReal3'));

end

 

if (length(r1) ~= length(r2)) || (length(r1) ~= length(r3))

    error(message('aero:angle2dcm:wrongDimension'));

end

 

if nargin == 3

    type = 'zyx';

else

    if ischar( varargin{1} )

        type = varargin{1};

    else

        error(message('aero:angle2dcm:notChar'));

    end

end

 

angles = [r1(:) r2(:) r3(:)];

 

dcm = zeros(3,3,size(angles,1));

cang = cos( angles );

sang = sin( angles );

 

switch lower( type )

    case 'zyx'

        %     [          cy*cz,          cy*sz,            -sy]

        %     [ sy*sx*cz-sz*cx, sy*sx*sz+cz*cx,          cy*sx]

        %     [ sy*cx*cz+sz*sx, sy*cx*sz-cz*sx,          cy*cx]

 

        dcm(1,1,:) = cang(:,2).*cang(:,1);

        dcm(1,2,:) = cang(:,2).*sang(:,1);

        dcm(1,3,:) = -sang(:,2);

        dcm(2,1,:) = sang(:,3).*sang(:,2).*cang(:,1) - cang(:,3).*sang(:,1);

        dcm(2,2,:) = sang(:,3).*sang(:,2).*sang(:,1) + cang(:,3).*cang(:,1);

        dcm(2,3,:) = sang(:,3).*cang(:,2);

        dcm(3,1,:) = cang(:,3).*sang(:,2).*cang(:,1) + sang(:,3).*sang(:,1);

        dcm(3,2,:) = cang(:,3).*sang(:,2).*sang(:,1) - sang(:,3).*cang(:,1);

        dcm(3,3,:) = cang(:,3).*cang(:,2);

 

    case 'zyz'

        %     [  cz2*cy*cz-sz2*sz,  cz2*cy*sz+sz2*cz,           -cz2*sy]

        %     [ -sz2*cy*cz-cz2*sz, -sz2*cy*sz+cz2*cz,            sz2*sy]

        %     [             sy*cz,             sy*sz,                cy]

        

        dcm(1,1,:) = cang(:,1).*cang(:,3).*cang(:,2) - sang(:,1).*sang(:,3);

        dcm(1,2,:) = sang(:,1).*cang(:,3).*cang(:,2) + cang(:,1).*sang(:,3);

        dcm(1,3,:) = -sang(:,2).*cang(:,3);  

        dcm(2,1,:) = -cang(:,1).*cang(:,2).*sang(:,3) - sang(:,1).*cang(:,3);                

        dcm(2,2,:) = -sang(:,1).*cang(:,2).*sang(:,3) + cang(:,1).*cang(:,3);

        dcm(2,3,:) = sang(:,2).*sang(:,3);     

        dcm(3,1,:) = cang(:,1).*sang(:,2);

        dcm(3,2,:) = sang(:,1).*sang(:,2);  

        dcm(3,3,:) = cang(:,2);

                

    case 'zxy'

        %     [ cy*cz-sy*sx*sz, cy*sz+sy*sx*cz,         -sy*cx]

        %     [         -sz*cx,          cz*cx,             sx]

        %     [ sy*cz+cy*sx*sz, sy*sz-cy*sx*cz,          cy*cx]

 

        dcm(1,1,:) = cang(:,3).*cang(:,1) - sang(:,2).*sang(:,3).*sang(:,1);

        dcm(1,2,:) = cang(:,3).*sang(:,1) + sang(:,2).*sang(:,3).*cang(:,1);

        dcm(1,3,:) = -sang(:,3).*cang(:,2);

        dcm(2,1,:) = -cang(:,2).*sang(:,1);

        dcm(2,2,:) = cang(:,2).*cang(:,1);

        dcm(2,3,:) = sang(:,2);

        dcm(3,1,:) = sang(:,3).*cang(:,1) + sang(:,2).*cang(:,3).*sang(:,1);

        dcm(3,2,:) = sang(:,3).*sang(:,1) - sang(:,2).*cang(:,3).*cang(:,1);

        dcm(3,3,:) = cang(:,2).*cang(:,3);

 

    case 'zxz'

        %     [  cz2*cz-sz2*cx*sz,  cz2*sz+sz2*cx*cz,            sz2*sx]

        %     [ -sz2*cz-cz2*cx*sz, -sz2*sz+cz2*cx*cz,            cz2*sx]

        %     [             sz*sx,            -cz*sx,                cx]

 

        dcm(1,1,:) = -sang(:,1).*cang(:,2).*sang(:,3) + cang(:,1).*cang(:,3);

        dcm(1,2,:) = cang(:,1).*cang(:,2).*sang(:,3) + sang(:,1).*cang(:,3);                

        dcm(1,3,:) = sang(:,2).*sang(:,3);     

        dcm(2,1,:) = -sang(:,1).*cang(:,3).*cang(:,2) - cang(:,1).*sang(:,3);

        dcm(2,2,:) = cang(:,1).*cang(:,3).*cang(:,2) - sang(:,1).*sang(:,3);

        dcm(2,3,:) = sang(:,2).*cang(:,3);  

        dcm(3,1,:) = sang(:,1).*sang(:,2);  

        dcm(3,2,:) = -cang(:,1).*sang(:,2);

        dcm(3,3,:) = cang(:,2);

 

    case 'yxz'

        %     [  cy*cz+sy*sx*sz,           sz*cx, -sy*cz+cy*sx*sz]

        %     [ -cy*sz+sy*sx*cz,           cz*cx,  sy*sz+cy*sx*cz]

        %     [           sy*cx,             -sx,           cy*cx]

 

        dcm(1,1,:) = cang(:,1).*cang(:,3) + sang(:,2).*sang(:,1).*sang(:,3);

        dcm(1,2,:) = cang(:,2).*sang(:,3);

        dcm(1,3,:) = -sang(:,1).*cang(:,3) + sang(:,2).*cang(:,1).*sang(:,3);

        dcm(2,1,:) = -cang(:,1).*sang(:,3) + sang(:,2).*sang(:,1).*cang(:,3);

        dcm(2,2,:) = cang(:,2).*cang(:,3);

        dcm(2,3,:) = sang(:,1).*sang(:,3) + sang(:,2).*cang(:,1).*cang(:,3);        

        dcm(3,1,:) = sang(:,1).*cang(:,2);

        dcm(3,2,:) = -sang(:,2);

        dcm(3,3,:) = cang(:,2).*cang(:,1);

 

    case 'yxy'

        %     [  cy2*cy-sy2*cx*sy,            sy2*sx, -cy2*sy-sy2*cx*cy]

        %     [             sy*sx,                cx,             cy*sx]

        %     [  sy2*cy+cy2*cx*sy,           -cy2*sx, -sy2*sy+cy2*cx*cy]

 

        dcm(1,1,:) = -sang(:,1).*cang(:,2).*sang(:,3) + cang(:,1).*cang(:,3);

        dcm(1,2,:) = sang(:,2).*sang(:,3);     

        dcm(1,3,:) = -cang(:,1).*cang(:,2).*sang(:,3) - sang(:,1).*cang(:,3);                

        dcm(2,1,:) = sang(:,1).*sang(:,2);  

        dcm(2,2,:) = cang(:,2);

        dcm(2,3,:) = cang(:,1).*sang(:,2);

        dcm(3,1,:) = sang(:,1).*cang(:,3).*cang(:,2) + cang(:,1).*sang(:,3);

        dcm(3,2,:) = -sang(:,2).*cang(:,3);  

        dcm(3,3,:) = cang(:,1).*cang(:,3).*cang(:,2) - sang(:,1).*sang(:,3);

        

    case 'yzx'

        %     [           cy*cz,              sz,          -sy*cz]

        %     [ -sz*cx*cy+sy*sx,           cz*cx,  sy*cx*sz+cy*sx]

        %     [  cy*sx*sz+sy*cx,          -cz*sx, -sy*sx*sz+cy*cx]

 

        dcm(1,1,:) = cang(:,1).*cang(:,2);

        dcm(1,2,:) = sang(:,2);

        dcm(1,3,:) = -sang(:,1).*cang(:,2);

        dcm(2,1,:) = -cang(:,3).*cang(:,1).*sang(:,2) + sang(:,3).*sang(:,1);

        dcm(2,2,:) = cang(:,2).*cang(:,3);

        dcm(2,3,:) = cang(:,3).*sang(:,1).*sang(:,2) + sang(:,3).*cang(:,1);        

        dcm(3,1,:) = sang(:,3).*cang(:,1).*sang(:,2) + cang(:,3).*sang(:,1);

        dcm(3,2,:) = -sang(:,3).*cang(:,2);

        dcm(3,3,:) = -sang(:,3).*sang(:,1).*sang(:,2) + cang(:,3).*cang(:,1);

        

    case 'yzy'

        %     [  cy2*cz*cy-sy2*sy,            cy2*sz, -cy2*cz*sy-sy2*cy]

        %     [            -cy*sz,                cz,             sy*sz]

        %     [  sy2*cz*cy+cy2*sy,            sy2*sz, -sy2*cz*sy+cy2*cy]

 

        dcm(1,1,:) = cang(:,1).*cang(:,3).*cang(:,2) - sang(:,1).*sang(:,3);

        dcm(1,2,:) = sang(:,2).*cang(:,3);  

        dcm(1,3,:) = -sang(:,1).*cang(:,3).*cang(:,2) - cang(:,1).*sang(:,3);

        dcm(2,1,:) = -cang(:,1).*sang(:,2);

        dcm(2,2,:) = cang(:,2);

        dcm(2,3,:) = sang(:,1).*sang(:,2);  

        dcm(3,1,:) = cang(:,1).*cang(:,2).*sang(:,3) + sang(:,1).*cang(:,3);                

        dcm(3,2,:) = sang(:,2).*sang(:,3);     

        dcm(3,3,:) = -sang(:,1).*cang(:,2).*sang(:,3) + cang(:,1).*cang(:,3);

 

    case 'xyz'

        %     [          cy*cz, sz*cx+sy*sx*cz, sz*sx-sy*cx*cz]

        %     [         -cy*sz, cz*cx-sy*sx*sz, cz*sx+sy*cx*sz]

        %     [             sy,         -cy*sx,          cy*cx]

 

        dcm(1,1,:) = cang(:,2).*cang(:,3);

        dcm(1,2,:) = sang(:,1).*sang(:,2).*cang(:,3) + cang(:,1).*sang(:,3);

        dcm(1,3,:) = -cang(:,1).*sang(:,2).*cang(:,3) + sang(:,1).*sang(:,3);

        dcm(2,1,:) = -cang(:,2).*sang(:,3);

        dcm(2,2,:) = -sang(:,1).*sang(:,2).*sang(:,3) + cang(:,1).*cang(:,3);

        dcm(2,3,:) = cang(:,1).*sang(:,2).*sang(:,3) + sang(:,1).*cang(:,3);        

        dcm(3,1,:) = sang(:,2);

        dcm(3,2,:) = -sang(:,1).*cang(:,2);

        dcm(3,3,:) = cang(:,1).*cang(:,2);

        

    case 'xyx'

        %     [                cy,             sy*sx,            -sy*cx]

        %     [            sx2*sy,  cx2*cx-sx2*cy*sx,  cx2*sx+sx2*cy*cx]

        %     [            cx2*sy, -sx2*cx-cx2*cy*sx, -sx2*sx+cx2*cy*cx]

 

        dcm(1,1,:) = cang(:,2);

        dcm(1,2,:) = sang(:,1).*sang(:,2);     

        dcm(1,3,:) = -cang(:,1).*sang(:,2);

        dcm(2,1,:) = sang(:,2).*sang(:,3);        

        dcm(2,2,:) = -sang(:,1).*cang(:,2).*sang(:,3) + cang(:,1).*cang(:,3);

        dcm(2,3,:) = cang(:,1).*cang(:,2).*sang(:,3) + sang(:,1).*cang(:,3);                

        dcm(3,1,:) = sang(:,2).*cang(:,3);        

        dcm(3,2,:) = -sang(:,1).*cang(:,3).*cang(:,2) - cang(:,1).*sang(:,3);

        dcm(3,3,:) = cang(:,1).*cang(:,3).*cang(:,2) - sang(:,1).*sang(:,3);

        

    case 'xzy'

        %     [          cy*cz, sz*cx*cy+sy*sx, cy*sx*sz-sy*cx]

        %     [            -sz,          cz*cx,          cz*sx]

        %     [          sy*cz, sy*cx*sz-cy*sx, sy*sx*sz+cy*cx]

 

        dcm(1,1,:) = cang(:,3).*cang(:,2);

        dcm(1,2,:) = cang(:,1).*cang(:,3).*sang(:,2) + sang(:,1).*sang(:,3);

        dcm(1,3,:) = sang(:,1).*cang(:,3).*sang(:,2) - cang(:,1).*sang(:,3);

        dcm(2,1,:) = -sang(:,2);

        dcm(2,2,:) = cang(:,1).*cang(:,2);

        dcm(2,3,:) = sang(:,1).*cang(:,2);        

        dcm(3,1,:) = sang(:,3).*cang(:,2);

        dcm(3,2,:) = cang(:,1).*sang(:,2).*sang(:,3) - sang(:,1).*cang(:,3);                

        dcm(3,3,:) = sang(:,1).*sang(:,2).*sang(:,3) + cang(:,1).*cang(:,3);

        

    case 'xzx'

        %     [                cz,             sz*cx,             sz*sx]

        %     [           -cx2*sz,  cx2*cz*cx-sx2*sx,  cx2*cz*sx+sx2*cx]

        %     [            sx2*sz, -sx2*cz*cx-cx2*sx, -sx2*cz*sx+cx2*cx]

 

        dcm(1,1,:) = cang(:,2);

        dcm(1,2,:) = cang(:,1).*sang(:,2);     

        dcm(1,3,:) = sang(:,1).*sang(:,2);

        dcm(2,1,:) = -sang(:,2).*cang(:,3);        

        dcm(2,2,:) = cang(:,1).*cang(:,3).*cang(:,2) - sang(:,1).*sang(:,3);

        dcm(2,3,:) = sang(:,1).*cang(:,3).*cang(:,2) + cang(:,1).*sang(:,3);

        dcm(3,1,:) = sang(:,2).*sang(:,3);

        dcm(3,2,:) = -cang(:,1).*cang(:,2).*sang(:,3) - sang(:,1).*cang(:,3);                

        dcm(3,3,:) = -sang(:,1).*cang(:,2).*sang(:,3) + cang(:,1).*cang(:,3);

 

    otherwise

        error(message('aero:angle2dcm:unknownRotation', type));

end


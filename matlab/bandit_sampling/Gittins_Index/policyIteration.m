function [V,D] = policyIteration(B,R,P)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [V,D] = policyIteration(B,R,P)
%
% This algorithm has been published in:
%
% Ronald A. Howard, "Dynamic Programming and Markov Processes",
% The Massachusetts Institute of Technology and John Wiley & Sons, Inc.,
% 1960, Library of Congress Catalog Card Number: 60-11030, p. 84
%
% and in:
%
% Martin L. Puterman, "Markov Decision Processes - Discrete Stochastic
% Dynamic Programming", John Wiley & Sons, Inc., 1994, ISBN 0-471-61977-9,
% P. 174
%
% The algorithm in Howard (1960) differs from the one in Puterman (1994) in
% the policy improvement part, because Howard did not include the discount
% factor.  This does not affect the policy determination but it affects the
% total value computation: I include the factor to avoid having to execute one
% value determination step more.
%
% Input:
%
% B = Discount Factor in ]0,1[
% R = State Reward per action (MxA)
% P = Transition Probability per action (MxMxA)
%
% Output:
%
% D = Policy (Mx1)
% V = Values (Mx1)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2005 Lorenzo Di Gregorio
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%         Lorenzo Di Gregorio <lorenzo.digregorio@gmail.com>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = size(P,3);
M = size(P,2);

Vd = zeros(M,1); % First guess is to max immediate reward
Dp(1:M,1) = 0;   % Set to 0 to avoid a successful first comparison

while true

    %%%%%%%%%%%%%%%%%%%%
    % Policy Improvement
    % ==================
    %
    % Starting with Vd=0 selects the policy for the maximum immediate
    % reward as a first guess.
    %
    % Here we include the discount factor B for obtaining the correct total
    % values without executing one final value determination step.
    
    for I=1:A
        Q(:,I)=R(:,I)+B*(P(:,:,I)*Vd);
    end
    [V,D] = max(Q,[],2);
    
    % If we converged to a policy we can stop here
    if isequal(D,Dp)
        break;
    end
    
    % otherwise we have to reiterate the policy
    Dp=D;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Value-Determination
    % ===================
    %
    % Based on the selected policy we:
    % 1. determine the reward and transition matrixes and
    % 2. calculate the total values Vd=Rd+B*Pd*Vd

    for I=1:M
      Pd(I,:) = P(I,:,D(I));
      Rd(I,1) = R(I,D(I));
    end
    Vd = (eye(M)-B*Pd)\Rd;
end


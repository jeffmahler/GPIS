function G = gittins_index_by_katehakis(B,R,P)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gittins Index Computation
% =========================
%
% This algorithm has been published in:
%
% Michael N. Katehakis and Arthur F. Veinott (1987)
% "The Multi-Armed Bandit Problem: Decomposition and
% Computation", Mathematics of Operations Research
% Vol. 12, No. 2, pages 262-268.
%
% Inputs:
%
%  B: discount factor (scalar)
%  R: state reward vector (m x 1)
%  P: stochastic transition matrix (m x m)
%
% Output:
%
%  G: Gittins indexes (1 x m)
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

M = size(R,1); % Number of states

Ps(:,:,1) = P;          % Normal Action
Rs(:,1)   = R;          % Normal Reward
G(1:M)    = 0;

for S=1:M        % Computing state S:

  % Solving the Restart-in-State-S Problem
  % by Policy Iteration

  for I=1:M               % Restart-in-State-S is the alternative action
      Ps(I,:,2) = P(S,:);
  end
  Rs(:,2)   = R(S);       % The reward of the restart is the state S one
  
  % Use the standard policy iteration to solve the problem
  
  [V D] = policyIteration(B,Rs,Ps);

  % The solution is the Gittins index (Whittle's characterization)
  
  G(S) = R(S)+B*P(S,:)*V;
  
end


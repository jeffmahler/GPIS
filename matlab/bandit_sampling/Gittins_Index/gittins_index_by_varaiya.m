function G = gittins_index_by_varaiya(B,R,P)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gittins Index Computation
% =========================
%
% This algorithm has been published in:
%
% P. P. Varaiya, J. C. Walrand and C. Buyukkoc (1985)
% "Extensions of the multiarmed bandit problem: The
% discounted case", IEEE Transactions on Automatic Control,
% Vol. 30, pages 426-439.
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

K = size(R,1);   % Number of states

Pm = zeros(K,K); % transitions to higher index states
C = ones(K,1);   % non-computed states

for M=1:K        % there are at the most K indexes to compute
    Am = ((eye(K)-B*Pm)\R);           % mean reward
    Bm = ((eye(K)-B*Pm)\ones(K,1));   % mean stopping time
    Am = Am.*C;                       % remove already computed indexes
    Bm = Bm.*C;                       % remove already computed indexes
    [V, I]=max(Am(1:K,1)./Bm(1:K,1)); % the index is the best ratio
    G(I)=V;                           % copy in the result vector
    C(I)=nan;                         % exclude from subsequent computations
    Pm(:,I)=P(:,I);                   % include in the transition matrix
end


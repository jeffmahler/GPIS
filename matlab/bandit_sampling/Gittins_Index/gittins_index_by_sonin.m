function G = gittins_index_by_sonin(B,R,P)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gittins Index Computation
% =========================
%
% This algorithm has been published in:
%
% Isaac M. Sonin "A generalized Gittins Index for a Markov
% Chain and its Recursive Calculation", Workshop on Optimal
% Stopping and Stochastic Control, 22-26 August 2005
% Petrozavodsk, Russia
%
% Inputs:
%
%  B: discount factors (m x 1)
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

G(1:size(P,1)+1) = 0;
S = 1:size(P,1)+1;

for I=1:size(P,1)
    P(I,:) = P(I,:)*B(I);
end

P(:,size(P,2)+1)       = 1-B;
P(size(P,1)+1,:)       = 0;
P(size(P,1),size(P,2)) = 1;

R(size(R,1)+1)         = 0;

while(true)

    % Calculate Stopping Retirements
    D = R./P(:,size(P,2));
    
    % Identify Stopping and Continuation Sets
    Sset = find(D == max(D));
    Cset = find(D <  max(D));
    
    % The stopping set gets its retirement reward
    G(S(Sset)) = max(D);
    
    if Cset
        % Extract minors of the continuation set
        P1 = P(Cset,Cset);
        T1 = P(Sset,Cset);
        R1 = P(Cset,Sset);
        Q1 = P(Sset,Sset);
        N1 = inv(eye(size(Q1,1))-Q1);

        P = P1 + R1*N1*T1;
        R = R(Cset) + R1*N1*R(Sset);
        S = S(Cset);
   else
       break;
   end 
end

G = G(1:size(G,2)-1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gittins Index Computation Demo
% ==============================
%
% Demonstrate the use of the index computation algorithms.
%
% The same indexes are computed using:
%
% - the largest-index-first algorithm of Varaiya et al.
% - the restart-state formulation of Katehakis el al.
% - the elimination algorithm of Sonin
%
% The results differ for a factor (1-Discount) because of
% the difference in the characterizations given by Gittins
% and Whittle.
%
% All remaining differences are due to rounding errors.
%
% The differences between the Varaiya and the Katehakis
% algorithms, in the Gittins' characterization, are reported
% in the matrix C_vk.
%
% The differences between the Katehakis and the Sonin
% algorithms, in the Whittle characterization, are reported
% in the matrix C_ks.
%
% The differences between the Sonin and the Varaiya
% algorithms, in the Gittins' characterization, are reported
% in the matrix C_sv.
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

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B = 0.9;     % Discount Factor
M = 10;      % Number of states
N = 10;      % Number of experiments

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp("")
disp("Executing N experiments of random models of M states with discount factor B")
disp("(change these values in the script)")
disp("")

% Set a termination vector for the Sonin's algorithm
Bv(1:M,1)=B;

clear C_vk C_ks C_sv; % Data Collectors

for I = 1:N

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Random Data Generation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    clear R P;
    
    R = rand(M,1);

    P(1:M,1:M) = rand(M,M);
    for J=1:M
        P(J,:) = P(J,:)./sum(P(J,:));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Index Computation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    G = gittins_index_by_varaiya(B,R,P);
    H = gittins_index_by_katehakis(B,R,P);
    S = gittins_index_by_sonin(Bv,R,P);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Data Collection
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    C_vk(I,:) = G-H.*(1-B);
    C_ks(I,:) = H-S;
    C_sv(I,:) = G-S.*(1-B);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Report Data    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp("The difference between the outcomes of the experiments are reported as rows of the following matrices")
disp("")
disp("C_vk: differences between the algorithms of Varaiya et al. and Katehakis et al.")
disp("C_ks: differences between the algorithms of Katehakis et al. and Sonin")
disp("C_sv: differences between the algorithms of Sonin and Varaiya et al.")
disp("")
disp("You should see only small rounding differences")
disp("")


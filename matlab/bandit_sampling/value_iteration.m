% value iteration:

function [V, pi] = value_iteration(mdp, precision,start)

%IN: mdp, precision
%OUT: V, pi

% Recall: to obtain an estimate of the value function within accuracy of
% "precision" it suffices that one of the following conditions is met:
%   (i)  max(abs(V_new-V)) <= precision / (2*gamma/(1-gamma))
%   (ii) gamma^i * Rmax / (1-gamma) <= precision  -- with i the value
%   iteration count, and Rmax = max_{s,a,s'} | R(s,a,s') |
    Vdim = 1;
    if mdp.horizon  > 0
        Vdim = mdp.horizon+1; % +1 to add a 0 entry
    end

    V = zeros(mdp.nStates, Vdim);
    next_V = zeros(mdp.nStates, 1);
    pi = zeros(mdp.nStates, Vdim);
    next_pi = zeros(mdp.nStates, 1);
    improvement = realmax;
    k = 1;
    
    % iterate until horizon reached or convergence
    while (k <= mdp.horizon && mdp.horizon > 0) || ...
          (improvement > precision && mdp.horizon == -1) 
      
      improvement = 0;
      curT = 1;
      if mdp.horizon > 0
         curT = k; 
      end
      
      for cState = start:mdp.nStates
          % maximize over all actions
          bestAction = 1;
          bestReward = -realmax;
          
          for action = 1:mdp.nActions
              % sum reward over all states
              reward = 0;
              
              for nState = [start sub2ind(size,[s+1 f]) sub2ind(size,[s f+1])] 
 
                  plus = mdp.T{action}(cState, nState) * ...
                      (mdp.R{action}(cState, nState) + ...
                       mdp.gamma * V(nState, curT));
                  reward = reward + plus;
              end
              
              % update the max, argmax
              if reward > bestReward
                 bestAction = action;
                 bestReward = reward;
              end
          end
          
          % update the optimal solution at the next timestep
          
          if bestReward > 1
             a = 0; 
          end
          improvement = improvement + (bestReward - V(cState, curT))^2;
          next_pi(cState, 1) = bestAction;
          next_V(cState, 1) = bestReward;
      end
      nextT = curT;
      if mdp.horizon > 0
         nextT = curT+1;
      end
      for s = 1:mdp.nStates
          pi(s, nextT) = next_pi(s, 1);
          V(s, nextT) = next_V(s, 1);
      end
      improvement = sqrt(improvement);
      fprintf('Iteration %d improvement: %f\n', k, improvement);
      k = k+1;
end
    
    
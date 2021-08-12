import numpy as np 
from env import TabularQubitEnv
import time

np.set_printoptions(suppress=True, precision=4)




def take_eps_greedy_action(Qs,eps,avail_actions=np.arange(3)):
    """
    Qs: np.ndarray
        the Q(s,:) values for a fixed state, over all available actions from that state.
    eps: double
        small number to define the eps-greedy policy.
    avail_actions: np.ndarray
        an array containing all allowed actions from the state s. 
        For the WindyGridWorld environment, these are always all four actions. 
        
    """
    
    
    if np.linalg.norm(Qs) < 1E-12:
    	a = np.random.choice( avail_actions )

    else:
    	# compute greedy action
	    a = np.argmax(Qs) 
	    
	    # draw a random number in [0,1]
	    delta=np.random.uniform()
	    
	    # take a non=greedy action with probability eps/|A|
	    if delta < eps/avail_actions.shape[0]:
	        a = np.random.choice( avail_actions[np.arange(len(avail_actions)) != a] )
        
    return a



def Q_Learning(N_episodes, alpha, gamma, eps):
    """
    N_episodes: int
        number of training episodes
    alpha: double
        learning rate or step-size parameter. Should be in the interval [0,1].
    gamma: double
        discount factor. Should be in the interval [0,1].
    eps: double
        the eps-greedy policy paramter. Control exploration. Should be in the interval [0,1].
    """

    # initialize Q function
    Q = np.zeros((env.theta_grid.shape[0]-1,env.phi_grid.shape[0]-1,env.r_grid.shape[0]-1,len(env.action_space)),) # Q(theta,phi,r; action) table
    

    # policy evaluation
    for episode in range(N_episodes):
        
        # reset environment and compute initial state
        S=env.reset(random=True).copy()
        Return=0.0

        # print(S)
        # print(env.psi)
        # print(env.state)
        # print(env.RL_state_to_rdm(env.state))
        #exit()

        print('init Q-func', Q[S[0],S[1],S[2],:], )
        
        # loop over the timesteps in the episode
        done=False
        while not done:
            
            # choose action
            A=take_eps_greedy_action(Q[S[0],S[1],S[2],:],eps)


            # take an environment step
            S_p, R, done = env.step(A)
                
            #print(R,done, env.env_step, env.theta_grid[S_p[0]], env.phi_grid[S_p[1]])

            # update value function
            bellmann_error = R + gamma * np.max(Q[S_p[0],S_p[1],S_p[2],:]) - Q[S[0],S[1],S[2],A]
            Q[S[0],S[1],S[2],A] += alpha * bellmann_error

            #print(Q[S[0],S[1],A])
                
            # update states
            S=S_p.copy()
            Return+=R

        print(episode, Return, env.env_step )
        print()
        #exit()


    return Q
      

#########################################


seed=0 # set seed of rng
env = TabularQubitEnv(seed) # tabular env


# learning rate
alpha = 0.8
# discount factor
gamma = 1.0
# epsilon: small positive number used in the definition of the epsilon-greedy policy
eps = 0.1
# number of episodes to collect data from
N_episodes=10000 #500000


ti=time.time()
Q=Q_Learning(N_episodes, alpha, gamma, eps)
tf=time.time()

#print(Q)

print('calculation took {} secs.'.format(tf-ti))




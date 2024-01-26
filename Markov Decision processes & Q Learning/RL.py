import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward
        self.n = np.zeros((self.mdp.nActions, self.mdp.nStates))

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs:
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with
        probabilty epsilon and performing Boltzmann exploration otherwise.
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs:
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        Q = initialQ
        rewards = np.zeros(nEpisodes)
        self.n = np.zeros((self.mdp.nActions, self.mdp.nStates))
        for episode in range(nEpisodes):
            state = s0
            totalReward = 0
            for step in range(nSteps):
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.mdp.nActions)
                else:
                    if temperature != 0:
                        prob = np.exp(Q[:, state] / temperature)
                        prob /= np.sum(prob)
                        action = np.random.choice(np.arange(self.mdp.nActions), p=prob)
                    else:
                        action = np.argmax(Q[:, state])

                reward, nextState = self.sampleRewardAndNextState(state, action)
                totalReward += (self.mdp.discount ** step) * reward

                # Update the visit count and calculate the learning rate
                self.n[action, state] += 1
                alpha = 1 / self.n[action, state]

                # Update the Q-value
                Q[action, state] = Q[action, state] + alpha * (
                            reward + self.mdp.discount * np.max(Q[:, nextState]) - Q[action, state])

                state = nextState
            rewards[episode] = totalReward
        policy = np.argmax(Q, axis=0)

        return [Q, policy, rewards]

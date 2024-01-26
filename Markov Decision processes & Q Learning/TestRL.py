import numpy as np
import MDP
import RL
import matplotlib.pyplot as plt


''' Construct simple MDP as described in Lecture 16 Slides 21-22'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9
mdp = MDP.MDP(T,R,discount)
rlProblem = RL.RL(mdp,np.random.normal)

# Test Q-learning
[Q,policy,rewards] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=1000,nSteps=100,epsilon=0.3)
print("\nQ-learning results")
print(Q)
print(policy)


epsilons = [0.05, 0.1, 0.3, 0.5]
nTrials = 100
nEpisodes = 200
nSteps = 100
rewards = np.zeros((len(epsilons), nEpisodes))

for i, epsilon in enumerate(epsilons):
    for _ in range(nTrials):
        [_, _, trialRewards] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=nEpisodes,nSteps=nSteps,epsilon=epsilon)
        rewards[i] += trialRewards
    rewards[i] /= nTrials
    [Q, policy, _] = rlProblem.qLearning(s0=0, initialQ=np.zeros([mdp.nActions, mdp.nStates]), nEpisodes=nEpisodes,
                                         nSteps=nSteps, epsilon=epsilon)
    print("\nQ-values for epsilon={}:".format(epsilon))
    print(Q)
    print("Policy for epsilon={}:".format(epsilon))
    print(policy)


plt.figure(figsize=(15, 6))
for i, epsilon in enumerate(epsilons):
    plt.plot(rewards[i], label='epsilon={}'.format(epsilon))
    print(epsilon)
    print(rewards[i])
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Average of cumulative discounted rewards')
plt.show()
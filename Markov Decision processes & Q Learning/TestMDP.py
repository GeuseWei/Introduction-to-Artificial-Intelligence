from MDP import *

''' Construct simple MDP as described in Lecture 16 Slides 21-22'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
policy = mdp.extractPolicy(V)
print("Value Iteration:\nPolicy = {}".format(policy))
print("V = {}\n# of iterations = {}\n".format(V, nIterations))
V = mdp.evaluatePolicy(np.array([1,0,1,0]))
[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print("Policy Iteration:\nPolicy = {}\nV = {}\n# of iterations = {}\n".format(policy, V, iterId))
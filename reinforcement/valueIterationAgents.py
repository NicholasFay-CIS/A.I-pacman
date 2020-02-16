# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        neg_inf = -1000000
        for i in range(0, self.iterations):
            new_vals = self.values.copy() 
            states = self.mdp.getStates()
            for j in range(0, len(states)):
                terminal = self.mdp.isTerminal(states[j])
                if not terminal:
                    actions = self.mdp.getPossibleActions(states[j])
                    best = neg_inf
                    for k in range(0, len(actions)):
                        best = max(self.getQValue(states[j],actions[k]), best)
                    new_vals[states[j]] = best
            self.values = new_vals
        return None

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        states = self.mdp.getTransitionStatesAndProbs(state, action)
        for i in range(0, len(states)):
            trans_state, p = states[i]
            reward = self.mdp.getReward(state, action, trans_state)
            t_state_val = self.values[trans_state]
            q_value += p*(reward+self.discount*t_state_val)
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        pol = util.Counter()
        pos_actions = self.mdp.getPossibleActions(state)
        for i in range(0, len(pos_actions)):
            pol[pos_actions[i]] = self.getQValue(state, pos_actions[i])
        best_args = pol.argMax()
        return best_args

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        neg_inf = -9999999
        #get all the states
        states = self.mdp.getStates()
        #get all the iterations
        iterations = self.iterations
        for i in range(iterations):
            stateIndex = i % len(states)
            #retrieve the current state at the ith iteration
            curState = states[stateIndex]
            #check for terminal state
            if self.mdp.isTerminal(curState):
                continue
            #get all the actions for the current state
            actions = self.mdp.getPossibleActions(curState)
            maxVal = neg_inf
            #iterate through all the possible actions
            for action in actions:
                #find which action is the best 
                qVal = self.getQValue(curState, action)
                maxVal = max(maxVal, qVal)
            self.values[curState] = maxVal
        return

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pQueue = util.PriorityQueue()
        allStates = self.mdp.getStates()
        iterations = self.iterations
        visitedStates = dict()

        for state in allStates:
            maxVal = -9999999
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                prevStates = self.mdp.getTransitionStatesAndProbs(state, action)
                for pS, prob in prevStates:
                    if pS not in visitedStates:
                        visitedStates[pS] = {state}
                    else:
                        visitedStates[pS].add(state)
        
                tempVal = self.computeQValueFromValues(state, action)
                maxVal = max(maxVal, tempVal)
            diff = -1 * abs(self.values[state] - maxVal)
            pQueue.update(state, diff)

        for i in range(iterations):
            if pQueue.isEmpty():
                break
            state = pQueue.pop()
            if self.mdp.isTerminal(state):
                continue
            else:
                maxVal = -9999999
                for action in self.mdp.getPossibleActions(state):
                    tempVal = self.computeQValueFromValues(state, action)
                    maxVal = max(maxVal, tempVal)
                self.values[state] = maxVal
            for vState in visitedStates[state]:
                if self.mdp.isTerminal(vState):
                    continue
                maxVal = -9999999
                for action in self.mdp.getPossibleActions(vState):
                    tempVal = self.computeQValueFromValues(vState, action)
                    maxVal = max(maxVal, tempVal)

                diff = abs(self.values[vState] - maxVal)
                if diff > self.theta:
                    diff = diff * -1
                    pQueue.update(vState, diff)
                



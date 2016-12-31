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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
    


    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        
        for i in range(self.iterations):
            newValues = self.values.copy()
            states = self.mdp.getStates()
            
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                if not len(actions) == 0: #  if there is at leat one action
                    QVAL = []
                    for action in actions:# for each action , I am calculating the Q val
 
                        QVAL.append(self.computeQValueFromValues(state,action))
                            #if newValues[state] < value:
                    newValues[state] = max(QVAL) #V(a) = max ( V(previous a) , value )

            self.values = newValues
    

    
            

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
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0
        for nextState, T in transitions:
            value += T * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        policies = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        
        for action in actions:
            policies[action] = self.getQValue(state, action)
        
        return policies.argMax()

        previous = float('-inf')
        maxAction = None
        for action in actions:
            
            a = self.computeQValueFromValues(state,action)
            if previous < a:
                maxAction = action
            previous = a


        return maxAction


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
        
        for i in range(self.iterations):

            newValues = self.values.copy()
            states = self.mdp.getStates()
            state = states[i % len(states)]
           
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            if not len(actions) == 0: #  if there is at leat one action
                QVAL = []
                for action in actions:# for each action , I am calculating the Q val
                    
                    QVAL.append(self.computeQValueFromValues(state,action))
                #if newValues[state] < value:
                newValues[state] = max(QVAL) #V(a) = max ( V(previous a) , value )
    
            self.values = newValues
        
        
        #if self.mdp.isTerminal(state):
        #continue
        
        "*** YOUR CODE HERE ***"

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
        states = self.mdp.getStates()


        # compute predecesssors of all states in a set sp that we do not have duplicates
        predecessors = {} #as dictionary : key : state : values - preescessors
        
        for state in states:
            predecessors[state] = set()


        for state in states:
            if not self.mdp.isTerminal(state):
            # Pre-process predcessors
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    for successor, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        if probability > 0:
                            predecessors[successor].add(state)



        #initialize an empty prioroty queue
        pQueue = util.PriorityQueue()

        # For each state
        

        for state in states:
            currentVal = self.values[state]
            actions = self.mdp.getPossibleActions(state)
            if not len(actions) == 0:
                QVAL = []
                for action in actions:
                    QVAL.append(self.getQValue(state , action))
                maxQvalue = max(QVAL)
                diff = abs(maxQvalue - currentVal)
                pQueue.push(state,-diff)

        for i in range(self.iterations):
            if pQueue.isEmpty():
                break
            tempstate = pQueue.pop()
            if not self.mdp.isTerminal(tempstate):
                actions = self.mdp.getPossibleActions(tempstate)
                if not len(actions) == 0:
                    QVAL =[]
                    for action in actions:
                        QVAL.append(self.getQValue(tempstate , action))
                    self.values[tempstate] = max(QVAL)


            for predecessor in predecessors[tempstate]:
                currentVal = self.values[predecessor]
                actions = self.mdp.getPossibleActions(predecessor)
                if not len(actions) == 0:
                    QVAL = []
                    for action in actions:
                        QVAL.append(self.getQValue(predecessor , action))
                    maxQvalue = max(QVAL)
                    diff = abs(maxQvalue - currentVal)
                    if diff > self.theta:
                        pQueue.update(predecessor,-diff)



















# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if action == 'Stop':
          return -100000

        distanceToNearestFood = []
        for food in newFood.asList():
          distanceToNearestFood.append(manhattanDistance(food , newPos ))
        
        foodRemaining = len(newFood.asList())
        
        
        distanceToNearestGhost = []
        if newScaredTimes == 0:
          for ghost in newGhostStates:
            pos = ghost.getPosition()
            distanceToNearestGhost.append(manhattanDistance(pos , newPos ))
        else:
          distanceToNearestGhost.append(10000)
        
        if min(distanceToNearestGhost) < 2:
            return -100000
        
        a = 1;
        if not(len(distanceToNearestFood)== 0):
            a = min(distanceToNearestFood)
        
        

        return 1000 / a + 100 * successorGameState.getScore()  + 1 / (foodRemaining + 1 )



        
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    
    def value(self, gameState, turn):
    #depth = 1 means that we go through pacman and all the ghosts once -> each one get a turn
      if gameState.isWin() or turn == self.depth * gameState.getNumAgents() or gameState.isLose():
        return (self.evaluationFunction(gameState), None)
      if turn % gameState.getNumAgents() == 0:
        return self.maxValue(gameState, turn)
      else:
        return self.minValue(gameState, turn)

    def minValue(self, gameState, turn):
      index  = turn % gameState.getNumAgents()
      actions = gameState.getLegalActions(index)
      minOfValues= (99999999, None)

      if actions == []: return (self.evaluationFunction(gameState), None)

      for a in actions:
        successor = gameState.generateSuccessor(index, a)
        successor_value = self.value(successor, turn + 1)
        if successor_value[0] < minOfValues[0]:
          minOfValues = (successor_value[0], a)
      return minOfValues


    def maxValue(self, gameState, turn):
      index  = turn % gameState.getNumAgents()
      actions = gameState.getLegalActions(index)
      maxOfValues= (-99999999, None)

      if actions == []: return (self.evaluationFunction(gameState), None)

      for a in actions:
        successor = gameState.generateSuccessor(index, a)
        successor_value = self.value(successor, turn + 1)
        if successor_value[0] > maxOfValues[0]:
          maxOfValues = (successor_value[0], a)
      return maxOfValues



    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0)[1]
        util.raiseNotDefined()
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def value(self, gameState, alpha, beta, turn):
    #depth = 1 means that we go through pacman and all the ghosts once -> each one get a turn
      if gameState.isWin() or turn == self.depth * gameState.getNumAgents() or gameState.isLose():
        return (self.evaluationFunction(gameState), None)
      if turn % gameState.getNumAgents() == 0:
        return self.maxValue(gameState, alpha, beta, turn)
      else:
        return self.minValue(gameState, alpha, beta, turn)

    def minValue(self, gameState, alpha, beta, turn):
      index  = turn % gameState.getNumAgents()
      actions = gameState.getLegalActions(index)
      minOfValues= (99999999, None)

      if actions == []: return (self.evaluationFunction(gameState), None)

      for a in actions:
        successor = gameState.generateSuccessor(index, a)
        successor_value = self.value(successor, alpha, beta, turn + 1)
        if successor_value[0] < minOfValues[0]:
          minOfValues = (successor_value[0], a)
        if minOfValues[0] < alpha: return minOfValues
        beta = min(beta, minOfValues[0])
      return minOfValues


    def maxValue(self, gameState, alpha, beta, turn):
      index  = turn % gameState.getNumAgents()
      actions = gameState.getLegalActions(index)
      maxOfValues= (-99999999, None)

      if actions == []: return (self.evaluationFunction(gameState), None)

      for a in actions:
        successor = gameState.generateSuccessor(index, a)
        successor_value = self.value(successor, alpha, beta, turn + 1)
        if successor_value[0] > maxOfValues[0]:
          maxOfValues = (successor_value[0], a)
        if maxOfValues[0] > beta: return maxOfValues
        alpha = max(maxOfValues[0], alpha) 
      return maxOfValues

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, -999999, +9999999,0)[1]
      
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def value(self, gameState, turn):
      if gameState.isWin() or turn == self.depth * gameState.getNumAgents() or gameState.isLose():
        return (self.evaluationFunction(gameState), None)
      if turn % gameState.getNumAgents() == 0:
        return self.maxValue(gameState, turn)
      else:
        return self.expValue(gameState, turn)

    def expValue(self, gameState, turn):
      index  = turn % gameState.getNumAgents()
      actions = gameState.getLegalActions(index)

      if actions == []: return (self.evaluationFunction(gameState), None)
      average = 0.0
      for a in actions:
        successor = gameState.generateSuccessor(index, a)
        successor_value = self.value(successor, turn + 1)
        average += successor_value[0]

      return (average * 1.0 / len(actions), None)


    def maxValue(self, gameState, turn):
      index  = turn % gameState.getNumAgents()
      actions = gameState.getLegalActions(index)
      maxOfValues= (-99999999, None)

      if actions == []: return (self.evaluationFunction(gameState), None)

      for a in actions:
        successor = gameState.generateSuccessor(index, a)
        successor_value = self.value(successor, turn + 1)
        if successor_value[0] > maxOfValues[0]:
          maxOfValues = (successor_value[0], a)
      return maxOfValues

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      1- Ghosts:
            a. get the distances to all ghosts
            b. if dist == 0 then we lost
            c. if ghosts not scared:
                i. utility must be negative (f1)
            d. if ghosts scared:
                i. utility either 0 or func of distance (f2)
      2- Food:
            a. get how many food remain
            b. get the distances to all Food
            c. closest food utility is inversly proportional to the distance 
            
      3- Pellets: as Food
            a. if there are pallets we need to go get them
            
    """
    newPos = currentGameState.getPacmanPosition()
    newPellets = currentGameState.getCapsules()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newScore = 0
    ghostDist = [util.manhattanDistance(newPos, Ghost.getPosition()) for Ghost in newGhostStates]

    # Base cases:
    for dist in ghostDist:
      if dist == 0: return -1000
    if currentGameState.getNumFood() == 0: return 1000

    # Ghost Part:
    ghostUtilityList = []
    for i in range(len(newGhostStates)):
      if newScaredTimes[i] > 0: #not a threat
        ghostUtilityList.append(5 * 1.0 /ghostDist[i])        
      else:
        ghostUtilityList.append(-15 + ghostDist[i] ** 2 if ghostDist[i] < 3 else -1.0 /ghostDist[i])
    ghostUtility = sum(ghostUtilityList)
    
    if all(timeLeft > 1 for timeLeft in newScaredTimes):
      ghostUtility = -1 * ghostUtility

    

    # Food Part:
    foodFactor = -2
    foodRemaining = len(newFood)    
    foodDist = [util.manhattanDistance(newPos, food) for food in newFood]
    foodUtility = 100
    if foodRemaining > 0:
      foodUtility = 1.0 / min(foodDist)

    # Pallet Part:
    pelletRemaining = len(newPellets)
    pelletDist = [util.manhattanDistance(newPos, pellet) for pellet in newPellets]
    pelletUtility = 0 
    if pelletRemaining > 0:
      pelletUtility = 1.0 / min(pelletDist)
    
    pelletFactor = 0
    scaredTime = sum(newScaredTimes)
    if scaredTime == 0:
      pelletFactor = -10

#    if newScaredTimes[ghostDist.index(min(ghostDist))] > 0:
#      ghostUtility = 100 / ( 1 + min(ghostDist))
    ''''  
    print  'F: ', foodUtility
    print 'g: ', 2 * ghostUtility
    print 'p', 10 * pelletUtility
    print 'pf', foodFactor * foodRemaining
    print 'pp', pelletRemaining * pelletFactor 
    '''
    return  currentGameState.getScore() + foodUtility + 2 * ghostUtility + 10 * pelletUtility + foodFactor * foodRemaining + pelletRemaining * pelletFactor
    '''

    food_rem_punish = -1.5
    pelete_re_punish = -8 if all((t == 0 for t in new_scared_times)) else 0
    if all((t > 0 for t in new_scared_times)):
        ghost_k *= (-1)
   


    score = score + near_food_bonus + 2 * ghost_k + 10 * near_pelet_bonus + food_rem_punish * food_count + peletsRemaining * pelete_re_punish
    return score
    '''
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction







'''''

'''''
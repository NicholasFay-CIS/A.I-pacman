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
from collections import namedtuple
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        "*** YOUR CODE HERE ***"
        curPacman = currentGameState.getPacmanPosition()
        distFromFood = set()
        nextPacman = list(newPos)
        menu = newFood.asList()

        if (action == "Stop"):
           return -9999999

        for mob in newGhostStates:
            ghost = mob.getPosition()
            currentGhostDist = manhattanDistance(ghost, curPacman)
            if (currentGhostDist < 1.0):
                return -9999
            nextGhostDist = manhattanDistance(newPos, ghost)
            if (nextGhostDist < 1.0):
                return -9999
        for food in menu:
            #numFood = currentGameState.getNumFood()
            #newNumFood = successorGameState.getNumFood()
            curFoodDist = manhattanDistance(curPacman, food)
            nextFoodDist = manhattanDistance(nextPacman, food)
            val = curFoodDist - nextFoodDist
            numfood = currentGameState.getNumFood()
            newNumFood = successorGameState.getNumFood()
            if (numfood > newNumFood):
                return 9999
            else:
                distFromFood.add(val)
        try:
            return min(distFromFood)
        except:
            return 2

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
        
        max_val = -9999999
        pac_man_actions = gameState.getLegalActions(0)
        best_action = None
        for action in pac_man_actions:
            curr_depth = 0
            gamestate_successor_move = gameState.generateSuccessor(0, action)
            next_state = gamestate_successor_move
            next_value = self.get_value(next_state, curr_depth, 1)
            max_value = max(next_value, max_val)
            if(max_value == next_value):
                max_val = next_value
                best_action = action
        return best_action

    def get_value(self, gameState, init_depth, agent):
        terminal_states = namedtuple('terminal', 'win lose')
        terminal_states.win = gameState.isWin()
        terminal_states.lose = gameState.isLose()
        if(terminal_states.win == True):
            return self.evaluationFunction(gameState)
        if(terminal_states.lose == True):
            return self.evaluationFunction(gameState)
        if(init_depth == self.depth):
            return self.evaluationFunction(gameState)
        if(agent):
            return self.min_value(gameState, init_depth, agent)
        return self.max_value(gameState, init_depth)

    def max_value(self, gameState, depth):
        maxVal = -999999
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            newVal = self.get_value(successor, depth, 1)
            maxVal = max(maxVal, newVal)
            if(newVal > maxVal):
                maxVal = newVal
                continue
            maxVal = maxVal
        return maxVal

    def min_value(self, gameState, currentDepth, agntInd):
        minVal = 999999
        numAgents = gameState.getNumAgents() - 1
        for action in gameState.getLegalActions(agntInd):
            if agntInd == numAgents:
                successor = gameState.generateSuccessor(agntInd, action)
                newVal = self.get_value(successor, currentDepth + 1, 0)
                minVal = min(minVal, newVal)
                continue
            successor = gameState.generateSuccessor(agntInd, action)
            newVal = self.get_value(successor, currentDepth, agntInd + 1)
            minVal = min(minVal, newVal)
        return minVal


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        max_value = -999999
        max_a= -999999
        min_b = 999999
        best_action = None
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            next_state = gameState.generateSuccessor(0, action)
            next_value = self.get_value(next_state, 0, 1, max_a, min_b)
            max_value= max(next_value, max_value)
            if(max_value == next_value):
                best_action = action
            max_a = max(max_a, max_value)
        return best_action

    def get_value(self, gameState, init_depth, agent, alpha, beta):
        terminal_states = namedtuple('terminal', 'win lose')
        terminal_states.win = gameState.isWin()
        terminal_states.lose = gameState.isLose()
        if(terminal_states.win == True):
            return self.evaluationFunction(gameState)
        if(terminal_states.lose == True):
            return self.evaluationFunction(gameState)
        if(init_depth == self.depth):
            return self.evaluationFunction(gameState)
        if(agent):
            return self.min_value(gameState, init_depth, agent, alpha, beta)
        return self.max_value(gameState, init_depth, alpha, beta)

    def max_value(self, gameState, init_depth, alpha, beta):
        max_value = -999999
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            next_value = self.get_value(gameState.generateSuccessor(0, action), init_depth, 1, alpha, beta)
            max_value = max(max_value, next_value)
            is_greater = max(max_value, beta)
            if(is_greater == beta):
                alpha = max(alpha, max_value)
            else:
                return max_value
        return max_value

    def min_value(self, gameState, currentDepth, agntInd, alpha, beta):
        minVal = 999999
        numAgents = gameState.getNumAgents() - 1
        for action in gameState.getLegalActions(agntInd):
            if agntInd == numAgents:
                successor = gameState.generateSuccessor(agntInd, action)
                newVal = self.get_value(successor, currentDepth + 1, 0, alpha, beta)
                minVal = min(minVal, newVal)
                minVal = min(minVal, newVal)
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)
                continue
            successor = gameState.generateSuccessor(agntInd, action)
            newVal = self.get_value(successor, currentDepth, agntInd + 1, alpha, beta)
            minVal = min(minVal, newVal)
            if minVal < alpha:
                return minVal
            beta = min(beta, minVal)
        return minVal

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        max_val = -9999999
        pac_man_actions = gameState.getLegalActions(0)
        best_action = None
        for action in pac_man_actions:
            curr_depth = 0
            gamestate_successor_move = gameState.generateSuccessor(0, action)
            new_value = self.get_value(gamestate_successor_move, curr_depth, 1)
            max_value = max(new_value, max_val)
            if(max_value == new_value):
                max_val = new_value
                best_action = action
        return best_action

    def get_value(self, gameState, init_depth, agent):
        terminal_states = namedtuple('terminal', 'win lose')
        terminal_states.win = gameState.isWin()
        terminal_states.lose = gameState.isLose()
        if(terminal_states.win == True):
            return self.evaluationFunction(gameState)
        if(terminal_states.lose == True):
            return self.evaluationFunction(gameState)
        if(init_depth == self.depth):
            return self.evaluationFunction(gameState)
        if(agent):
            return self.avg_value(gameState, init_depth, agent)
        return self.max_value(gameState, init_depth)

    def max_value(self, gameState, currentDepth):
        maxVal = -999999
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            newVal = self.get_value(successor, currentDepth, 1)
            if (newVal > maxVal):
                maxVal = newVal
        return maxVal

    def avg_value(self, gameState, currentDepth, agntInd):
        avgVal = 0
        numAgents = gameState.getNumAgents() - 1
        for action in gameState.getLegalActions(agntInd):
            if agntInd == numAgents:
                successor = gameState.generateSuccessor(agntInd, action)
                newVal = self.get_value(successor, currentDepth + 1, 0)
                avgVal += newVal
            else:
                successor = gameState.generateSuccessor(agntInd, action)
                newVal = self.get_value(successor, currentDepth, agntInd + 1)
                avgVal += newVal
        return avgVal

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

# Abbreviation
better = betterEvaluationFunction

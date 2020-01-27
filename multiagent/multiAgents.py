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
        #best action is currently unknown
        best_action = None
        #iterate through all possible actions for pacman
        for action in pac_man_actions:
            curr_depth = 0
            #get the successor state using the action
            gamestate_successor_move = gameState.generateSuccessor(0, action)
            next_state = gamestate_successor_move
            #get the value of the move
            next_value = self.get_value(next_state, curr_depth, 1)
            #if the move has a greater value than the max, set it and its action to max/best
            if(next_value >= max_val):
                max_val = next_value
                best_action = action
        return best_action

    def get_value(self, gameState, init_depth, agent):
        terminal_states = namedtuple('terminal', 'win lose')
        terminal_states.win = gameState.isWin()
        terminal_states.lose = gameState.isLose()
        #check if state is a win or a terminal state
        if(terminal_states.win == True):
            return self.evaluationFunction(gameState)
        if(terminal_states.lose == True):
            return self.evaluationFunction(gameState)
        if(init_depth == self.depth):
            return self.evaluationFunction(gameState)
        #if agent is not pacman, find the min value
        if(agent):
            return self.min_value(gameState, init_depth, agent)
        #if agent is pacman, find the max value
        return self.max_value(gameState, init_depth)

    def max_value(self, gameState, depth):
        max_val = -999999
        #iterate through actions for pacman
        for action in gameState.getLegalActions(0):
            #look through pacmans successors and find the max value
            successor = gameState.generateSuccessor(0, action)
            new_val = self.get_value(successor, depth, 1)
            if(new_val > max_val):
                max_val = new_val
                continue
        return max_val

    def min_value(self, gameState, currentDepth, agent):
        min_val = 999999
        #get the number of agents
        numAgents = gameState.getNumAgents() - 1
        for action in gameState.getLegalActions(agent):
            if agent == numAgents:
                #iterate through the successors, checking the min value for each move
                successor = gameState.generateSuccessor(agent, action)
                #in this iteration, the last agent has been reached so increase the depth
                new_val = self.get_value(successor, currentDepth + 1, 0)
                if (new_val < min_val):
                    min_val = new_val
            else:
                #iterate through the successors, checking the min value for each move
                successor = gameState.generateSuccessor(agent, action)
                #in this iteration, increment the agent to check their other moves
                new_val = self.get_value(successor, currentDepth, agent + 1)
                if (new_val < min_val):
                    min_val = new_val
        return min_val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        neg_inf = -999999
        inf = 999999
        #current best action is unknown
        best_action = None
        max_value = neg_inf
        #set alpha to neg infinity so we can find the max
        max_a = neg_inf
        #set beta to inf so we can find the min
        min_b = inf
        legal_actions = gameState.getLegalActions(0)
        #iterate through all the legal gamestate actions 
        for action in legal_actions:
            #get the successor state
            next_state = gameState.generateSuccessor(0, action)
            #get the value of that state
            next_value = self.get_value(next_state, 0, 1, max_a, min_b)
            #find the max
            max_value = max(next_value, max_value)
            #if the next state yields the max value
            if(max_value == next_value):
                #that is the best action so far
                best_action = action
            #adjust the alpha var according to the max value and current alpha
            max_a = max(max_a, max_value)
        return best_action

    def get_value(self, gameState, init_depth, agent, alpha, beta):
        terminal_states = namedtuple('terminal', 'win lose')
        terminal_states.win = gameState.isWin()
        terminal_states.lose = gameState.isLose()
        #check if it is terminal state one or win state
        if(terminal_states.win == True):
            return self.evaluationFunction(gameState)
        #check if it is terminal state two or lose state
        if(terminal_states.lose == True):
            return self.evaluationFunction(gameState)
        #check if the current depth is our class depth
        if(init_depth == self.depth):
            return self.evaluationFunction(gameState)
        #if the agent is not pacman
        if(agent):
            return self.min_value(gameState, init_depth, agent, alpha, beta)
        #otherwise it is pacman
        return self.max_value(gameState, init_depth, alpha, beta)

    def max_value(self, gameState, init_depth, alpha, beta):
        neg_inf = -999999
        #to find our max we compare initially against neg inf
        max_value = neg_inf
        #get the legal actions of the game state for pacman
        legal_actions = gameState.getLegalActions(0)
        #iterate through the legal actions
        for action in legal_actions:
            #generate the value for a ghost
            next_value = self.get_value(gameState.generateSuccessor(0, action), init_depth, 1, alpha, beta)
            #find the max of our new value we just got and our stored max value 
            max_value = max(max_value, next_value)
            #check if our max value is greater than beta
            is_greater = max(max_value, beta)
            if(is_greater == beta):
                #if beta is greater update alpha
                alpha = max(alpha, max_value)
            else:
                #otherwise we found our max so return the max value 
                return max_value
        return max_value

    def min_value(self, gameState, init_depth, agent, alpha, beta):
        inf = 999999
        min_value = inf
        #get the agents, minus one due to indexing concerns 
        num_agents = gameState.getNumAgents() - 1
        #get the legal actions
        legal_actions = gameState.getLegalActions(agent)
        #iterate through the legal actions 
        for action in legal_actions:
            #if the agent does not equal the number of agents
            if(agent != num_agents):
                #get the successor state of the agent and the legal action
                successor = gameState.generateSuccessor(agent, action)
                #get the value of the successor with the next agent
                new_value = self.get_value(successor, init_depth, agent + 1, alpha, beta)
                #set the min value to be the smallest value
                min_value = min(min_value, new_value)
                #if min value is less than alpha then we have our min value
                if(min_value < alpha):
                    return min_value
                #if min value is less than beta updata beta
                if(min_value < beta):
                    beta = min_value
                continue
            else:
                #get successor state and action of the agent if it is the same number
                successor = gameState.generateSuccessor(agent, action)
                #get the value by changing the depth count and use pacman as the agent
                new_value = self.get_value(successor, init_depth + 1, 0, alpha, beta)
                #get min value
                min_value = min(min_value, new_value)
                #if alpha is greater than our value return we have the min value
                if(min_value < alpha):
                    return min_value
                #update beta if neccessary
                if(min_value < beta):
                    beta = min_value
        return min_value

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
        #iterate through pacmans legal actions
        for action in pac_man_actions:
            curr_depth = 0
            gamestate_successor_move = gameState.generateSuccessor(0, action)
            #get the value for each successor and check against the max
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
        #check if state is terminal or a win
        if(terminal_states.win == True):
            return self.evaluationFunction(gameState)
        if(terminal_states.lose == True):
            return self.evaluationFunction(gameState)
        if(init_depth == self.depth):
            return self.evaluationFunction(gameState)
        #if the agent is a ghost, find the average value
        if(agent):
            return self.avg_value(gameState, init_depth, agent)
        #if the agent is pacman, find the max value
        return self.max_value(gameState, init_depth)

    def max_value(self, gameState, currentDepth):
        max_val = -999999
        for action in gameState.getLegalActions(0):
            #check the value for each successor
            successor = gameState.generateSuccessor(0, action)
            new_val = self.get_value(successor, currentDepth, 1)
            #check value against the amx
            if (new_val > max_val):
                max_val = new_val
        return max_val

    def avg_value(self, gameState, currentDepth, agntInd):
        avg_val = 0
        numAgents = gameState.getNumAgents() - 1
        for action in gameState.getLegalActions(agntInd):
            if agntInd == numAgents:
                successor = gameState.generateSuccessor(agntInd, action)
                #increment depth if you are at last agent
                new_val = self.get_value(successor, currentDepth + 1, 0)
                #increment average for each agent
                avg_val += new_val
            else:
                successor = gameState.generateSuccessor(agntInd, action)
                #increment agent id until we reach the final one
                new_val = self.get_value(successor, currentDepth, agntInd + 1)
                #increment average for each agent
                avg_val += new_val
        return avg_val

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

# Abbreviation
better = betterEvaluationFunction

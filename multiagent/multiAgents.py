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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        inf = 9999999
        neg_inf = -9999999
        curPacman = currentGameState.getPacmanPosition()
        distFromFood = set()
        nextPacman = list(newPos)
        menu = newFood.asList()
        i = 0

        if (action == "Stop"):
           return neg_inf

        for mob in newGhostStates:
            ghost = mob.getPosition()
            currentGhostDist = manhattanDistance(ghost, curPacman)
            if (currentGhostDist < 1.0):
                return neg_inf
            nextGhostDist = manhattanDistance(newPos, ghost)
            if (nextGhostDist < 1.0):
                return neg_inf
        while(i < len(menu)):
            curFoodDist = manhattanDistance(curPacman, menu[i])
            nextFoodDist = manhattanDistance(nextPacman, menu[i])
            val = curFoodDist - nextFoodDist
            numfood = currentGameState.getNumFood()
            newNumFood = successorGameState.getNumFood()
            if (numfood > newNumFood):
                return inf
            else:
                distFromFood.add(val)
            i += 1
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
        neg_inf = -9999999
        max_val = neg_inf
        pac_man_actions = gameState.getLegalActions(0)
        #best action is currently unknown
        best_action = None
        #iterate through all possible actions for pacman
        i = 0
        while(i < len(pac_man_actions)):
            curr_depth = 0
            #get the successor state using the action
            gamestate_successor_move = gameState.generateSuccessor(0, pac_man_actions[i])
            next_state = gamestate_successor_move
            #get the value of the move
            next_value = self.get_value(next_state, 1, curr_depth)
            #if the move has a greater value than the max, set it and its action to max/best
            if(next_value >= max_val):
                max_val = next_value
                best_action = pac_man_actions[i]
            i += 1
        return best_action

    def get_value(self, game_state, agent, init_depth):
        terminal_states = namedtuple('terminal', 'win lose')
        terminal_states.win = game_state.isWin()
        terminal_states.lose = game_state.isLose()
        #check if state is a win or a terminal state
        if(terminal_states.win == True):
            return self.evaluationFunction(game_state)
        if(terminal_states.lose == True):
            return self.evaluationFunction(game_state)
        if(init_depth == self.depth):
            return self.evaluationFunction(game_state)
        #if agent is not pacman, find the min value
        if(agent):
            min_value = self.min_value(game_state,  agent, init_depth)
            return min_value
        #if agent is pacman, find the max value
        max_value = self.max_value(game_state, init_depth)
        return max_value

    def min_value(self, game_state, agent, init_depth):
        inf = 999999
        min_val = inf
        #get the number of agents
        agent_count = game_state.getNumAgents() - 1
        legal_actions = game_state.getLegalActions(agent)
        i = 0
        while(i < len(legal_actions)):
            if agent == agent_count:
                #iterate through the successors, checking the min value for each move
                successor = game_state.generateSuccessor(agent, legal_actions[i])
                #in this iteration, the last agent has been reached so increase the depth
                new_depth = init_depth + 1
                new_val = self.get_value(successor, 0, new_depth)
                if (new_val < min_val):
                    min_val = new_val
            else:
                #iterate through the successors, checking the min value for each move
                successor = game_state.generateSuccessor(agent, legal_actions[i])
                #in this iteration, increment the agent to check their other moves
                next_agent = agent + 1
                new_val = self.get_value(successor, next_agent, init_depth)
                if (new_val < min_val):
                    min_val = new_val
            i += 1
        return min_val

    def max_value(self, game_state, depth):
        neg_inf = -999999
        max_val = neg_inf
        #iterate through actions for pacman
        legal_actions = game_state.getLegalActions(0)
        i = 0
        while(i < len(legal_actions)):
            #look through pacmans successors and find the max value
            successor = game_state.generateSuccessor(0, legal_actions[i])
            new_val = self.get_value(successor, 1, depth)
            if(new_val > max_val):
                max_val = new_val
                i += 1
                continue
            i += 1
        return max_val

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
        i = 0
        #iterate through all the legal gamestate actions 
        while(i < len(legal_actions)):
            #get the successor state
            next_state = gameState.generateSuccessor(0, legal_actions[i])
            #get the value of that state
            next_value = self.get_value(next_state, max_a, min_b, 1, 0)
            #find the max
            max_value = max(next_value, max_value)
            #if the next state yields the max value
            if(max_value == next_value):
                #that is the best action so far
                best_action = legal_actions[i]
            #adjust the alpha var according to the max value and current alpha
            max_a = max(max_a, max_value)
            i += 1
        return best_action

    def get_value(self, gameState,  alpha, beta, agent, init_depth):
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
            min_value = self.min_value(gameState, alpha, beta, agent, init_depth)
            return min_value
        #otherwise it is pacman
        max_value = self.max_value(gameState, alpha, beta, init_depth)
        return max_value

    def min_value(self, gameState, alpha, beta, agent, init_depth):
        inf = 999999
        min_value = inf
        #get the agents, minus one due to indexing concerns 
        num_agents = gameState.getNumAgents() - 1
        #get the legal actions
        legal_actions = gameState.getLegalActions(agent)
        #iterate through the legal actions 
        i = 0
        while(i < len(legal_actions)):
            #if the agent does not equal the number of agents
            if(agent != num_agents):
                #get the successor state of the agent and the legal action
                successor = gameState.generateSuccessor(agent, legal_actions[i])
                #get the value of the successor with the next agent
                next_agent = agent + 1
                new_value = self.get_value(successor, alpha, beta, next_agent, init_depth)
                #set the min value to be the smallest value
                min_value = min(min_value, new_value)
                #if min value is less than alpha then we have our min value
                if(min_value < alpha):
                    return min_value
                #if min value is less than beta updata beta
                if(min_value < beta):
                    beta = min_value
                i += 1
                continue
            else:
                #get successor state and action of the agent if it is the same number
                successor = gameState.generateSuccessor(agent, legal_actions[i])
                #get the value by changing the depth count and use pacman as the agent
                new_depth = init_depth + 1
                new_value = self.get_value(successor, alpha, beta, 0, new_depth)
                #get min value
                min_value = min(min_value, new_value)
                #if alpha is greater than our value return we have the min value
                if(min_value < alpha):
                    return min_value
                #update beta if neccessary
                if(min_value < beta):
                    beta = min_value
                i += 1
        return min_value

    def max_value(self, gameState, alpha, beta, init_depth):
        neg_inf = -999999
        #to find our max we compare initially against neg inf
        max_value = neg_inf
        #get the legal actions of the game state for pacman
        legal_actions = gameState.getLegalActions(0)
        #iterate through the legal actions
        i = 0
        while (i < len(legal_actions)):
            #generate the value for a ghost
            next_value = self.get_value(gameState.generateSuccessor(0, legal_actions[i]),  alpha, beta, 1, init_depth)
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
            i += 1
        return max_value

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
        neg_inf = -9999999
        max_val = neg_inf
        pac_man_actions = gameState.getLegalActions(0)
        best_action = None
        i = 0
        #iterate through pacmans legal actions
        while(i < len(pac_man_actions)):
            curr_depth = 0
            gamestate_successor_move = gameState.generateSuccessor(0, pac_man_actions[i])
            #get the value for each successor and check against the max
            new_value = self.get_value(gamestate_successor_move, 1, curr_depth)
            max_value = max(new_value, max_val)
            if(max_value == new_value):
                max_val = new_value
                best_action = pac_man_actions[i]
            i += 1
        return best_action

    def get_value(self, game_state, agent, init_depth):
        terminal_states = namedtuple('terminal', 'win lose')
        terminal_states.win = game_state.isWin()
        terminal_states.lose = game_state.isLose()
        #check if state is terminal or a win
        if(terminal_states.win == True):
            return self.evaluationFunction(game_state)
        if(terminal_states.lose == True):
            return self.evaluationFunction(game_state)
        if(init_depth == self.depth):
            return self.evaluationFunction(game_state)
        #if the agent is a ghost, find the average value
        if(agent):
            exp_value = self.exp_value(game_state, agent, init_depth)
            return exp_value
        #if the agent is pacman, find the max value
        max_value = self.max_value(game_state, init_depth)
        return max_value

    def exp_value(self, game_state, agent, init_depth):
        exp_val = 0
        numAgents = game_state.getNumAgents() - 1
        legal_actions = game_state.getLegalActions(agent)
        i = 0
        while(i < len(legal_actions)):
            if agent == numAgents:
                successor = game_state.generateSuccessor(agent, legal_actions[i])
                #increment depth if you are at last agent
                next_depth = init_depth + 1
                new_val = self.get_value(successor, 0, next_depth)
                #increment average for each agent
                exp_val += new_val
            else:
                successor = game_state.generateSuccessor(agent, legal_actions[i])
                #increment agent id until we reach the final one
                next_agent = agent + 1
                new_val = self.get_value(successor, next_agent, init_depth)
                #increment average for each agent
                exp_val += new_val
            #go to the next index
            i += 1
        return exp_val
    
    def max_value(self, game_state, init_depth):
        neg_inf = -999999
        max_val = neg_inf
        legal_actions = game_state.getLegalActions(0)
        i = 0
        while(i < len(legal_actions)):
            #check the value for each successor
            successor = game_state.generateSuccessor(0, legal_actions[i])
            new_val = self.get_value(successor, 1, init_depth)
            #check value against the max
            max_ = max(new_val, max_val)
            if (new_val == max_):
                max_val = new_val
            i += 1
        return max_val

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: "Prioritize eating the closest food, chasing down scared ghosts, and avoiding nearby ghosts"
    """
    "*** YOUR CODE HERE ***"
    #get pacman location
    pacman = currentGameState.getPacmanPosition()
    #get location of food
    food = currentGameState.getFood()
    menu = food.asList()
    #get location of ghosts and times of scared ghosts
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    curScore = currentGameState.data.score * .5

    #return value "Most Effective Tactics Available"
    meta = 0
    #determine the closest food as well as distance to it
    closestFood = 0
    pacDistFromFood = 9999
    for food in menu:
        foodDist = manhattanDistance(food, pacman)
        if foodDist < pacDistFromFood:
            closestFood = food
            pacDistFromFood = foodDist
    
    if (closestFood):
        #if far away from nearest food, change direction to get closer
        if (pacDistFromFood > 7):
            meta -= pacDistFromFood 
        #if food is nearby, keep going until you can reach it
        elif (pacDistFromFood <= 3):
            meta += pacDistFromFood * .75 
        else:
            meta -= pacDistFromFood * .25
       
    for time in newScaredTimes:
        ghost = newScaredTimes.index(time)
        #location of ghost
        scaredGhost = newGhostStates[ghost].getPosition()
        #calculate distance from ghost and pacman
        scaredDist = manhattanDistance(pacman, scaredGhost)
        if time:
            #if the ghost is scared and right in front of you, you might as well get it
            if scaredDist == 1:
                return 9999999
            elif scaredDist <= 2:
                #very close to the ghost, keep going
                meta += scaredDist * 900
                return meta
            elif scaredDist <= 4:
                #very close to the ghost, keep going
                meta += scaredDist * 400
                return meta
            elif scaredDist <= 9:
                #try to catch it
                meta += scaredDist * 50
        else:
            #you can travel freely while a ghost is scared
            meta += scaredDist

    #now if the ghosts are not scared...
    for mob in newGhostStates:
        #get ghost position and calculate distance
        ghost = mob.getPosition()
        dist = manhattanDistance(ghost, pacman)
        if (max(newScaredTimes) == 0):
            if (dist <= 2):
                #if the ghost is nearby, get out of there
                meta -=  dist * 10
            else:
                #you have time, move freely
                meta += dist * .25

    meta += curScore
    return meta


# Abbreviation
better = betterEvaluationFunction

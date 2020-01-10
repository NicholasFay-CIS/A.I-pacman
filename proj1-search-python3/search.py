# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    # initialize the stack  
    stack = util.Stack()
    #have a list for visted_branches
    visited_branches = []
    #empty list if nothing able to be returned
    empty_list = []
    #get the problem start state
    problem_start_State = problem.getStartState()
    #path list
    path_list = []
    #length
    length = 0 
    #variable to symbolize start state xy position index in info lists
    xy_position = 0
    #variable to symbolize the path list index in info lists
    path_ = 1
    #variable to symbolize length index in info lists
    length_ = 2
    #create the root info list with the start state, path list (of directions) and length
    root_info_list = [problem_start_State, path_list, length]
    #push the info list onto the stack
    stack.push(root_info_list) 
   
   #Forever loop
    while(True):
        #if the stack is empty break out of the loop
        if(stack.isEmpty == False):
            break
        #retrieve a child node
        child_node = stack.pop()
        #get child node position
        child_node_xy = child_node[xy_position]
        #get child node direction
        direction = child_node[path_]
        #check is goal state of child node position 
        is_goal_state_success = problem.isGoalState(child_node_xy)

        #if the goal state is a success return the direction
        if(is_goal_state_success):
            return direction
        #if the goal state is not a success 
        elif(child_node_xy not in visited_branches):
            #append the position to visited branches
            visited_branches.append(child_node_xy)
            #get the successors of the child nodes position
            successors = problem.getSuccessors(child_node_xy)
            #iterate through each successor node
            for successor_node in successors:
                #get that nodes start state
                new_node_startState = successor_node[xy_position]
                #if the start state is not been visited
                if(new_node_startState not in visited_branches):
                    #get the nodes length
                    new_node_length = successor_node[length_]
                    #get the nodes directions
                    new_node_path = direction + [successor_node[path_]]
                    #create the successor info list
                    successor_info_list = [new_node_startState, new_node_path, new_node_length]
                    #push the successor info list to the stack
                    stack.push(successor_info_list)
    #if the stack was empty return an empty list
    return empty_list

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    problem_start_State = problem.getStartState()
    path_list = []
    length = 0 
    root_info_list = [problem_start_State, path_list, length]
    queue.push(root_info_list) 
    visited_branches = []
    empty_list = []
    xy_position = 0
    #variable to symbolize the path list index in info lists
    path_ = 1
    #variable to symbolize length index in info lists
    length_ = 2
   
    while True:

        if(queue.isEmpty == False):
            break

        child_node = queue.pop()
        child_node_xy = child_node[xy_position]
        direction = child_node[path_]
        is_goal_state_success = problem.isGoalState(child_node_xy)

        if(is_goal_state_success):
            return direction
        
        elif(child_node_xy not in visited_branches):
            visited_branches.append(child_node_xy)
            for successor_node in problem.getSuccessors(child_node_xy):
                new_node_startState = successor_node[xy_position]
                if(new_node_startState not in visited_branches):
                    new_node_length = successor_node[length_]
                    new_node_path = direction + [successor_node[path_]]
                    successor_info_list = [new_node_startState, new_node_path, new_node_length]
                    queue.push(successor_info_list)
    return empty_list

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

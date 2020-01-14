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
#import named tuple data structure
from collections import namedtuple

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
    #indeces for start state, path_list (directions) and length
    start_state_index = 0
    path_list_index = 1
    length_index = 2
    #initialize the stack
    stack = util.Stack()
    #create a named tuple for the root
    root_info = namedtuple('root', 'startState path_list length')
    problem_start_state = problem.getStartState() # set the start state
    root_info.startState = problem_start_state 
    root_info.path_list = [] #initialize an empty list for a path of directions
    root_info.length = 0 #initialize the length to be zero
    stack.push(root_info)   #push the info onto the stack (as a single node)
    expanded_nodes = [] #empty list for visted or expanded sections
    empty_list = [] 

    while(True):
        #if the stack is empty break and return the empty list
        if(stack.isEmpty() == True):
            break
        else:
            #if the stack is not empty pop a child node
            child_node = stack.pop()
            #is the goal reached
            reached_goal = problem.isGoalState(child_node.startState)
            #if the goal is not reached
            if(not reached_goal):
                #check if the child node start state has been visited, if so continue
                if(child_node.startState in expanded_nodes):
                    continue
                #if it has not been expanded
                else:
                    #get the successors of the nodes start state
                    successors = problem.getSuccessors(child_node.startState)
                    #iterate through the successors
                    for successor in successors:
                        successor_start_state = successor[start_state_index] #get the start state index of the successor
                        successor_direction_list = [successor[path_list_index]] #get the successors new path list. example) ['West']
                        successor_length = successor[length_index] #get the successors length
                        #check if the successors start state has already been recorded
                        if(successor_start_state in expanded_nodes):
                            #if so continue to the next successor node
                            continue
                        #otherwise it has not been recorded
                        else:
                            #create a next successor named tuple with the same attributes as the named tuple for the root
                            next_successor = namedtuple('successor', 'startState path_list length')
                            next_successor.startState = successor_start_state #set the start state
                            next_successor.path_list = child_node.path_list + successor_direction_list #add the child nodes direction list to the successor nodes direction list. Ex
                            next_successor.length = successor_length #set the length to be the successors length
                            stack.push(next_successor) #push the successor node onto the stack
                    expanded_nodes.append(child_node.startState) #add the child node start state to the visited nodes (nodes we expanded)
            else:
                #if the goal has been reached return the direction list of the poped node
                return child_node.path_list 
    #return empty list if the stack is empty
    return empty_list

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    problem_start_State = problem.getStartState()
    path_list = []
    length = 0 
    root_info= namedtuple('root', 'startState path_list length')
    root_info.startState = problem_start_State
    root_info.path_list = path_list
    root_info.length = length
    queue.push(root_info) 
    visited_branches = []
    empty_list = []
    xy_position = 0
    #variable to symbolize the path list index in info lists
    #variable to symbolize the path list index in info lists
    path_ = 1
    #variable to symbolize length index in info lists
    length_ = 2
    while True:
        if(queue.isEmpty() == True):
            break
        child_node = queue.pop()
        child_node_xy = child_node.startState
        direction = child_node.path_list
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
                    successor_info = namedtuple('successor', 'startState path_list length')
                    successor_info.startState = new_node_startState
                    successor_info.path_list = new_node_path
                    successor_info.length = new_node_length
                    queue.push(successor_info)
    return empty_list

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pQueue = util.PriorityQueue()
    prob_start_state = problem.getStartState()
    visited_branches = []
    og_path_list = []
    root_info= namedtuple('root', 'start_state path_list cost')
    root_info.start_state = prob_start_state
    root_info.path_list = og_path_list
    root_info.cost = 0
    pQueue.push(root_info, 0)

    while(True):
        if (pQueue.isEmpty()):
            break
        else:
            child_node = pQueue.pop()
            print(child_node)
            child_node_xy = child_node.start_state
            direction = child_node.path_list
            child_node_cost = child_node.cost
            if (problem.isGoalState(child_node_xy)):
                return direction
            else:
                if (child_node_xy not in visited_branches):
                    visited_branches.append(child_node_xy)
                    for successor_node in problem.getSuccessors(child_node_xy):
                        if (successor_node[0]not in visited_branches):
                            successor_info = namedtuple ('succ', 'start_state path_list cost')
                            successor_info.start_state = successor_node[0]
                            successor_info.path_list = direction + [successor_node[1]]
                            successor_info.cost = child_node_cost + successor_node[2]
                            pQueue.push(successor_info, successor_info.cost)
    return
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pQueue = util.PriorityQueue()       #frontier
    problem_start_state = problem.getStartState()      #initial node
    cost = 0
    path_list = []
    root_list = [problem_start_state, path_list, cost]   #priority for the root node is 0
    pQueue.push(root_list, 0)
    visited_branches = []           #nodes that have been visited
    while(True):
        if (pQueue.isEmpty()):
            return False

        child_node = pQueue.pop()
        child_node_xy = child_node[0]
        direction = child_node[1]

        if (problem.isGoalState(child_node_xy)):
            return direction

        elif(child_node_xy not in visited_branches):
            visited_branches.append(child_node_xy)
            for successor_node in problem.getSuccessors(child_node_xy):
                new_node_startState = successor_node[0]
                if(new_node_startState not in visited_branches):
                    new_node_path = direction + [successor_node[1]]
                    parent_node_cost = child_node[2]
                    successor_cost = successor_node[2]
                    parent_child_cost = parent_node_cost + successor_cost
                    new_node_list = [new_node_startState, new_node_path, parent_child_cost]
                    total_cost = new_node_list[2] + heuristic(new_node_startState, problem)
                    pQueue.push(new_node_list, total_cost)
                elif(new_node_startState in visited_branches):
                    pQueue.update(successor_node, total_cost)
    return


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

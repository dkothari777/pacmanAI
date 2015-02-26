# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostPositions()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    distances_to_foods = []
    for food in oldFood:
        distances_to_foods.append(manhattanDistance(newPos, food))
        # print newPos, food
    closest_food = max(distances_to_foods)

    distances_to_ghosts=[]
    for i in newGhostStates:
        distances_to_ghosts.append(manhattanDistance(newPos, i))
    closest_ghost = min(distances_to_ghosts)

    state_score = min(closest_ghost,closest_food)
    if closest_ghost-closest_food >= 0:
        state_score += min(closest_food, closest_ghost)

    return currentGameState.getScore() + state_score

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

# node sub class
class Node():

    def __init__(self, value):
      self.value = value
      self.children = []

    def getChildren(self):
        return self.children
    def setNodeValue(self, newValue): # currently not needed/used
        self.value = newValue
    def getNodeValue(self):
        return self.value
    def addChild(self, newNode):
        self.children.append(newNode)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def miniMax(self, node, depth, player):
      if depth == 0 or len(node.getChildren()) == 0:
          return node.getNodeValue()
      if player:
          bestValue = -float("inf")
          for child in node.getChildren():
              val = self.miniMax(child, depth-1, False)
              bestValue = max(bestValue, val)
          return bestValue
      else:
          bestValue = float("inf")
          for child in node.getChildren():
              val = self.miniMax(child, depth-1, True)
              bestValue = min(bestValue, val)
          return bestValue

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    # create a root node for the tree
    root = self.Node.__init__(self, 0)
    # set the children of the root node to be a layer of possible moves and the heuristic value of each
    root.children = self.createLayers(root, gameState, 0)

    # return the minimax value of the tree
    return self.miniMax(root, self.depth, True)

# recursive method to create the tree with all the values
def createLayers(self, node, gameState, curDepth):

    # get a list of all the legal moves that can be made
    legalMoves = gameState.getLegalActions()

    # for each legal move,
    for action in legalMoves:
        # create a successor state of pacman where that action is applied
        successorGameState = gameState.generatePacmanSuccessor(action)
        # create a new node with a heuristic value of the value returned by the evaluation function
        newChild = self.Node.__init__(self.evaluationFunction(successorGameState, action))
        # add the new node as a child of the current node
        self.addChild(newChild)

    # for each child of the current node
    for child in node.children:
        # if we've gone as far in depth as we are supposed to,
        if curDepth == self.depth:
            # then break
            break
        # otherwise, create a layer of possible moves for the child node
        child.children = self.createLayers(child, gameState, curDepth+1)

    # return the list of children
    # when the recursive calls are done, it will be a reference to the entire tree
    return self.getChildren()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def alphaBeta(self, node, depth, a, b, player):
      if depth == 0 or len(node.getChildren()) == 0:
          return node.getNodeValue()
      if player:
          v = -float("inf")
          for child in node.getChildren():
              v = max(v, self.alphaBeta(child, depth-1, a, b, False))
              a = max(a, v)
              if b <= a:
                  break # beta cut off
          return v
      else:
          v = float("inf")
          for child in node.getChildren():
              v = min(v, self.alphaBeta(child, depth-1, a, b, True))
              b = min(b, v)
              if b <= a:
                  break # alpha cut off
          return v

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    # create a root node for the tree
    root = self.Node.__init__(self, 0)
    # set the children of the root node to be a layer of possible moves and the heuristic value of each
    root.children = self.createLayers(root, gameState, 0)

    a = -float("inf")
    b = float("inf")

    # return the alphaBeta value of the tree
    return self.alphaBeta(root, self.depth, a, b, True)

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
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


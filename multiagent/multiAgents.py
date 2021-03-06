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
import math

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

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def miniMax(self, gameState, depth, score):
      legalActions = gameState.getLegalPacmanActions()
      maxAction =""
      maxScore = -99999999999999999999
      depthScore = {}
      for action in legalActions:
          z = self.evaluationFunction(gameState.generatePacmanSuccessor(action))
          depthScore[action] = self.mini(gameState.generatePacmanSuccessor(action), depth, z)
      for x, y in depthScore.iteritems():
          if(y > maxScore):
              maxScore = y
              maxAction = x
      print maxScore, maxAction
      return maxAction

  def maxi(self, gameState, depth, score):
      if(depth == 0):
          return score
      legalActions = gameState.getLegalPacmanActions()
      depthScore = []
      maxScore = -99999999999999999
      for action in legalActions:
          z = self.evaluationFunction(gameState.generatePacmanSuccessor(action))
          depthScore.append(self.mini(gameState.generatePacmanSuccessor(action), depth-1, z))
      for x in depthScore:
          if x > maxScore:
              maxScore = x
      return maxScore

  def mini(self, gameState, depth, score):
      if(depth == 0):
          return score
      depthScore=[]
      miniScore = -9999999999999999999
      for i in range(0, gameState.getNumAgents()):
        for action in gameState.getLegalActions(i):
          z = self.evaluationFunction(gameState.generateSuccessor(i, action))
          depthScore.append(self.maxi(gameState.generateSuccessor(i, action), depth-1, z))
      for x in depthScore:
          if x > miniScore:
              miniScore = x
      return miniScore

  def minimax2(self, gameState, depth, maximizingPlayer):
      if depth == 0:
          return self.evaluationFunction(gameState)
      if(maximizingPlayer):
          maxValue = -999
          for action in gameState.getLegalPacmanActions():
              score = self.minimax2(gameState.generatePacmanSuccessor(action), depth - 1, False)
              # print score
              maxValue = max(maxValue, score)
          return maxValue
      else:
          minValue = -999
          for i in range(1, gameState.getNumAgents()):
            for action in gameState.getLegalActions(i):
                score = self.minimax2(gameState.generateSuccessor(i, action), depth - 1, True)
                minValue = max(minValue, score)
          return minValue


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
    legalActions = gameState.getLegalPacmanActions()
    scoresForAction = {}
    maxAction = ""
    maxScore = -99999999999999
    for action in legalActions:
        scoresForAction[action] = self.minimax2(gameState.generatePacmanSuccessor(action), self.depth, False)
    for x,y in scoresForAction.iteritems():
        if(maxScore < y):
            maxScore = y
            maxAction = x
    return maxAction
    # return self.miniMax(gameState, self.depth, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def alphaBeta(self, gameState, depth, a, b, maximizingPlayer):
      if depth == 0:
          return betterEvaluationFunction(gameState)
      if maximizingPlayer:
          v = -(float("inf"))
          for i in range(1, gameState.getNumAgents()):
            for action in gameState.getLegalActions(i):
              v = max(v, self.alphaBeta(gameState.generateSuccessor(i, action), depth - 1, a, b, False))
              a = max(a, v)
              if b <= a:
                  break # beta cut-off
          return v
      else:
          v = float("inf")
          legalActions = gameState.getLegalPacmanActions()
          for action in legalActions:
              v = min(v, self.alphaBeta(gameState.generatePacmanSuccessor(action), depth - 1, a, b, True))
              b = min(b, v)
              if b <= a:
                  break # alpha cut-off
          return v

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    legalActions = gameState.getLegalPacmanActions()
    scoresForAction = {}
    maxAction = ""
    maxScore = -99999999999999
    for action in legalActions:
        scoresForAction[action] = self.alphaBeta(gameState.generatePacmanSuccessor(action), self.depth, -(float("inf")), float("inf"), True)
    for x,y in scoresForAction.iteritems():
        if(maxScore < y):
            maxScore = y
            maxAction = x
    return maxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def expectiminimax(self, gameState, depth):
      scareTimes = [ghosts.scaredTimer for ghosts in gameState.getGhostStates()]
      if(depth == 0):
          return betterEvaluationFunction(gameState)
      if(depth%2 !=0):
          a = -float("inf")
          for i in range(1,gameState.getNumAgents()):
              for action in gameState.getLegalActions(i):
                a = max(a, self.expectiminimax(gameState.generateSuccessor(i, action), depth - 1))
      elif(depth%2 == 0 and scareTimes[0] == 0 ):
          a = -(float("inf"))
          for action in gameState.getLegalPacmanActions():
              a = max(a, self.expectiminimax(gameState.generatePacmanSuccessor(action), depth - 1))
      else:
          a = 0
          for action in gameState.getLegalPacmanActions():
              a = a + (betterEvaluationFunction(gameState.generatePacmanSuccessor(action)) + self.expectiminimax(gameState.generatePacmanSuccessor(action), depth -1))
      return a


  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    maxScore = {}
    bestVal = -(float("inf"))
    maxAction = ""
    for action in gameState.getLegalPacmanActions():
          maxScore[action] = self.expectiminimax(gameState.generatePacmanSuccessor(action), self.depth)
    for x,y in maxScore.iteritems():
        if(bestVal< y):
            bestVal = y
            maxAction = x
    return maxAction

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This heuristic prioritizes winning regardless where the ghosts are. It will avoid all actions
    from getting a loss unless if it is trapped.
  """
  "*** YOUR CODE HERE ***"
  ghostScaredTimes =  [ghosts.scaredTimer for ghosts in currentGameState.getGhostStates()]
  if(ghostScaredTimes[0] > 0):
      return ghostScaredTimes + currentGameState.getScore()
  if(currentGameState.isLose()):
      return -(float("inf"))
  elif(currentGameState.isWin()):
      return float("inf")
  else:
    return currentGameState.getScore()

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


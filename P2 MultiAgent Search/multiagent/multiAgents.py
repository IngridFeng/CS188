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

        "*** YOUR CODE HERE ***"
        distOfNearFood, distOfNearGhost = 0, 0
        distKey = lambda x: manhattanDistance(x, newPos)

        ghostPosList = [x.getPosition() for x in newGhostStates if x.scaredTimer == 0]
        ghostPosList = sorted(ghostPosList, key = distKey)
        if ghostPosList:
            distOfNearGhost = distKey(ghostPosList[0])

        foodPosList = sorted(newFood.asList(), key = distKey)
        foodNum = newFood.count()
        if foodPosList:
            distOfNearFood = distKey(foodPosList[0])
        return distOfNearGhost - distOfNearFood - 20 * foodNum

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
        def whichAgent(gameState, depth, agentIndex):
            if gameState.getNumAgents() <= agentIndex:
                depth, agentIndex = depth + 1, 0
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            return agentAction(gameState, depth, agentIndex)

        def agentAction(gameState, depth, agentIndex):
            if agentIndex == 0:
                decision = [float("-inf"), ""]
            else:
                decision = [float("inf"), ""]
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                succDecision = whichAgent(successor, depth, agentIndex + 1)
                if type(succDecision) is list:
                    succValue = succDecision[0]
                else:
                    succValue = succDecision
                if (agentIndex == 0 and succValue > decision[0]) or (agentIndex != 0 and succValue < decision[0]):
                    decision = [succValue, action]
            return decision
        return whichAgent(gameState, 0, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def whichAgent(gameState, depth, agentIndex, alpha, beta):
            if gameState.getNumAgents() <= agentIndex:
                depth, agentIndex = depth + 1, 0
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            return agentAction(gameState, depth, agentIndex, alpha, beta)

        def agentAction(gameState, depth, agentIndex, alpha, beta):
            if agentIndex == 0:
                decision = [float("-inf"), ""]
            else:
                decision = [float("inf"), ""]
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                succDecision = whichAgent(successor, depth, agentIndex + 1, alpha, beta)
                if type(succDecision) is list:
                    succValue = succDecision[0]
                else:
                    succValue = succDecision
                if agentIndex == 0:
                    if succValue > decision[0]:
                        decision = [succValue, action]
                    if succValue > beta:
                        return [succValue, action]
                    alpha = max(alpha, succValue)
                else:
                    if succValue < decision[0]:
                        decision = [succValue, action]
                    if succValue < alpha:
                        return [succValue, action]
                    beta = min(beta, succValue)
            return decision
        return whichAgent(gameState, 0, 0, float("-inf"), float("inf"))[1]

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
        def whichAgent(gameState, depth, agentIndex):
            if gameState.getNumAgents() <= agentIndex:
                depth, agentIndex = depth + 1, 0
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            return agentAction(gameState, depth, agentIndex)

        def agentAction(gameState, depth, agentIndex):
            if agentIndex == 0:
                decision = [float("-inf"), ""]
            else:
                decision = [0, ""]
            actions = gameState.getLegalActions(agentIndex)
            denom = len(actions)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                succDecision = whichAgent(successor, depth, agentIndex + 1)
                if type(succDecision) is list:
                    succValue = succDecision[0]
                else:
                    succValue = succDecision
                if agentIndex == 0:
                    if succValue > decision[0]:
                        decision = [succValue, action]
                else:
                    decision[0], decision[1] = succValue / denom + decision[0], action
            return decision
        return whichAgent(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    distToPac = lambda x: manhattanDistance(x, currentGameState.getPacmanPosition())
    distOfNearGhost, ghostFeature = float("inf"), 0
    for ghost in currentGameState.getGhostStates():
        ghostDist = distToPac(ghost.getPosition())
        if ghost.scaredTimer == 0:
            distOfNearGhost = min(ghostDist, distOfNearGhost)
        elif ghost.scaredTimer > ghostDist:
            ghostFeature += 200 - ghostDist
    if distOfNearGhost == float("inf"):
        distOfNearGhost = 0
    ghostFeature += distOfNearGhost

    distOfNearFood = 0
    foodPosList = sorted(currentGameState.getFood().asList(), key = distToPac)
    if foodPosList:
        distOfNearFood = distToPac(foodPosList[0])
    numFood = currentGameState.getNumFood()
    score = currentGameState.getScore()
    numCaps = len(currentGameState.getCapsules())

    return score + ghostFeature + 2 * numCaps - 2 * distOfNearFood - 20 * numFood
# Abbreviation
better = betterEvaluationFunction

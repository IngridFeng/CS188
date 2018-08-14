# myAgentP3.py
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
# This file was based on the starter code for student bots, and refined
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#########
# Agent #
#########
class myAgentP3(CaptureAgent):
  """
  Students' Names: Yige Feng
  Phase Number: 3
  Description of Bot: I let my bot receive broadcast from his teammate and I filtered out food
  that is going to be eaten by his teammate. I tried to eliminate all food in a distance of 30
  to my bot but it decreases my score overall. Some games even time out. Then I get 3 future actions
  for my bot. I assume my bot has performed the current action and generate successor game state
  and then loop to find the next optimal action based on features and weights. Lastly, I deleted
  the 'stopPenalty' since that case is already handled and also the 'successorScore' feature since
  it is not that important. Instead, I added numRepeats feature. Now my bot is using 3 features: 
  teammateDistance, numRepeats, and closestFood.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    # Make sure you do not delete the following line.
    # If you would like to use Manhattan distances instead
    # of maze distances in order to save on initialization
    # time, please take a look at:
    # CaptureAgent.registerInitialState in captureAgents.py.
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    teammateActions = self.receivedBroadcast
    # Process your teammate's broadcast!
    # Use it to pick a better action for yourself
    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1
    teammateIndex = teammateIndices[0]
    global otherAgentPositions
    otherAgentPositions = getFuturePositions(gameState, teammateActions, teammateIndex)

    futureActions = []
    for count in range(4):
        actions = gameState.getLegalActions(self.index)
        filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), gameState, self.index)
        values = [self.evaluate(gameState, a) for a in filteredActions]
        currentAction = [a for a in actions if self.evaluate(gameState, a) == max(values)][0]# Change this!
        successorGameState = gameState.generateSuccessor(self.index, currentAction)
        gameState = successorGameState
        if count == 0:
            chosenAction = currentAction
        else:
            futureActions.append(currentAction)
    if not futureActions:
        self.toBroadcast = None
    else:
        self.toBroadcast = futureActions
    return chosenAction

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()

    ### Useful information you can extract from a GameState (pacman.py) ###
    successorGameState = gameState.generateSuccessor(self.index, action)
    newPos = successorGameState.getAgentPosition(self.index)
    oldFood = gameState.getFood()
    newFood = successorGameState.getFood()
    ghostIndices = self.getOpponents(successorGameState)

    # Determines how many times the agent has already been in the newPosition in the last 20 moves
    numRepeats = sum([1 for x in self.observationHistory[-20:] if x.getAgentPosition(self.index) == newPos])

    ghostPositions = [successorGameState.getAgentPosition(ghostIndex) for ghostIndex in ghostIndices]
    ghostDistances = [self.getMazeDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
    ghostDistances.append( 1000 )
    closestGhost = min( ghostDistances ) + 1.0

    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    teammatePos = successorGameState.getAgentPosition(teammateIndex)
    teammateDistance = self.getMazeDistance(newPos, teammatePos) + 1.0

    pacmanDeath = successorGameState.data.num_deaths

    features['successorScore'] = self.getScore(successorGameState)

    # CHANGE YOUR FEATURES HERE
    foodPositions = oldFood.asList()
    if otherAgentPositions:
        foodPositions = [foodPosition for foodPosition in foodPositions if foodPosition not in otherAgentPositions]
    foodDistances = [self.getMazeDistance(newPos, foodPosition) for foodPosition in foodPositions]
    closestFood = 0
    if foodDistances:
        closestFood = min( foodDistances ) + 1.0
    features['closestFood'] = - closestFood
    features['teammateDistance'] = teammateDistance
    features['numRepeats'] = numRepeats
    return features

  def getWeights(self, gameState, action):
    # CHANGE YOUR WEIGHTS HERE
    return {'closestFood': 300, 'teammateDistance': 100, 'numRepeats': 100}

def getFuturePositions(gameState, plannedActions, agentIndex):
  """
  Returns list of future positions given by a list of actions for a
  specific agent starting form gameState

  NOTE: this does not take into account other agent's movements
  (such as ghosts) that might impact the *actual* positions visited
  by such agent
  """
  if plannedActions is None:
    return None

  planPositions = [gameState.getAgentPosition(agentIndex)]
  for action in plannedActions:
    if action in gameState.getLegalActions(agentIndex):
      gameState = gameState.generateSuccessor(agentIndex, action)
      planPositions.append(gameState.getAgentPosition(agentIndex))
    else:
      break
  return planPositions

def actionsWithoutStop(legalActions):
  """
  Filters actions by removing the STOP action
  """
  legalActions = list(legalActions)
  if Directions.STOP in legalActions:
    legalActions.remove(Directions.STOP)
  return legalActions

def actionsWithoutReverse(legalActions, gameState, agentIndex):
  """
  Filters actions by removing REVERSE, i.e. the opposite action to the previous one
  """
  legalActions = list(legalActions)
  reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
  if len (legalActions) > 1 and reverse in legalActions:
    legalActions.remove(reverse)
  return legalActions

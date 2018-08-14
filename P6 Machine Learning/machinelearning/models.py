import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.b1, self.b2 = nn.Variable(250), nn.Variable(1)
        self.w1, self.w2 = nn.Variable(1 ,250), nn.Variable(250,1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        g = nn.Graph([self.w1, self.b1, self.w2, self.b2])
        x1 = nn.MatrixMultiply(g, nn.Input(g, x), self.w1)
        x1_add_b1 = nn.MatrixVectorAdd(g, x1, self.b1)
        relu = nn.ReLU(g, x1_add_b1)
        x2 = nn.MatrixMultiply(g, relu, self.w2)
        x2_add_b2 = nn.MatrixVectorAdd(g, x2, self.b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            nn.SquareLoss(g, x2_add_b2, nn.Input(g, y))
            return g
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return g.get_output(x2_add_b2)

class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.b1, self.b2 = nn.Variable(10), nn.Variable(1)
        self.w1, self.w2 = nn.Variable(1, 10), nn.Variable(10, 1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        ide = nn.Variable(1)
        ide.data = -np.identity(1)
        g = nn.Graph([self.w1, self.b1, self.w2, self.b2, ide])
        x1 = nn.MatrixMultiply(g, nn.Input(g, x), self.w1)
        x1_add_b1 = nn.MatrixVectorAdd(g, x1, self.b1)
        relu = nn.ReLU(g, x1_add_b1)
        x2 = nn.MatrixMultiply(g, relu, self.w2)
        x2_add_b2 = nn.MatrixVectorAdd(g, x2, self.b2)

        n_x1 = nn.MatrixMultiply(g, nn.Input(g, -x), self.w1)
        n_x1_add_b1 = nn.MatrixVectorAdd(g, n_x1, self.b1)
        n_relu = nn.ReLU(g, n_x1_add_b1)
        n_x2 = nn.MatrixMultiply(g, n_relu, self.w2)
        n_x2_add_b2 = nn.MatrixVectorAdd(g, n_x2, self.b2)
        n_x2_add_b2 = nn.MatrixMultiply(g, n_x2_add_b2, ide)
        f = nn.Add(g, x2_add_b2, n_x2_add_b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            nn.SquareLoss(g, f, nn.Input(g, y))
            return g
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return g.get_output(f)

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.7
        self.b1, self.b2 = nn.Variable(250), nn.Variable(10)
        self.w1, self.w2 = nn.Variable(784 ,250), nn.Variable(250, 10)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        g = nn.Graph([self.w1, self.b1, self.w2, self.b2])
        x1 = nn.MatrixMultiply(g, nn.Input(g, x), self.w1)
        x1_add_b1 = nn.MatrixVectorAdd(g, x1, self.b1)
        relu = nn.ReLU(g, x1_add_b1)
        x2 = nn.MatrixMultiply(g, relu, self.w2)
        x2_add_b2 = nn.MatrixVectorAdd(g, x2, self.b2)

        if y is not None:
            "*** YOUR CODE HERE ***"
            nn.SoftmaxLoss(g, x2_add_b2, nn.Input(g, y))
            return g
        else:
            "*** YOUR CODE HERE ***"
            return g.get_output(x2_add_b2)


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.b1, self.b2 = nn.Variable(250), nn.Variable(2)
        self.w1, self.w2 = nn.Variable(4 ,250), nn.Variable(250, 2)

    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        g = nn.Graph([self.w1, self.b1, self.w2, self.b2])
        x1 = nn.MatrixMultiply(g, nn.Input(g, states), self.w1)
        x1_add_b1 = nn.MatrixVectorAdd(g, x1, self.b1)
        relu = nn.ReLU(g, x1_add_b1)
        x2 = nn.MatrixMultiply(g, relu, self.w2)
        x2_add_b2 = nn.MatrixVectorAdd(g, x2, self.b2)


        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            nn.SquareLoss(g, x2_add_b2, nn.Input(g, Q_target))
            return g
        else:
            "*** YOUR CODE HERE ***"
            return g.get_output(x2_add_b2)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.dimensionality = 100
        self.b = nn.Variable(self.dimensionality)
        self.w1 = nn.Variable(self.dimensionality, self.dimensionality)
        self.w2 = nn.Variable(47, self.dimensionality)
        self.w3 = nn.Variable(self.dimensionality, 5)

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"
        h = nn.Variable(batch_size, self.dimensionality)
        h.data = np.zeros((batch_size, self.dimensionality))
        g = nn.Graph([h, self.w1, self.w2, self.w3, self.b])
        for x in xs:
            h1 = nn.MatrixMultiply(g, h, self.w1)
            x2 = nn.MatrixMultiply(g, nn.Input(g, x), self.w2)
            h1_add_x2 = nn.Add(g, h1, x2)
            add_b = nn.MatrixVectorAdd(g, h1_add_x2, self.b)
            relu = nn.ReLU(g, add_b)
            h = relu
        result = nn.MatrixMultiply(g, h, self.w3)

        if y is not None:
            "*** YOUR CODE HERE ***"
            nn.SoftmaxLoss(g, result, nn.Input(g, y))
            return g
        else:
            "*** YOUR CODE HERE ***"
            return g.get_output(result)

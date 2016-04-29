# hello.py

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox: ``mdp`` module
=====================================================

The ``mdp`` module provides classes for the resolution of descrete-time Markov
Decision Processes.

Available classes
-----------------
:class:`~mdptoolbox.mdp.MDP`
    Base Markov decision process class
:class:`~mdptoolbox.mdp.FiniteHorizon`
    Backwards induction finite horizon MDP
:class:`~mdptoolbox.mdp.PolicyIteration`
    Policy iteration MDP
:class:`~mdptoolbox.mdp.PolicyIterationModified`
    Modified policy iteration MDP
:class:`~mdptoolbox.mdp.QLearning`
    Q-learning MDP
:class:`~mdptoolbox.mdp.RelativeValueIteration`
    Relative value iteration MDP
:class:`~mdptoolbox.mdp.ValueIteration`
    Value iteration MDP
:class:`~mdptoolbox.mdp.ValueIterationGS`
    Gauss-Seidel value iteration MDP

"""

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp

import mdptoolbox.example as example

# import mdptoolbox.util as _util

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

_MSG_STOP_MAX_ITER = "Iterating stopped due to maximum number of iterations " \
    "condition."
_MSG_STOP_EPSILON_OPTIMAL_POLICY = "Iterating stopped, epsilon-optimal " \
    "policy found."
_MSG_STOP_EPSILON_OPTIMAL_VALUE = "Iterating stopped, epsilon-optimal value " \
    "function found."
_MSG_STOP_UNCHANGING_POLICY = "Iterating stopped, unchanging policy found."


def _computeDimensions(transition):
    A = len(transition)
    try:
        if transition.ndim == 3:
            S = transition.shape[1]
        else:
            S = transition[0].shape[0]
    except AttributeError:
        S = transition[0].shape[0]
    return S, A


def _printVerbosity(iteration, variation):
    if isinstance(variation, float):
        print("{:>10}{:>12f}".format(iteration, variation))
    elif isinstance(variation, int):
        print("{:>10}{:>12d}".format(iteration, variation))
    else:
        print("{:>10}{:>12}".format(iteration, variation))

class final_mdp(object):
    
    """A Markov Decision Problem.

    Let ``S`` = the number of states, and ``A`` = the number of acions.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. These can be defined in a variety of
        ways. The simplest is a numpy array that has the shape ``(A, S, S)``,
        though there are other possibilities. It can be a tuple or list or
        numpy object array of length ``A``, where each element contains a numpy
        array or matrix that has the shape ``(S, S)``. This "list of matrices"
        form is useful when the transition matrices are sparse as
        ``scipy.sparse.csr_matrix`` matrices can be used. In summary, each
        action's transition matrix must be indexable like ``transitions[a]``
        where ``a`` ∈ {0, 1...A-1}, and ``transitions[a]`` returns an ``S`` ×
        ``S`` array-like object.
    reward : array
        Reward matrices or vectors. Like the transition matrices, these can
        also be defined in a variety of ways. Again the simplest is a numpy
        array that has the shape ``(S, A)``, ``(S,)`` or ``(A, S, S)``. A list
        of lists can be used, where each inner list has length ``S`` and the
        outer list has length ``A``. A list of numpy arrays is possible where
        each inner array can be of the shape ``(S,)``, ``(S, 1)``, ``(1, S)``
        or ``(S, S)``. Also ``scipy.sparse.csr_matrix`` can be used instead of
        numpy arrays. In addition, the outer list can be replaced by any object
        that can be indexed like ``reward[a]`` such as a tuple or numpy object
        array of length ``A``.
    discount : float
        Discount factor. The per time-step discount factor on future rewards.
        Valid values are greater than 0 upto and including 1. If the discount
        factor is 1, then convergence is cannot be assumed and a warning will
        be displayed. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a discount factor.
    epsilon : float
        Stopping criterion. The maximum change in the value function at each
        iteration is compared against ``epsilon``. Once the change falls below
        this value, then the value function is considered to have converged to
        the optimal value function. Subclasses of ``MDP`` may pass ``None`` in
        the case where the algorithm does not use an epsilon-optimal stopping
        criterion.
    max_iter : int
        Maximum number of iterations. The algorithm will be terminated once
        this many iterations have elapsed. This must be greater than 0 if
        specified. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a maximum number of iterations.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Attributes
    ----------
    P : array
        Transition probability matrices.
    R : array
        Reward vectors.
    V : tuple
        The optimal value function. Each element is a float corresponding to
        the expected value of being in that state assuming the optimal policy
        is followed.
    discount : float
        The discount rate on future rewards.
    max_iter : int
        The maximum number of iterations.
    policy : tuple
        The optimal policy.
    time : float
        The time used to converge to the optimal policy.
    verbose : boolean
        Whether verbose output should be displayed or not.

    Methods
    -------
    run
        Implemented in child classes as the main algorithm loop. Raises an
        exception if it has not been overridden.
    setSilent
        Turn the verbosity off
    setVerbose
        Turn the verbosity on

    """

    def __init__(self, transitions, reward, discount, epsilon, max_iter,
                 skip_check=False):
        # Initialise a MDP based on the input parameters.

        # if the discount is None then the algorithm is assumed to not use it
        # in its computations
        if discount is not None:
            self.discount = float(discount)
            assert 0.0 < self.discount <= 1.0, (
                "Discount rate must be in ]0; 1]"
            )
            if self.discount == 1:
                print("WARNING: check conditions of convergence. With no "
                      "discount, convergence can not be assumed.")

        # if the max_iter is None then the algorithm is assumed to not use it
        # in its computations
        if max_iter is not None:
            self.max_iter = int(max_iter)
            assert self.max_iter > 0, (
                "The maximum number of iterations must be greater than 0."
            )

        # check that epsilon is something sane
        if epsilon is not None:
            self.epsilon = float(epsilon)
            assert self.epsilon > 0, "Epsilon must be greater than 0."

        if not skip_check:
            # We run a check on P and R to make sure they are describing an
            # MDP. If an exception isn't raised then they are assumed to be
            # correct.
            check(transitions, reward)

        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)
        self.R = self._computeReward(reward, transitions)
        

        # the verbosity is by default turned off
        self.verbose = True
        # Initially the time taken to perform the computations is set to None
        self.time = None
        # set the initial iteration count to zero
        self.iter = 0
        # V should be stored as a vector ie shape of (S,) or (1, S)
        self.V = None
        # policy can also be stored as a vector
        self.policy = None

    def __repr__(self):
        P_repr = "P: \n"
        R_repr = "R: \n"
        for aa in range(self.A):
            P_repr += repr(self.P[aa]) + "\n"
            R_repr += repr(self.R[aa]) + "\n"
        return(P_repr + "\n" + R_repr)

    def _bellmanOperator_mpi(self, a, b, ln, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        # print (a, b, ln)
        print (rank)
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((ln,), (1, ln)), "V is not the " \
                    "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = _np.empty((ln, self.A))
        for ss in xrange(ln):
            Q[ss] = _np.array(self.R)[:,ss] + 0.96 * self.myfunc_mpi(ss, ln)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        # print ('Size of Q is: ' + str(Q.shape))
        return Q
        #return (Q.argmax(axis=1), Q.max(axis=1))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)

    def myfunc_mpi(self, ss, ln):
        arr = []
        new_p = _np.transpose(_np.array(self.P)[:,[ss][:]].reshape(self.A, ln))
        for i in xrange(self.A):
            arr.append(new_p[:,i].dot(self.V))
        arr = _np.array(arr).reshape(1, len(arr))
        return arr

    def _bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.q,), (1, self.S)), "V is not the " \
                    "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = _np.empty((self.S, self.A))
        # for aa in range(self.A):
        #     Q[aa] = self.R[aa] + self.discount * self.P[aa].dot(V)
        for ss in xrange(self.S):
            Q[ss] = _np.array(self.R)[:,ss] + 0.96 * self.myfunc(ss)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        # print ('Size of Q is: ' + str(Q.shape))
        return (Q.argmax(axis=1), Q.max(axis=1))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)

    def myfunc(self, ss):
        arr = []
        new_p = _np.transpose(_np.array(self.P)[:,[ss][:]].reshape(self.A, self.S))
        for i in xrange(self.A):
            arr.append(new_p[:,i].dot(self.V))
        arr = _np.array(arr).reshape(1, len(arr))
        return arr

    def _computeTransition(self, transition):
        return tuple(transition[a] for a in range(self.A))

    def _computeReward(self, reward, transition):
        # Compute the reward for the system in one state chosing an action.
        # Arguments
        # Let S = number of states, A = number of actions
        # P could be an array with 3 dimensions or  a cell array (1xA),
        # each cell containing a matrix (SxS) possibly sparse
        # R could be an array with 3 dimensions (SxSxA) or  a cell array
        # (1xA), each cell containing a sparse matrix (SxS) or a 2D
        # array(SxA) possibly sparse
        try:
            if reward.ndim == 1:
                return self._computeVectorReward(reward)
            elif reward.ndim == 2:
                return self._computeArrayReward(reward)
            else:
                r = tuple(map(self._computeMatrixReward, reward, transition))
                return r
        except (AttributeError, ValueError):
            if len(reward) == self.A:
                r = tuple(map(self._computeMatrixReward, reward, transition))
                return r
            else:
                return self._computeVectorReward(reward)

    def _computeVectorReward(self, reward):
        if _sp.issparse(reward):
            raise NotImplementedError
        else:
            r = _np.array(reward).reshape(self.S)
            return tuple(r for a in range(self.A))

    def _computeArrayReward(self, reward):
        if _sp.issparse(reward):
            raise NotImplementedError
        else:
            def func(x):
                return _np.array(x).reshape(self.S)

            return tuple(func(reward[:, a]) for a in range(self.A))

    def _computeMatrixReward(self, reward, transition):
        if _sp.issparse(reward):
            # An approach like this might be more memory efficeint
            # reward.data = reward.data * transition[reward.nonzero()]
            # return reward.sum(1).A.reshape(self.S)
            # but doesn't work as it is.
            return reward.multiply(transition).sum(1).A.reshape(self.S)
        elif _sp.issparse(transition):
            return transition.multiply(reward).sum(1).A.reshape(self.S)
        else:
            return _np.multiply(transition, reward).sum(1).reshape(self.S)

    def _startRun(self):
        if self.verbose:
            _printVerbosity('Iteration', 'Variation')

        self.time = _time.time()

    def _endRun(self):
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())

        try:
            self.policy = tuple(self.policy.tolist())
        except AttributeError:
            self.policy = tuple(self.policy)

        self.time = _time.time() - self.time

    def run(self):
        """Raises error because child classes should implement this function.
        """
        raise NotImplementedError("You should create a run() method.")

    def setSilent(self):
        """Set the MDP algorithm to silent mode."""
        self.verbose = False

    def setVerbose(self):
        """Set the MDP algorithm to verbose mode."""
        self.verbose = True

class ValueIteration(final_mdp):

    """A discounted MDP solved using the value iteration algorithm.

    Description
    -----------
    ValueIteration applies the value iteration algorithm to solve a
    discounted MDP. The algorithm consists of solving Bellman's equation
    iteratively.
    Iteration is stopped when an epsilon-optimal policy is found or after a
    specified number (``max_iter``) of iterations.
    This function uses verbose and silent modes. In verbose mode, the function
    displays the variation of ``V`` (the value function) for each iteration and
    the condition which stopped the iteration: epsilon-policy found or maximum
    number of iterations reached.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details.  Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. If the value given is greater than a
        computed bound, a warning informs that the computed bound will be used
        instead. By default, if ``discount`` is not equal to 1, a bound for
        ``max_iter`` is computed, otherwise ``max_iter`` = 1000. See the
        documentation for the ``MDP`` class for further details.
    initial_value : array, optional
        The starting value function. Default: a vector of zeros.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Data Attributes
    ---------------
    V : tuple
        The optimal value function.
    policy : tuple
        The optimal policy function. Each element is an integer corresponding
        to an action which maximises the value function in that state.
    iter : int
        The number of iterations taken to complete the computation.
    time : float
        The amount of CPU time used to run the algorithm.

    Methods
    -------
    run()
        Do the algorithm iteration.
    setSilent()
        Sets the instance to silent mode.
    setVerbose()
        Sets the instance to verbose mode.

    Notes
    -----
    In verbose mode, at each iteration, displays the variation of V
    and the condition which stopped iterations: epsilon-optimum policy found
    or maximum number of iterations reached.

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
    >>> vi.verbose
    False
    >>> vi.run()
    >>> expected = (5.93215488, 9.38815488, 13.38815488)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (0, 0, 0)
    >>> vi.iter
    4

    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)
    >>> vi.iter
    26

    >>> import mdptoolbox
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix as sparse
    >>> P = [None] * 2
    >>> P[0] = sparse([[0.5, 0.5],[0.8, 0.2]])
    >>> P[1] = sparse([[0, 1],[0.1, 0.9]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)

    """
    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=1000, initial_value=0, skip_check=False):
        # Initialise a value iteration MDP.

        final_mdp.__init__(self, transitions, reward, discount, epsilon, max_iter,
                     skip_check=skip_check)

        # initialization of optional arguments
        if initial_value == 0:
            self.V = _np.zeros(self.S)
        else:
            assert len(initial_value) == self.S, "The initial value must be " \
                "a vector of length S."
            self.V = _np.array(initial_value).reshape(self.S)
        if self.discount < 1:
            # compute a bound for the number of iterations and update the
            # stored value of self.max_iter
            self._boundIter(epsilon)
            # computation of threshold of variation for V for an epsilon-
            # optimal policy
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else:  # discount == 1
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon

    def _boundIter(self, epsilon):
        # Compute a bound for the number of iterations.
        #
        # for the value iteration
        # algorithm to find an epsilon-optimal policy with use of span for the
        # stopping criterion
        #
        # Arguments -----------------------------------------------------------
        # Let S = number of states, A = number of actions
        #    epsilon   = |V - V*| < epsilon,  upper than 0,
        #        optional (default : 0.01)
        # Evaluation ----------------------------------------------------------
        #    max_iter  = bound of the number of iterations for the value
        #    iteration algorithm to find an epsilon-optimal policy with use of
        #    span for the stopping criterion
        #    cpu_time  = used CPU time
        #
        # See Markov Decision Processes, M. L. Puterman,
        # Wiley-Interscience Publication, 1994
        # p 202, Theorem 6.6.6
        # k =    max     [1 - S min[ P(j|s,a), p(j|s',a')] ]
        #     s,a,s',a'       j
        k = 0
        h = _np.zeros(self.S)

        for ss in range(self.S):
            PP = _np.zeros((self.A, self.S))
            for aa in range(self.A):
                try:
                    PP[aa] = self.P[aa][:, ss]
                except ValueError:
                    PP[aa] = self.P[aa][:, ss].todense().A1
            # minimum of the entire array.
            h[ss] = PP.min()

        k = 1 - h.sum()
        Vprev = self.V
        null, value = self._bellmanOperator()
        # p 201, Proposition 6.6.5
        span = getSpan(value - Vprev)
        max_iter = (_math.log((epsilon * (1 - self.discount) / self.discount) /
                    span) / _math.log(self.discount * k))
        # self.V = Vprev

        self.max_iter = int(_math.ceil(max_iter))
    def get_range(self,rank, s):
        local_n = s / size
        local_a = rank * local_n
        local_b = local_a + local_n
        return (local_a, local_b)

    def run(self):
        # Run the value iteration algorithm.
        self._startRun()
        local_n = self.S / size
        takeQ = _np.empty((local_n,self.A))
        local_V = _np.zeros(self.S)
        local_P = _np.zeros(self.S)
        global_V =_np.zeros(self.S)
        global_P =_np.zeros(self.S)
        newrank = -1

        while True:
            self.iter += 1
            local_a,local_b = self.get_range(rank,self.S)
            Vprev = self.V.copy()

            # Bellman Operator: compute policy and value functions
            takeQ = self._bellmanOperator_mpi(local_a, local_b, local_n)
            if rank == 0:
                local_P = takeQ.argmax(axis=1)
                local_V = takeQ.max(axis=1)
                global_V[local_a:local_b] = local_V
                global_P[local_a:local_b] = local_P
                for i in xrange(1, size):
                    status = MPI.Status()
                    comm.Recv(takeQ, source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
                    newrank = status.Get_source()
                    la,lb = self.get_range(newrank,self.S)
                    local_P = takeQ.argmax(axis=1)
                    local_V = takeQ.max(axis=1)
                    global_V[la : lb] = local_V
                    global_P[la : lb] = local_P
                # print (repr(local_V) + "\n" + repr(local_P))
                self.V = global_V
                self.policy = global_P
            else:
                comm.send(takeQ, dest = 0, tag=rank)
            comm.Bcast(self.V,root=0)
            comm.Bcast(self.policy, root=0)

            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            variation = getSpan(self.V - Vprev)

            if self.verbose:
                _printVerbosity(self.iter, variation)

            if variation < self.thresh:
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

        self._endRun()

"""Markov Decision Process (MDP) Toolbox: ``util`` module
======================================================

The ``util`` module provides functions to check that an MDP is validly
described. There are also functions for working with MDPs while they are being
solved.

Available functions
-------------------

:func:`~mdptoolbox.util.check`
    Check that an MDP is properly defined
:func:`~mdptoolbox.util.checkSquareStochastic`
    Check that a matrix is square and stochastic
:func:`~mdptoolbox.util.getSpan`
    Calculate the span of an array
:func:`~mdptoolbox.util.isNonNegative`
    Check if a matrix has only non-negative elements
:func:`~mdptoolbox.util.isSquare`
    Check if a matrix is square
:func:`~mdptoolbox.util.isStochastic`
    Check if a matrix is row stochastic

"""

# import numpy as _np

# import mdptoolbox.error as _error

_MDPERR = {
"mat_nonneg" :
    "Transition probabilities must be non-negative.",
"mat_square" :
    "A transition probability matrix must be square, with dimensions S×S.",
"mat_stoch" :
    "Each row of a transition probability matrix must sum to one (1).",
"obj_shape" :
    "Object arrays for transition probabilities and rewards "
    "must have only 1 dimension: the number of actions A. Each element of "
    "the object array contains an SxS ndarray or matrix.",
"obj_square" :
    "Each element of an object array for transition "
    "probabilities and rewards must contain an SxS ndarray or matrix; i.e. "
    "P[a].shape = (S, S) or R[a].shape = (S, S).",
"P_type" :
    "The transition probabilities must be in a numpy array; "
    "i.e. type(P) is ndarray.",
"P_shape" :
    "The transition probability array must have the shape "
    "(A, S, S)  with S : number of states greater than 0 and A : number of "
    "actions greater than 0. i.e. R.shape = (A, S, S)",
"PR_incompat" :
    "Incompatibility between P and R dimensions.",
"R_type" :
    "The rewards must be in a numpy array; i.e. type(R) is "
    "ndarray, or numpy matrix; i.e. type(R) is matrix.",
"R_shape" :
    "The reward matrix R must be an array of shape (A, S, S) or "
    "(S, A) with S : number of states greater than 0 and A : number of "
    "actions greater than 0. i.e. R.shape = (S, A) or (A, S, S)."
}


def _checkDimensionsListLike(arrays):
    """Check that each array in a list of arrays has the same size.

    """
    dim1 = len(arrays)
    dim2, dim3 = arrays[0].shape
    for aa in range(1, dim1):
        dim2_aa, dim3_aa = arrays[aa].shape
        if (dim2_aa != dim2) or (dim3_aa != dim3):
            raise InvalidError(_MDPERR["obj_square"])
    return dim1, dim2, dim3


def _checkRewardsListLike(reward, n_actions, n_states):
    """Check that a list-like reward input is valid.

    """
    try:
        lenR = len(reward)
        if lenR == n_actions:
            dim1, dim2, dim3 = _checkDimensionsListLike(reward)
        elif lenR == n_states:
            dim1 = n_actions
            dim2 = dim3 = lenR
        else:
            raise InvalidError(_MDPERR["R_shape"])
    except AttributeError:
        raise InvalidError(_MDPERR["R_shape"])
    return dim1, dim2, dim3


def isSquare(matrix):
    """Check that ``matrix`` is square.

    Returns
    =======
    is_square : bool
        ``True`` if ``matrix`` is square, ``False`` otherwise.

    """
    try:
        try:
            dim1, dim2 = matrix.shape
        except AttributeError:
            dim1, dim2 = _np.array(matrix).shape
    except ValueError:
        return False
    if dim1 == dim2:
        return True
    return False


def isStochastic(matrix):
    """Check that ``matrix`` is row stochastic.

    Returns
    =======
    is_stochastic : bool
        ``True`` if ``matrix`` is row stochastic, ``False`` otherwise.

    """
    try:
        absdiff = (_np.abs(matrix.sum(axis=1) - _np.ones(matrix.shape[0])))
    except AttributeError:
        matrix = _np.array(matrix)
        absdiff = (_np.abs(matrix.sum(axis=1) - _np.ones(matrix.shape[0])))
    return (absdiff.max() <= 10*_np.spacing(_np.float64(1)))


def isNonNegative(matrix):
    """Check that ``matrix`` is row non-negative.

    Returns
    =======
    is_stochastic : bool
        ``True`` if ``matrix`` is non-negative, ``False`` otherwise.

    """
    try:
        if (matrix >= 0).all():
            return True
    except (NotImplementedError, AttributeError, TypeError):
        try:
            if (matrix.data >= 0).all():
                return True
        except AttributeError:
            matrix = _np.array(matrix)
            if (matrix.data >= 0).all():
                return True
    return False


def checkSquareStochastic(matrix):
    """Check if ``matrix`` is a square and row-stochastic.

    To pass the check the following conditions must be met:

    * The matrix should be square, so the number of columns equals the
      number of rows.
    * The matrix should be row-stochastic so the rows should sum to one.
    * Each value in the matrix must be positive.

    If the check does not pass then a mdptoolbox.util.Invalid

    Arguments
    ---------
    ``matrix`` : numpy.ndarray, scipy.sparse.*_matrix
        A two dimensional array (matrix).

    Notes
    -----
    Returns None if no error has been detected, else it raises an error.

    """
    if not isSquare(matrix):
        raise SquareError
    if not isStochastic(matrix):
        raise StochasticError
    if not isNonNegative(matrix):
        raise NonNegativeError


def check(P, R):
    """Check if ``P`` and ``R`` define a valid Markov Decision Process (MDP).

    Let ``S`` = number of states, ``A`` = number of actions.

    Arguments
    ---------
    P : array
        The transition matrices. It can be a three dimensional array with
        a shape of (A, S, S). It can also be a one dimensional arraye with
        a shape of (A, ), where each element contains a matrix of shape (S, S)
        which can possibly be sparse.
    R : array
        The reward matrix. It can be a three dimensional array with a
        shape of (S, A, A). It can also be a one dimensional array with a
        shape of (A, ), where each element contains matrix with a shape of
        (S, S) which can possibly be sparse. It can also be an array with
        a shape of (S, A) which can possibly be sparse.

    Notes
    -----
    Raises an error if ``P`` and ``R`` do not define a MDP.

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P_valid, R_valid = mdptoolbox.example.rand(100, 5)
    >>> mdptoolbox.util.check(P_valid, R_valid) # Nothing should happen
    >>>
    >>> import numpy as np
    >>> P_invalid = np.random.rand(5, 100, 100)
    >>> mdptoolbox.util.check(P_invalid, R_valid) # Raises an exception
    Traceback (most recent call last):
    ...
    StochasticError:...

    """
    # Checking P
    try:
        if P.ndim == 3:
            aP, sP0, sP1 = P.shape
        elif P.ndim == 1:
            aP, sP0, sP1 = _checkDimensionsListLike(P)
        else:
            raise InvalidError(_MDPERR["P_shape"])
    except AttributeError:
        try:
            aP, sP0, sP1 = _checkDimensionsListLike(P)
        except AttributeError:
            raise InvalidError(_MDPERR["P_shape"])
    msg = ""
    if aP <= 0:
        msg = "The number of actions in P must be greater than 0."
    elif sP0 <= 0:
        msg = "The number of states in P must be greater than 0."
    if msg:
        raise InvalidError(msg)
    # Checking R
    try:
        ndimR = R.ndim
        if ndimR == 1:
            aR, sR0, sR1 = _checkRewardsListLike(R, aP, sP0)
        elif ndimR == 2:
            sR0, aR = R.shape
            sR1 = sR0
        elif ndimR == 3:
            aR, sR0, sR1 = R.shape
        else:
            raise InvalidError(_MDPERR["R_shape"])
    except AttributeError:
        aR, sR0, sR1 = _checkRewardsListLike(R, aP, sP0)
    msg = ""
    if sR0 <= 0:
        msg = "The number of states in R must be greater than 0."
    elif aR <= 0:
        msg = "The number of actions in R must be greater than 0."
    elif sR0 != sR1:
        msg = "The matrix R must be square with respect to states."
    elif sP0 != sR0:
        msg = "The number of states must agree in P and R."
    elif aP != aR:
        msg = "The number of actions must agree in P and R."
    if msg:
        raise InvalidError(msg)
    # Check that the P's are square, stochastic and non-negative
    for aa in range(aP):
        checkSquareStochastic(P[aa])


def getSpan(array):
    """Return the span of `array`

    span(array) = max array(s) - min array(s)

    """
    return array.max() - array.min()

class Error(Exception):
    """Base class for exceptions in this module."""

    def __init__(self):
        Exception.__init__(self)
        self.message = "PyMDPToolbox - "

    def __str__(self):
        return repr(self.message)

class InvalidError(Error):
    """Class for invalid definitions of a MDP."""

    def __init__(self, msg):
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)

class NonNegativeError(Error):
    """Class for transition matrix stochastic errors"""

    default_msg = "The transition probability matrix is negative."

    def __init__(self, msg=None):
        if msg is None:
            msg = self.default_msg
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)

class SquareError(Error):
    """Class for transition matrix square errors"""

    default_msg = "The transition probability matrix is not square."

    def __init__(self, msg=None):
        if msg is None:
            msg = self.default_msg
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)

class StochasticError(Error):
    """Class for transition matrix stochastic errors"""

    default_msg = "The transition probability matrix is not stochastic."

    def __init__(self, msg=None):
        if msg is None:
            msg = self.default_msg
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)
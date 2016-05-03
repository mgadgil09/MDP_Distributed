# -*- coding: utf-8 -*-

import math as _math
import time as _time
import numpy as np
import scipy.sparse as _sp
import mdptoolbox.example as example
from mpi4py import MPI
import mdptoolbox
from mdptoolbox.final_mdp import final_mdp

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

def _boundIter(mdp_obj):

    k = 0
    h = np.zeros(mdp_obj.S)

    for ss in range(mdp_obj.S):
        PP = np.zeros((mdp_obj.A, mdp_obj.S))
        for aa in range(mdp_obj.A):
            try:
                PP[aa] = mdp_obj.P[aa][:, ss]
            except ValueError:
                PP[aa] = mdp_obj.P[aa][:, ss].todense().A1
        # minimum of the entire array.
        h[ss] = PP.min()

    k = 1 - h.sum()
    Vprev = mdp_obj.V
    null, value = mdp_obj._bellmanOperator()
    # p 201, Proposition 6.6.5
    span = getSpan(value - Vprev)
    max_iter = (_math.log((mdp_obj.epsilon * (1 - mdp_obj.discount) / mdp_obj.discount) /
                span) / _math.log(mdp_obj.discount * k))
    # mdp_obj.V = Vprev

    mdp_obj.max_iter = int(_math.ceil(max_iter))
def get_range(rank, s):
    local_n = s / size
    #print "local_n", local_n
    local_a = rank * local_n
    local_b = local_a + local_n
    return (local_a, local_b)

def run(mdp_obj):
    # Run the value iteration algorithm.
    if rank==0:
        mdp_obj._startRun()
    local_n = mdp_obj.S / size
    takeQ = np.empty((local_n,mdp_obj.A))
    local_V = np.zeros(local_n)
    local_P = np.zeros(local_n)
    global_V =np.zeros(mdp_obj.S)
    global_P =np.zeros(mdp_obj.S)
    newrank = -1

    #takeQ = np.empty((local_n, mdp_obj.A))

    while True:
    #for itr in xrange(4):
        mdp_obj.iter += 1
        local_a,local_b = get_range(rank,mdp_obj.S)
        Vprev = mdp_obj.V.copy()
        #print ("rank is :" + str(rank) + " this is vprev: " + str(Vprev))
        # Bellman Operator: compute policy and value functions
        #print(repr(local_a) + " , " + repr(local_b) + " , " + repr(local_n))
        myV=comm.bcast(mdp_obj.V, root=0)
        myP=comm.bcast(mdp_obj.policy, root=0)
        #print "myV", rank, myV
        #print rank, takeQ
        mdp_obj.V = myV
        mdp_obj.policy = myP
        takeQ = mdp_obj._bellmanOperator_mpi(local_a, local_b, local_n, V=myV, rankk=rank)
        #print rank, takeQ
        if rank == 0:
            local_P = takeQ.argmax(axis=1)
            local_V = takeQ.max(axis=1)
            global_V[local_a:local_b] = local_V
            global_P[local_a:local_b] = local_P
            for i in xrange(1, size):
                status = MPI.Status()
                takeQ = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                #print "received"
                newrank = status.Get_source()
                la,lb = get_range(newrank,mdp_obj.S)
                local_P = takeQ.argmax(axis=1)
                local_V = takeQ.max(axis=1)
                global_V[la:lb] = local_V
                global_P[la:lb] = local_P
            mdp_obj.V = global_V
            mdp_obj.policy = global_P
            #comm.bcast(mdp_obj.V, root=0)
            #comm.bcast(mdp_obj.policy, root=0)
        else:
            comm.send(takeQ, dest=0, tag=rank)
            #print "sent"
        #comm.bcast(mdp_obj.V, root=0)
        #comm.bcast(mdp_obj.policy, root=0)
        # The values, based on Q. For the function "max()": the option
        # "axis" means the axis along which to operate. In this case it
        # finds the maximum of the the rows. (Operates along the columns?)
        variation = getSpan(mdp_obj.V - Vprev)
        variation = comm.bcast(variation,root=0)
        #print "variation: ", variation
        #print("rank is: " + repr(rank) + " V is " + repr(mdp_obj.V) + "Vprev is " + repr(Vprev))
        #if mdp_obj.verbose:
        #    _printVerbosity(mdp_obj.iter, variation)

        if variation < mdp_obj.thresh:
            #if mdp_obj.verbose:
            #    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
            break
        elif mdp_obj.iter == mdp_obj.max_iter:
            #if mdp_obj.verbose:
            #    print(_MSG_STOP_MAX_ITER)
            break
    if rank==0:
        print str("rank is :")+ str(rank) + str("this is v: ") + str(mdp_obj.V)
        print str("rank is :")+ str(rank) + str("this is policy: ") + str(mdp_obj.policy)
        mdp_obj._endRun(mdp_obj)


def _printVerbosity(iteration, variation):
    if isinstance(variation, float):
        print("{:>10}{:>12f}".format(iteration, variation))
    elif isinstance(variation, int):
        print("{:>10}{:>12d}".format(iteration, variation))
    else:
        print("{:>10}{:>12}".format(iteration, variation))

def getSpan(array):
    """Return the span of `array`

    span(array) = max array(s) - min array(s)

    """
    return array.max() - array.min()

P, R = mdptoolbox.example.forest(S=100)
mdp_obj = final_mdp(P,R,0.96)

initial_value=0
if initial_value == 0:
    mdp_obj.V = np.zeros(mdp_obj.S)
else:
    assert len(initial_value) == mdp_obj.S, "The initial value must be " \
        "a vector of length S."
    mdp_obj.V = np.array(initial_value).reshape(mdp_obj.S)
if mdp_obj.discount < 1:
    # compute a bound for the number of iterations and update the
    # stored value of mdp_obj.max_iter
    _boundIter(mdp_obj)
    # computation of threshold of variation for V for an epsilon-
    # optimal policy
    mdp_obj.thresh = mdp_obj.epsilon * (1 - mdp_obj.discount) / mdp_obj.discount
else:  # discount == 1
    # threshold of variation for V for an epsilon-optimal policy
    mdp_obj.thresh = mdp_obj.epsilon

run(mdp_obj)

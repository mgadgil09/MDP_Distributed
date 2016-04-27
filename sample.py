while True:
    iter += 1

    Vprev = V.copy()

    # Bellman Operator: compute policy and value functions
    policy, V = _bellmanOperator()

    # The values, based on Q. For the function "max()": the option
    # "axis" means the axis along which to operate. In this case it
    # finds the maximum of the the rows. (Operates along the columns?)
    variation = _util.getSpan(V - Vprev)

    if verbose:
    _printVerbosity(iter, variation)

    if variation < thresh:
    if verbose:
        print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
    break
    elif iter == max_iter:
    if verbose:
        print(_MSG_STOP_MAX_ITER)
    break

def _bellmanOperator(self, V=None):
      if V is None:
          V = V
      else:
      # make sure the user supplied V is of the right shape
          try:
              assert V.shape in ((S,), (1, S)), "V is not the " \
                  "right shape (Bellman operator)."
          except AttributeError:
              raise TypeError("V must be a numpy array or matrix.")
      Q = _np.empty((A, S))
      for aa in range(A):
          Q[aa] = R[aa] + discount * P[aa].dot(V)
      return (Q.argmax(axis=0), Q.max(axis=0))

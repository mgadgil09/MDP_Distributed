# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox
=====================================

The MDP toolbox provides classes and functions for the resolution of
descrete-time Markov Decision Processes.

Available modules
-----------------

:mod:`~mdptoolbox.example`
    Examples of transition and reward matrices that form valid MDPs
:mod:`~mdptoolbox.mdp`
    Makov decision process algorithms
:mod:`~mdptoolbox.util`
    Functions for validating and working with an MDP

How to use the documentation
----------------------------
Documentation is available both as docstrings provided with the code and
in html or pdf format from
`The MDP toolbox homepage <http://www.somewhere.com>`_. The docstring
examples assume that the ``mdptoolbox`` package is imported like so::

  >>> import mdptoolbox

To use the built-in examples, then the ``example`` module must be imported::

  >>> import mdptoolbox.example

Once the ``example`` module has been imported, then it is no longer neccesary
to issue ``import mdptoolbox``.

Code snippets are indicated by three greater-than signs::

  >>> x = 17
  >>> x = x + 1
  >>> x
  18

The documentation can be displayed with
`IPython <http://ipython.scipy.org>`_. For example, to view the docstring of
the ValueIteration class use ``mdp.ValueIteration?<ENTER>``, and to view its
source code use ``mdp.ValueIteration??<ENTER>``.

Acknowledgments
---------------
This module is modified from the MDPtoolbox (c) 2009 INRA available at
http://www.inra.fr/mia/T/MDPtoolbox/.

"""

from . import final_mdp
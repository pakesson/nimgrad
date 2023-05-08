import std/math
import std/random
import std/sequtils
import std/strformat
import std/sugar

import nimgrad
import nimgrad/datasets

# This example is based on
#   https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors-and-autograd
when isMainModule:
  randomize()

  let
    X = linspace(-PI, PI, 2000)
    y = X.map(sin)

    Xv = X.map(it => Value(value: it))
    yv = y.map(it => Value(value: it))

    # We want to fit the third-degree polynomial
    #   y = a + b * x + c * x^2 + d * x^3
    # Start with random coefficients w = [a b c d]
    w = randn(4) 

    lr = 1e-7

  for i in countup(1, 10000):
    let
      yPred = Xv.map(x => w[0] + w[1] * x + w[2] * x^2 + w[3] * x^3)
      loss = zip(yPred, yv).map(it => (it[0] - it[1])^2).foldl(a + b)

    if i mod 100 == 0:
      echo &"{i}: loss = {loss.value}"

    for v in w:
      v.grad = 0.0
    loss.backward()
    for v in w:
      v.value -= lr * v.grad

  echo &"Result: y = {w[0].value} + {w[1].value} x + {w[2].value} x^2 + {w[3].value} x^3"

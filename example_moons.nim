import std/random
import std/sequtils
import std/strformat
import std/sugar

import datasets
import mlp
import value

when isMainModule:
  randomize()

  let m = newMLP(@[2, 16, 16, 1])
  echo &"Number of parameters: {m.parameters().toSeq().len}"

  let (X, y) = makeMoons(100, 0.1)

  proc loss(): (Value, float64) =
    let
      Xv = X.map(it => @[Value(value: it[0]), Value(value: it[1])])
      yv = y.map(it => Value(value: it))

      scores = Xv.map(x => m.forward(x)[0])

      losses = zip(yv, scores).map(it => ((-it[0])*it[1] + 1.0).relu())
      dataLoss = losses.foldl(a + b) * (1.0 / losses.len.float)

      alpha = 1e-4
      regLoss = toSeq(m.parameters()).map(it => it*it).foldl(a + b) * alpha
      totalLoss = dataLoss + regLoss

      correctScores = zip(yv, scores).map(it => (if (it[0].value > 0) == (it[1].value > 0): 1 else: 0))
      accuracy = correctScores.foldl(a + b).float * (1.0 / correctScores.len.float)
    result = (totalLoss, accuracy)

  var (totalLoss, accuracy) = loss()
  echo &"Loss: {totalLoss.value}"
  echo &"Accuracy: {accuracy}"

  for i in countup(1, 100):
    (totalLoss, accuracy) = loss()

    m.zeroGrad()
    totalLoss.backward()

    let lr = 1.0 - 0.9*i.float/100.0
    for p in m.parameters():
      p.value -= lr * p.grad

    echo &"Step {i}, loss {totalLoss.value}, accuracy {accuracy*100.float}%"


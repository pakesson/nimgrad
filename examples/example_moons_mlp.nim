import std/random
import std/sequtils
import std/strformat
import std/sugar

import nimgrad
import nimgrad/mlp
import nimgrad/dataprocessing
import nimgrad/datasets

# This example is based on
#   https://github.com/karpathy/micrograd/blob/master/demo.ipynb
when isMainModule:
  randomize()

  let m = newMLP(@[2, 16, 16, 1])
  echo &"Number of parameters: {m.parameters().toSeq().len}"

  let (X, y) = makeMoons(100, 0.1)

  var
    xv = X.map(it => @[Value(value: it[0]), Value(value: it[1])])
    yv = y.map(it => Value(value: it))

  shuffle(xv, yv)

  let
    (xTrain, xTest) = splitData(xv, 0.7)
    (yTrain, yTest) = splitData(yv, 0.7)

  echo &"Train split size: {xTrain.len}"
  echo &"Test split size: {xTest.len}"

  proc loss(yPred: seq[Value], y: seq[Value]): (Value, float64) =
    let
      losses = zip(y, yPred).map(it => ((-it[0])*it[1] + 1.0).relu())
      dataLoss = losses.foldl(a + b) * (1.0 / losses.len.float)

      alpha = 1e-4
      regLoss = toSeq(m.parameters()).map(it => it*it).foldl(a + b) * alpha
      totalLoss = dataLoss + regLoss

      correctScores = zip(y, yPred).map(
        it => (if (it[0].value > 0) == (it[1].value > 0): 1 else: 0))
      accuracy = correctScores.foldl(a + b).float *
        (1.0 / correctScores.len.float)
    result = (totalLoss, accuracy)

  var yPred = xTrain.map(x => m.forward(x)[0])

  var (totalLoss, accuracy) = loss(yPred, yTrain)
  echo &"Loss: {totalLoss.value}"
  echo &"Accuracy: {accuracy}"

  for i in countup(1, 100):
    yPred = xTrain.map(x => m.forward(x)[0])
    (totalLoss, accuracy) = loss(yPred, yTrain)

    m.zeroGrad()
    totalLoss.backward()

    let lr = 1.0 - 0.9*i.float/100.0
    for p in m.parameters():
      p.value -= lr * p.grad

    echo &"Step {i}, loss {totalLoss.value}, accuracy {accuracy*100.float:.4f}%"

  echo "Test split:"
  var yPredTest = xTest.map(x => m.forward(x)[0])
  (totalLoss, accuracy) = loss(yPredTest, yTest)
  echo &"Loss: {totalLoss.value}"
  echo &"Accuracy: {accuracy*100.float:.4f}%"
import std/sequtils
import std/strformat
import std/sugar

import layer
import value

type MLP* = ref object
  layers*: seq[Layer]

proc `$`*(self: MLP): string =
  &"MLP(layers: {self.layers})"

iterator parameters*(self: MLP): Value =
  for l in self.layers:
    for p in l.parameters():
      yield p

proc forward*(self: MLP, input: seq[Value]): seq[Value] =
  assert input.len == self.layers[0].neurons[0].w.len
  result = input
  for l in self.layers:
    result = l.forward(result)

proc zeroGrad*(self: MLP) =
  for p in self.parameters():
    p.grad = 0.0

proc newMLP*(n: seq[int], nonlinear: bool = false): MLP =
  # TODO: Make this more readable
  result = MLP(layers: toSeq(countup(0, n.len()-2)).map(
    i => newLayer(n[i], n[i+1], i < n.len()-2)))

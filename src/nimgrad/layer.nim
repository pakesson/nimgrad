import std/sequtils
import std/strformat
import std/sugar

import neuron
import value

type Layer* = ref object
  neurons*: seq[Neuron]

proc `$`*(self: Layer): string =
  &"Layer(neurons: {self.neurons})"

iterator parameters*(self: Layer): Value =
  for n in self.neurons:
    for p in n.parameters():
      yield p

proc forward*(self: Layer, input: seq[Value]): seq[Value] =
  result = self.neurons.map(x => x.forward(input))

proc newLayer*(nin, nout: int, nonlinear: bool = false): Layer =
  result = Layer(neurons: newSeqWith(nout, newNeuron(nin, nonlinear)))

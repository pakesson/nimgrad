import std/random
import std/sequtils
import std/strformat
import std/sugar

import value

type Neuron* = ref object
  w*: seq[Value]
  b*: Value
  nonlinear: bool

proc `$`*(self: Neuron): string =
  &"Neuron(w: {self.w}, b: {self.b})"

iterator parameters*(self: Neuron): Value =
  for v in self.w:
    yield v
  yield self.b

proc forward*(self: Neuron, input: seq[Value]): Value =
  result = zip(self.w, input).map(x => x[0]*x[1]).foldl(a+b) + self.b
  if self.nonlinear:
    result = relu(result)

# Generate a seq[Value] with random values between -1 and 1
proc randn*(n: int): seq[Value] =
  newSeqWith(n, Value(value: rand(2.0)-1.0))

# Create a new Neuron with n weights
# Weights will have uniform random values between -1 and 1 and zero bias
proc newNeuron*(n: int, nonlinear: bool = false): Neuron =
  # TODO: Make bias optional
  Neuron(w: randn(n), b: Value(value: 0.0), nonlinear: nonlinear)

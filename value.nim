import hashes
import sets
import std/math
import std/strformat

type Value* = ref object
  value*: float64
  grad*: float64
  backproc: proc (self: Value)
  prev: HashSet[Value]

proc `$`*(self: Value): string =
  &"Value(value: {self.value}, grad: {self.grad}, prev: {self.prev})"

# Hash Value based on memory address
proc hash*(v: Value): Hash =
  # !& and !$ are procs from https://nim-lang.org/docs/hashes.html
  var h: Hash = 0
  h = h !& addr(v[]).hash
  result = !$h

proc `+`*(x, y: Value): Value =
  result = Value(value: (x.value + y.value), grad: 0.0, prev: toHashSet([x, y]))
  result.backproc = proc(self: Value) =
    x.grad += self.grad
    y.grad += self.grad

proc `*`*(x, y: Value): Value =
  let res = Value(value: (x.value * y.value), grad: 0.0, prev: toHashSet([x, y]))
  res.backproc = proc(self: Value) =
    x.grad += y.value * res.grad
    y.grad += x.value * res.grad
  result = res

proc `^`*(x: Value, y: float64): Value =
  result = Value(value: (pow(x.value, y)), grad: 0.0, prev: toHashSet([x]))
  result.backproc = proc(self: Value) =
    x.grad += (y * pow(x.value, y-1)) * self.grad

proc `+`*(x: Value, y: float64): Value =
  result = x + Value(value: y)

proc `*`*(x: Value, y: float64): Value =
  result = x * Value(value: y)

proc `-`*(x, y: Value): Value =
  result = x + (y * -1.0)

proc `-`*(x: Value): Value =
  result = x * -1.0

proc `/`*(x, y: Value): Value =
  x * (y^(-1))

proc relu*(x: Value): Value =
  result = Value(value: (if x.value < 0.0: 0.0 else: x.value), grad: 0.0, prev: toHashSet([x]))
  result.backproc = proc(self: Value) =
    x.grad += (if self.value > 0: self.grad else: 0.0)

proc backward*(self: Value) =
  var
    topo: seq[Value]
    visited: HashSet[Value]
  proc buildTopo(v: Value) =
    if v notin topo:
      visited.incl(v)
      for child in v.prev:
        buildTopo(child)
      topo.add(v)
  buildTopo(self)
  self.grad = 1
  for i in countdown(topo.len-1, 0):
    let v: Value = topo[i]
    if v.backproc != nil:
      v.backproc(v)

import std/random

proc shuffle*[T, U](r: var Rand; x: var openArray[T], y: var openArray[U]) =
  ## Shuffles two sequences of elements in-place using the given state.
  assert x.len == y.len
  for i in countdown(x.high, 1):
    let j = r.rand(i)
    swap(x[i], x[j])
    swap(y[i], y[j])

proc shuffle*[T, U](x: var openArray[T], y: var openArray[U]) =
  ## Shuffles two sequences of elements in-place using the given state.
  assert x.len == y.len
  for i in countdown(x.high, 1):
    let j = rand(i)
    swap(x[i], x[j])
    swap(y[i], y[j])

proc splitData*[T](x: seq[T], split: float): (seq[T], seq[T]) =
  let idxSplit = int(x.high.float * split)
  result = (x[0..idxSplit], x[idxSplit+1..^1])

import std/math
import std/random
import std/sequtils
import std/sugar

proc linspace(start: float64, stop: float64, n: int): seq[float64] =
  let step: float64 = (stop - start) / n.float
  result = toSeq(countup(0, n-1)).map(x => x.float*step)

# Based on make_moons from scikit-learn
#   https://github.com/scikit-learn/scikit-learn/blob/27cf2dd22a0399247bff8281bb4e1c75a8cbc532/sklearn/datasets/_samples_generator.py#L789
proc makeMoons*(n: int = 100, noise: float64): (seq[(float64, float64)], seq[float64]) =
  let
    samplesOuter = n div 2
    samplesInner = n - samples_outer
    linOuter = linspace(0, PI, samplesOuter)
    linInner = linspace(0, PI, samplesInner)

    # Note: gauss() is not thread safe
    outerCircX = linOuter.map(x => cos(x) + gauss(0.0, noise))
    outerCircY = linOuter.map(x => sin(x) + gauss(0.0, noise))
    innerCircX = linInner.map(x => 1 - cos(x) + gauss(0.0, noise))
    innerCircY = linInner.map(x => 0.5 - sin(x) + gauss(0.0, noise))

    X: seq[(float64, float64)] = zip(outerCircX & innerCircX, outerCircY & innerCircY)
    y: seq[float64] = repeat(-1.0, samplesOuter) & repeat(1.0, samplesInner)

  result = (X, y)

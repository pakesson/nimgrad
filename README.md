# Nimgrad

A Nim implementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)
engine.

Nimgrad implements backpropagation (reverse-mode automatic differentiation)
with a dynamic computational graph for scalar values.

## Example: Moons

```
$ ./example_moons
Number of parameters: 337
Loss: 1.147359684619476
Accuracy: 0.54
Step 1, loss 1.147359684619476, accuracy 54.0%
Step 2, loss 1.965013178924068, accuracy 69.0%
Step 3, loss 0.9287053082033497, accuracy 51.0%
Step 4, loss 0.4714732925523774, accuracy 83.0%
Step 5, loss 0.3258205843367792, accuracy 84.0%
Step 6, loss 0.2704531411171724, accuracy 86.0%
Step 7, loss 0.2392688366712554, accuracy 88.0%
Step 8, loss 0.2194895852417177, accuracy 88.0%
Step 9, loss 0.196107510332574, accuracy 89.0%
Step 10, loss 0.2043266041114352, accuracy 90.0%
Step 11, loss 0.2302008706081193, accuracy 92.0%
Step 12, loss 0.1686595677833073, accuracy 90.0%
Step 13, loss 0.1442449358816608, accuracy 94.0%
Step 14, loss 0.1614259123216442, accuracy 96.0%
Step 15, loss 0.1157517404193201, accuracy 95.0%
Step 16, loss 0.1290660788981484, accuracy 95.0%
Step 17, loss 0.1285530948931498, accuracy 95.0%
Step 18, loss 0.1534836698750388, accuracy 95.0%
Step 19, loss 0.08046316742823116, accuracy 99.0%
Step 20, loss 0.1732505082207434, accuracy 94.0%
Step 21, loss 0.3244990927395166, accuracy 87.0%
Step 22, loss 0.2040792111031532, accuracy 92.0%
Step 23, loss 0.08565089077193677, accuracy 97.0%
Step 24, loss 0.06467808493307796, accuracy 97.0%
Step 25, loss 0.05968275638057095, accuracy 97.0%
Step 26, loss 0.1435251714184601, accuracy 93.0%
Step 27, loss 0.05748710900496506, accuracy 98.0%
Step 28, loss 0.04432617657804992, accuracy 99.0%
Step 29, loss 0.05533495097027716, accuracy 99.0%
Step 30, loss 0.05152180423546464, accuracy 98.0%
Step 31, loss 0.05290320467896766, accuracy 99.0%
Step 32, loss 0.0385763886870687, accuracy 100.0%
Step 33, loss 0.02365311590111107, accuracy 100.0%
Step 34, loss 0.03050618331783393, accuracy 100.0%
Step 35, loss 0.05359055481288524, accuracy 99.0%
Step 36, loss 0.05358567747905438, accuracy 98.0%
Step 37, loss 0.03597418573321854, accuracy 99.0%
Step 38, loss 0.01965703016085822, accuracy 100.0%
Step 39, loss 0.0251733842273363, accuracy 100.0%
Step 40, loss 0.02570469797013274, accuracy 100.0%
Step 41, loss 0.02135415017381018, accuracy 100.0%
Step 42, loss 0.02476180877040379, accuracy 100.0%
Step 43, loss 0.01533579812315701, accuracy 100.0%
Step 44, loss 0.01223289471208943, accuracy 100.0%
Step 45, loss 0.01747709250844502, accuracy 100.0%
Step 46, loss 0.01744414800557899, accuracy 100.0%
Step 47, loss 0.01698971166565929, accuracy 100.0%
Step 48, loss 0.01936284214319623, accuracy 100.0%
Step 49, loss 0.01474255750375453, accuracy 100.0%
Step 50, loss 0.01182182772980455, accuracy 100.0%
Step 51, loss 0.01181922707074811, accuracy 100.0%
Step 52, loss 0.01181666952838053, accuracy 100.0%
Step 53, loss 0.01181415507488093, accuracy 100.0%
Step 54, loss 0.01181168368289982, accuracy 100.0%
Step 55, loss 0.0118092553255586, accuracy 100.0%
Step 56, loss 0.01180686997644905, accuracy 100.0%
Step 57, loss 0.01180452760963288, accuracy 100.0%
Step 58, loss 0.01180222819964125, accuracy 100.0%
Step 59, loss 0.01179997172147429, accuracy 100.0%
Step 60, loss 0.01179775815060068, accuracy 100.0%
Step 61, loss 0.01179558746295719, accuracy 100.0%
Step 62, loss 0.01179345963494825, accuracy 100.0%
Step 63, loss 0.01179137464344549, accuracy 100.0%
Step 64, loss 0.01178933246578737, accuracy 100.0%
Step 65, loss 0.01178733307977872, accuracy 100.0%
Step 66, loss 0.01178537646369042, accuracy 100.0%
Step 67, loss 0.01178346259625889, accuracy 100.0%
Step 68, loss 0.0117815914566858, accuracy 100.0%
Step 69, loss 0.01177976302463763, accuracy 100.0%
Step 70, loss 0.01177797728024538, accuracy 100.0%
Step 71, loss 0.01177623420410411, accuracy 100.0%
Step 72, loss 0.01177453377727266, accuracy 100.0%
Step 73, loss 0.01177287598127329, accuracy 100.0%
Step 74, loss 0.01177126079809134, accuracy 100.0%
Step 75, loss 0.01176968821017492, accuracy 100.0%
Step 76, loss 0.01176815820043451, accuracy 100.0%
Step 77, loss 0.01176667075224283, accuracy 100.0%
Step 78, loss 0.01176522584943434, accuracy 100.0%
Step 79, loss 0.01176382347630505, accuracy 100.0%
Step 80, loss 0.01176246361761224, accuracy 100.0%
Step 81, loss 0.01176114625857415, accuracy 100.0%
Step 82, loss 0.01175987138486973, accuracy 100.0%
Step 83, loss 0.01175863898263838, accuracy 100.0%
Step 84, loss 0.0117574490384797, accuracy 100.0%
Step 85, loss 0.0117563015394532, accuracy 100.0%
Step 86, loss 0.01175519647307815, accuracy 100.0%
Step 87, loss 0.01175413382733333, accuracy 100.0%
Step 88, loss 0.01175311359065673, accuracy 100.0%
Step 89, loss 0.01175213575194545, accuracy 100.0%
Step 90, loss 0.01175120030055546, accuracy 100.0%
Step 91, loss 0.01175030722630135, accuracy 100.0%
Step 92, loss 0.01174945651945624, accuracy 100.0%
Step 93, loss 0.01174864817075153, accuracy 100.0%
Step 94, loss 0.0117478821713768, accuracy 100.0%
Step 95, loss 0.01174715851297954, accuracy 100.0%
Step 96, loss 0.01174647718766515, accuracy 100.0%
Step 97, loss 0.01174583818799666, accuracy 100.0%
Step 98, loss 0.01174524150699463, accuracy 100.0%
Step 99, loss 0.01174468713813716, accuracy 100.0%
Step 100, loss 0.01174417507535948, accuracy 100.0%
```
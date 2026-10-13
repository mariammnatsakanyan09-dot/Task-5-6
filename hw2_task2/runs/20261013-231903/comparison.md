# Comparison with the focusing result from the lecture

| | macropixels N | enhancement | textbook N/(2 pi) | enhancement / (N/(2 pi)) |
|---|--:|--:|--:|--:|
| lecture | not given | 33 | - | - |
| this run, best (experiment 1), ring background as in the task | 256 | 29.1 | 40.7 | 0.71 |
| same experiment, against the far field | 256 | 13.5 | 40.7 | 0.33 |

The best enhancement of the agent, 29.1 (experiment 1, 256 macropixels), is somewhat lower than the lecture's 33 (ratio 29.1 / 33 = 0.88).

What the enhancement is made of in the best experiment: the mean focus intensity rose by a factor 9.3 (from 36.1 to 336.1 counts) and the mean background fell by a factor 2.3 (from 26.3 to 11.6 counts). The search keeps a flip when the ratio focus/background rises, so it lowers the proximal background as well as raising the focus. Against the camera area that the search never used (the far field) the same focus is 13.5 times brighter than the background there, which is the enhancement of the focus without the lowered ring.

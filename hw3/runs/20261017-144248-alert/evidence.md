# Human-in-the-loop evidence

## Pause 1: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please uncover the lens (and wipe it if it is dirty) and reply "done".
reason: Sharpness_norm 134.1 is below the 228 limit while brightness (132.36) and noise (-53.3 dBFS) are normal, so only a person can uncover the lens before verifying.
```

**Reply typed** (turn `human`):

```
done
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: The person has uncovered the lens and replied done. Take one verification reading now: measure_region on the 'scene' at the current resolution (1920x1080), do not change the resolution, do not play tones and do not call camera_info. Report mean_brightness, sharpness and sharpness_norm. Compare with the watchdog limits: brightness 93-155, sharpness_norm at least 228 (a reading inside the limits is normal). Earlier reading before the action: brightness 132.36, sharpness_norm 134.1, room noise -53.3 dBFS. If sharpness_norm is still below 228, take one more measure_region a few seconds later to confirm. Report in at most 3 sentences, without tables.
reason: The person did the requested action, so one hardware verification reading is needed to compare with the limits.
```

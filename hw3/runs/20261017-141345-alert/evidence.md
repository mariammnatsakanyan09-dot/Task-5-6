# Human-in-the-loop evidence

## Pause 1: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please uncover the lens (remove whatever covers or blocks it) and reply done.
reason: Sharpness_norm is about 217 (21% of normal 1027.4) while brightness (105.4) is near normal, which points to a covered or blocked lens that only a person can uncover.
```

**Reply typed** (turn `human`):

```
done
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: The person has uncovered the lens and replied done. Measure once more and compare with the normal values (scene brightness 119.6, sharpness_norm 1027.4, room noise -50.6 dBFS). 1) Call measure_region on 'scene' with n_frames=10 at the current resolution (1920x1080) and report mean_brightness, sharpness, sharpness_norm, saturated_fraction, rms_contrast and brightness_noise_std. 2) If sharpness_norm is still low (the camera needs about 10 s to readjust), repeat measure_region once more a few seconds later; do not measure the same thing more than twice. 3) Take one listen_noise reading and report noise_rms_db in dBFS. 4) Compare each reading with the normal values (sharpness_norm 1027.4, brightness 119.6, noise -50.6 dBFS) and the earlier readings (before: brightness 105.4, sharpness_norm 216.5, noise -44.1 dBFS) and say whether they are back within the normal range. Do not ask the person for anything. Return the numbers exactly as measured.
reason: The person has uncovered the lens, so a new real measurement is needed to compare with the normal values and confirm the diagnosis.
```

## Pause 2: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please wipe the camera lens gently with a clean, soft cloth, then reply "done".
reason: After the lens was uncovered, two readings a few seconds apart are still low (sharpness_norm 395.5 and 403.1 against a normal 1027.4) while brightness and noise are back to normal, so a dirty or smudged lens is likely and only a person can wipe it.
```

**Reply typed** (turn `human`):

```
done
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: The person has wiped the lens and replied done. Wait a few seconds for the camera to readjust, then call measure_region on 'scene' with n_frames=10 at 1920x1080 and report mean_brightness, sharpness, sharpness_norm, saturated_fraction, rms_contrast, colour_saturation and brightness_noise_std. If sharpness_norm is still low, repeat measure_region once more a few seconds later; do not measure the same thing more than twice. Compare with the normal values (brightness 119.6, sharpness_norm 1027.4) and with the earlier readings: before uncovering brightness 105.4 and sharpness_norm 216.5; after uncovering 122.45/395.5 and 122.97/403.1. Optionally take one listen_noise reading and compare it with the normal -50.6 dBFS (earlier -49.2). Say whether the readings are back within the normal range. Do not ask the person for anything. Return the numbers exactly as measured.
reason: The person has wiped the lens, so one more real measurement is needed to compare with the normal values and confirm or reject the dirty-lens diagnosis.
```

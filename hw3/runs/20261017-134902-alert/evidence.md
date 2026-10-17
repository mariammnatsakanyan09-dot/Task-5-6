# Human-in-the-loop evidence

## Pause 1: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please uncover the lens (remove anything covering or blocking it) and reply "done".
reason: Sharpness collapsed with near-normal brightness while the sound side is normal, so a covered, dirty or blocked lens is likely and only a person can uncover it before the re-measurement.
```

**Reply typed** (turn `human`):

```
done
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: The person has uncovered the lens and replied 'done'. Measure the scene region once more at 1920x1080 (measure_region on 'scene', n_frames=5, plus a capture labelled 'after_uncover' so the picture is saved). Report mean_brightness, sharpness, sharpness_norm, saturated_fraction, rms_contrast and colour_saturation exactly as measured. Compare with the normal values: brightness 111.3 and sharpness_norm 940.2. The earlier readings with the lens blocked were: mean_brightness 103.54, sharpness_norm 203.7 (main_view sharpness_norm 6.4, mean_brightness 40.54). Say whether sharpness_norm and brightness are back within the normal range. If they are back, the sound does not need to be measured again, since it was normal (room noise -49.6 dBFS, 1000 Hz tone 18.7 dB above the noise). Do not ask the person for anything.
reason: The person has uncovered the lens, so one new camera reading is needed to compare with the normal values before finishing.
```

## Pause 2: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please wipe the camera lens gently with a clean, dry, soft cloth (no liquid) and reply "done".
reason: After uncovering, scene sharpness_norm is only 311 against a normal 940 while brightness is normal, which points to a smudge or film on the lens that only a person can wipe off before one more measurement.
```

**Reply typed** (turn `human`):

```
done
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: The person has wiped the lens and replied 'done'. Measure the scene region once more at 1920x1080 (measure_region on 'scene', n_frames=5, plus a capture labelled 'after_wipe' so the picture is saved). Report mean_brightness, sharpness, sharpness_norm, saturated_fraction, rms_contrast and colour_saturation exactly as measured. Compare with the normal values: brightness 111.3 and sharpness_norm 940.2. Earlier readings: lens blocked: mean_brightness 103.54, sharpness_norm 203.7, colour_saturation 92.82, rms_contrast 75.29; after uncovering: mean_brightness 120.34, sharpness_norm 311.3, colour_saturation 16.2, rms_contrast 37.25 (main_view sharpness_norm 153.9, panorama 501.8). Say whether brightness and sharpness_norm are back within the normal range. The sound was normal earlier (room noise -49.6 dBFS, 1000 Hz tone 18.7 dB above the noise), so do not measure it again. Do not ask the person for anything.
reason: The person has wiped the lens, so one new camera reading is needed to compare with the normal values before finishing.
```

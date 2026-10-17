# Human-in-the-loop evidence

## Pause 1: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: The check tone is not heard even at the maximum level (-12 dBFS), while the camera and microphone readings are normal. Please press the volume-up button on the Pano-360 device (and make sure the loudspeaker is not muted), then reply "done".
reason: The tone is not heard at -20 or -12 dBFS while the camera and microphone readings are normal, so only a person pressing the device's volume button can fix it, and that must happen before the next hardware measurement.
```

**Reply typed** (turn `human`):

```
done
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: The person pressed the volume-up button on the Pano-360 and replied done. Measure again and compare with the normal values (scene brightness 118.2, sharpness_norm 389.5, room noise -52.2 dBFS, 1000 Hz check tone 19.7 dB above the noise). 1) Play the 1000 Hz check tone at -20 dBFS (label 'after_volume20') and read snr_db. If it is not heard 15 dB or more above the noise, replay at -12 dBFS (the maximum). Do not repeat the same measurement more than twice. 2) Run listen_noise once for the room noise. 3) Run measure_region on the scene once more. Report how many dB you had to add (the loss, 0 if -20 dBFS is enough) and whether the tone is now heard 15 dB or more above the noise, and whether the readings are back within the normal range. Return all readings exactly. If the tone is still not heard at -12 dBFS, say so plainly. Do not ask the person for anything.
reason: The person has pressed the volume button, so the instrument must be measured again to check whether the tone is heard and to compensate with the tone level.
```

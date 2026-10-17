# Human-in-the-loop evidence

## Pause 1: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please press the volume – button on the device three times, then let me know when you are done.
reason: We need the person to press the volume – button three times before we can take the 'after' measurement.
```

**Reply typed** (turn `human`):

```
done, - 3 times
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: Play a 1000 Hz tone at -30 dBFS through the loudspeaker and measure it with the microphone; label this reading as 'after' and return the measured loudness, room noise, and heard/not heard verdict.
reason: We need the measurement after the volume – button presses to compare with the baseline.
```

## Pause 2: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please press the volume + button on the device three times, then let me know when you are done.
reason: We need the person to press the volume + button three times before we can take the next tone measurement.
```

**Reply typed** (turn `human`):

```
done, + 3 times
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: Play a 1000 Hz tone at -30 dBFS through the loudspeaker and measure it with the microphone; label this reading as 'after_plus'. Then compare this reading with the earlier 'before' reading and return the measured loudness, room noise, heard/not heard verdict for both, and the delta tone_db.
reason: To check if pressing the volume + button three times restores the loudness to the original level.
```

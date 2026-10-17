# Human-in-the-loop evidence

## Pause 1: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please set the device's volume (using its volume button) to a level where a 1000 Hz tone would be heard clearly (you can judge by listening). When you have done so, reply with 'done'.
reason: We need the user to set the device volume so that a 1000 Hz tone is audible before measuring noise.
```

**Reply typed** (turn `human`):

```
done, volume in the middle
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: First, run listen_noise to obtain the room noise level (in dBFS). Then set the tone level to that noise value plus 20 dBFS. Verify that a 1000 Hz tone played at this level is heard (the play_tone result should show SNR >= 20 dB and verdict 'heard'). If not, adjust the level upward in 2 dB steps until the condition is met. Once the level is established, play_tone at each of the frequencies 300, 600, 1000, 2000, 3000, 4000, and 6000 Hz using that fixed level, and record for each frequency the microphone-measured level (in dBFS) returned by play_tone. Save all these readings (frequency and measured level) to the working directory (they will be appended to instrument_log.jsonl).
reason: We need to determine a suitable playback level based on ambient noise and then measure the device's loudspeaker-to-microphone response across the specified frequencies.
```

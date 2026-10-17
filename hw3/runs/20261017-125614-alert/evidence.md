# Human-in-the-loop evidence

## Pause 1: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please check if the lens is covered or obstructed, and if the volume or mute button needs adjustment. Reply 'done' after addressing these issues.
reason: The next step is to ask the person to check for obstructions or issues with the lens and adjust the volume or mute button if necessary.
```

**Reply typed** (turn `human`):

```
done
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: Measure the brightness and sharpness of the scene again at the resolution 1920x1080 to confirm if the readings have returned to normal values after the adjustments made by the person.
reason: The next step is to recheck the readings to see if the adjustments made by the person resolved the sharpness issue.
```

# Human-in-the-loop evidence

## Pause 1: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please check if the lens is covered or if there are any obstructions. Let me know once you've made any adjustments, and I will take the readings again to confirm if they return to normal values.
reason: The next step is to ask the person to check the lens or any obstructions that might be affecting the sharpness readings.
```

**Reply typed** (turn `human`):

```
lens are not covered anymore, done
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: Take readings of the scene brightness, sharpness, and sharpness norm again to confirm if they return to normal values after checking the lens.
reason: The next step is to gather data again from the instrument to see if the adjustments made by the person have resolved the sharpness issue.
```

## Pause 2: scene_action

**Question the graph posed** (orchestrator turn):

```
next: human
kind: scene_action
question: Please check for any additional obstructions or changes in the environment that could be affecting the sharpness readings. Let me know if any adjustments can be made.
reason: The next step is to ask the person if there are any additional obstructions or changes in the environment that could be affecting the readings.
```

**Reply typed** (turn `human`):

```
done
```

**Next message(s) in the dialogue**:

`[orchestrator]`

```
next: hardware
instruction: Take readings of the scene brightness, sharpness, and sharpness norm again to confirm if they return to normal values after checking for additional obstructions or changes in the environment.
reason: The next step is to gather data again from the instrument to see if the adjustments made by the person have resolved the sharpness issue.
```

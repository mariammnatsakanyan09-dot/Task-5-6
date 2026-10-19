#!/bin/bash
# Starts the guard on a Mac so that it survives a night: the Mac does not go to sleep (caffeinate), the key comes from the Keychain (it is never written
# in a file), and watching begins by itself (--autostart). Extra arguments are passed on:
#   MODEL=<openrouter model id> ./start_guard.sh --guard-hours 22:00-07:00 --sensor-port 8001
#
# One time (the commands ask for the secrets, nothing is typed on the command line):
#   security add-generic-password -a "$USER" -s openrouter -w                  # the OpenRouter key
#   security add-generic-password -a "$USER" -s labwatch-sensor -w             # (optional) the secret of the sensor door, 12+ characters
#   security add-generic-password -a "$USER" -s labwatch-heartbeat -w          # (optional) the ping address of healthchecks.io or similar
#   security add-generic-password -a "$USER" -s labwatch-notify -w             # (optional) the push address, for example https://ntfy.sh/<secret-topic>
cd "$(dirname "$0")" || exit 1
kc() { security find-generic-password -a "$USER" -s "$1" -w 2>/dev/null; }
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$(kc openrouter)}"
[ -n "$OPENROUTER_API_KEY" ] || { echo "no key: security add-generic-password -a \"$USER\" -s openrouter -w"; exit 1; }
[ -n "$HW3_SENSOR_TOKEN" ]   || { v="$(kc labwatch-sensor)";    [ -n "$v" ] && export HW3_SENSOR_TOKEN="$v"; }
[ -n "$HW3_HEARTBEAT_URL" ]  || { v="$(kc labwatch-heartbeat)"; [ -n "$v" ] && export HW3_HEARTBEAT_URL="$v"; }
[ -n "$HW3_NOTIFY_URL" ]     || { v="$(kc labwatch-notify)";    [ -n "$v" ] && export HW3_NOTIFY_URL="$v"; }
[ -n "$MODEL" ] || [ -n "$AGENT_MODEL" ] || { echo "say which model: MODEL=<openrouter model id> ./start_guard.sh ..."; exit 1; }
MODEL_ARGS=(); [ -n "$MODEL" ] && MODEL_ARGS=(--model "$MODEL")
exec caffeinate -i python3 web_app.py "${MODEL_ARGS[@]}" --autostart "$@"

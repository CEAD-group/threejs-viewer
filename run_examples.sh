#!/usr/bin/env bash
total=$(ls examples/0{1..9}_*.py examples/1[0-9]_*.py 2>/dev/null | wc -l | tr -d ' ')
i=0
for f in examples/0{1..9}_*.py examples/1[0-9]_*.py; do
    i=$((i + 1))
    echo "[$i/$total] Starting $f..."
    uv run "$f" &
    pid=$!
    echo "[$i/$total] PID $pid — waiting 5s (press any key to skip)..."
    read -t 20 -n 1 -s key
    if kill -0 $pid 2>/dev/null; then
        echo "[$i/$total] Killing $pid (still running)"
        kill $pid 2>/dev/null
    else
        echo "[$i/$total] $f exited on its own"
    fi
    wait $pid 2>/dev/null
    echo "[$i/$total] Done with $f"
    echo
done
echo "All $total examples done."

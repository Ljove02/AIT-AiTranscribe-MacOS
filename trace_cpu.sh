#!/bin/bash
#
# trace_cpu.sh — AiTranscribe CPU tracer
# =======================================
# Watches the running AiTranscribe app and captures everything needed to
# explain where CPU goes:
#
#   - 1s CPU/memory/thread timeline (timeline.csv), with true CPU% computed
#     from cputime deltas (not ps's decayed average)
#   - automatic deep call-stack captures (`sample`, 5s) whenever CPU crosses
#     the spike threshold, plus one idle baseline sample at startup
#   - per-thread CPU snapshot (ps -M) at every spike
#   - window census every tick via a tiny compiled Swift helper: total
#     windows, and whether a floating indicator window (layer 3) exists —
#     this is how we catch the "indicator window leaked and still animating
#     off-screen" bug
#   - recording detection: the indicator level monitor writes
#     ~/Library/Application Support/AiTranscribe/indicator_level.log while
#     the mic level monitor runs, so a fresh mtime marks recording periods
#     in the timeline automatically
#   - manual markers: type anything + Enter while it runs ("rec start",
#     "stopped", "opened settings") and it lands in the timeline
#   - SUMMARY.md on Ctrl-C: peak/mean CPU, spike list, hottest app frames
#     aggregated across all samples, window-leak verdict
#
# Usage:
#   ./trace_cpu.sh                 # defaults: spike at 40%, 5s samples
#   SPIKE=60 ./trace_cpu.sh        # only deep-sample above 60% CPU
#
# Stop with Ctrl-C. Results in perf_traces/trace_<timestamp>/.

set -u

APP_NAME="AiTranscribe"
SPIKE="${SPIKE:-40}"              # CPU % that triggers a deep sample
SAMPLE_SECS="${SAMPLE_SECS:-5}"   # duration of each `sample` capture
COOLDOWN="${COOLDOWN:-20}"        # min seconds between deep samples
MAX_SAMPLES="${MAX_SAMPLES:-15}"  # safety cap per run
TICK=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/perf_traces/trace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/samples"

CSV="$OUT/timeline.csv"
INDLOG="$HOME/Library/Application Support/AiTranscribe/indicator_level.log"

echo "epoch,time,pid,cpu_true,cpu_ps,rss_mb,threads,windows,indicator_win,recording,note" > "$CSV"

# ---------------------------------------------------------------------------
# Window census helper (compiled once; falls back gracefully if swiftc fails)
# ---------------------------------------------------------------------------
cat > "$OUT/listwins.swift" <<'EOF'
import CoreGraphics
import Foundation
guard CommandLine.arguments.count > 1, let pid = Int(CommandLine.arguments[1]) else { exit(1) }
let info = (CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as? [[String: Any]]) ?? []
var total = 0
var indicator = 0
var lines: [String] = []
for w in info where (w["kCGWindowOwnerPID"] as? Int) == pid {
    total += 1
    let layer = w["kCGWindowLayer"] as? Int ?? -1
    let on = (w["kCGWindowIsOnscreen"] as? Bool) ?? false
    var width = 0, height = 0
    if let b = w["kCGWindowBounds"] as? [String: Any] {
        width = (b["Width"] as? Int) ?? 0
        height = (b["Height"] as? Int) ?? 0
    }
    // The recording indicator is a small borderless window at floating level
    if layer == 3 && width <= 100 && height <= 60 { indicator += 1 }
    lines.append("layer=\(layer) onscreen=\(on) size=\(width)x\(height)")
}
print("\(total) \(indicator)")
for l in lines { print(l) }
EOF

WINHELPER="$OUT/listwins"
if ! swiftc -O "$OUT/listwins.swift" -o "$WINHELPER" 2>"$OUT/swiftc.err"; then
    WINHELPER=""
    echo "note: swiftc unavailable — window census disabled (see swiftc.err)"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
find_pid() {
    pgrep -x "$APP_NAME" 2>/dev/null | head -1
}

# ps cputime looks like MM:SS.ss or HH:MM:SS — convert to seconds
cputime_secs() {
    ps -o cputime= -p "$1" 2>/dev/null | awk -F: '
        NF==3 { printf "%.2f", $1*3600 + $2*60 + $3; exit }
        NF==2 { printf "%.2f", $1*60 + $2; exit }
        { printf "0"; exit }'
}

recording_active() {
    [ -f "$INDLOG" ] || { echo 0; return; }
    local mtime now
    mtime=$(stat -f %m "$INDLOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ $((now - mtime)) -le 2 ]; then echo 1; else echo 0; fi
}

SAMPLE_COUNT=0
LAST_SAMPLE=0
SAMPLING_FLAG="$OUT/.sampling"

take_deep_capture() {
    local pid="$1" cpu="$2" tag="$3"
    local stamp
    stamp=$(date +%H%M%S)
    local base="$OUT/samples/${stamp}_${tag}_cpu${cpu%%.*}"

    touch "$SAMPLING_FLAG"
    ps -M -p "$pid" > "${base}_threads.txt" 2>&1
    if [ -n "$WINHELPER" ]; then
        "$WINHELPER" "$pid" > "${base}_windows.txt" 2>&1
    fi
    (
        sample "$pid" "$SAMPLE_SECS" -file "${base}_sample.txt" >/dev/null 2>&1
        rm -f "$SAMPLING_FLAG"
    ) &
    echo ">>> deep capture started: ${base##*/} (cpu=${cpu}%)"
}

# ---------------------------------------------------------------------------
# Summary on exit
# ---------------------------------------------------------------------------
write_summary() {
    echo ""
    echo "Writing summary..."
    local S="$OUT/SUMMARY.md"
    {
        echo "# CPU trace summary — $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        awk -F, 'NR>1 && $4!="" {
            n++; sum+=$4; if ($4>max) {max=$4; maxt=$2}
            if ($10==1) { rn++; rsum+=$4; if ($4>rmax) rmax=$4 }
            else        { in_++; isum+=$4; if ($4>imax) imax=$4 }
        } END {
            if (n) printf "- Ticks: %d | mean CPU %.1f%% | peak %.1f%% at %s\n", n, sum/n, max, maxt
            if (rn) printf "- While RECORDING: %d ticks | mean %.1f%% | peak %.1f%%\n", rn, rsum/rn, rmax
            if (in_) printf "- While idle: %d ticks | mean %.1f%% | peak %.1f%%\n", in_, isum/in_, imax
        }' "$CSV"
        echo ""

        echo "## Window-leak check"
        awk -F, 'NR>1 && $9!="" {
            last_ind=$9; last_rec=$10; last_t=$2
            if ($10==0 && $9>0) leak++
        } END {
            printf "- Final state at %s: indicator windows=%s, recording=%s\n", last_t, last_ind, last_rec
            if (leak) printf "- LEAK: indicator window existed while NOT recording for %d ticks\n", leak
            else print "- No indicator window observed outside recording."
        }' "$CSV"
        echo ""

        echo "## Deep samples"
        if ls "$OUT"/samples/*_sample.txt >/dev/null 2>&1; then
            for f in "$OUT"/samples/*_sample.txt; do
                echo ""
                echo "### ${f##*/}"
                echo ""
                echo "Hottest app frames (self+children sample counts):"
                echo '```'
                grep "(in $APP_NAME)" "$f" \
                    | sed -E "s/^[^0-9]*([0-9]+) +(.+) +\(in $APP_NAME\)[^[]*\[[^]]+\] *(.*)$/\1  \2  \3/" \
                    | sort -rn | head -15
                echo '```'
                echo ""
                echo "Top of stack (leaf functions, all binaries):"
                echo '```'
                awk '/Sort by top of stack/{f=1;next} f&&NF==0{exit} f' "$f" | head -12
                echo '```'
            done
        else
            echo "(no spike samples were triggered)"
        fi
        echo ""
        echo "## Manual markers"
        awk -F, 'NR>1 && $11!="" && $11!=" " {print "- " $2 ": " $11}' "$CSV"
    } > "$S"

    echo ""
    echo "Done. Results:"
    echo "  $OUT/SUMMARY.md"
    echo "  $OUT/timeline.csv"
    echo "  $OUT/samples/"
}

trap 'write_summary; exit 0' INT TERM

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo "Tracing $APP_NAME → $OUT"
echo "Spike threshold: ${SPIKE}% | sample length: ${SAMPLE_SECS}s | Ctrl-C to stop"
echo "Tip: type a note + Enter at any time to mark an event (e.g. 'rec start')"
echo ""

PREV_PID=""
PREV_CPUTIME=""
PREV_EPOCH=""
BASELINE_DONE=0

while true; do
    NOTE=""
    # read doubles as the tick sleeper and the marker input; if stdin is
    # closed (EOF, rc<=128) read returns instantly, so sleep instead
    if read -t "$TICK" -r NOTE 2>/dev/null; then
        NOTE=$(echo "$NOTE" | tr ',' ';')
    elif [ $? -le 128 ]; then
        sleep "$TICK"
    fi

    PID=$(find_pid)
    EPOCH=$(date +%s)
    TS=$(date +%H:%M:%S)

    if [ -z "$PID" ]; then
        echo "$EPOCH,$TS,,,,,,,,,${NOTE:-app not running}" >> "$CSV"
        PREV_PID=""; PREV_CPUTIME=""; PREV_EPOCH=""
        continue
    fi

    if [ "$PID" != "$PREV_PID" ]; then
        [ -n "$PREV_PID" ] && NOTE="${NOTE:+$NOTE; }app restarted (pid $PREV_PID -> $PID)"
        PREV_PID="$PID"; PREV_CPUTIME=""; PREV_EPOCH=""
    fi

    read -r CPU_PS RSS_KB <<< "$(ps -o pcpu=,rss= -p "$PID" 2>/dev/null | awk '{print $1, $2}')"
    CPU_PS="${CPU_PS:-0}"
    RSS_MB=$(( ${RSS_KB:-0} / 1024 ))
    THREADS=$(ps -M -p "$PID" 2>/dev/null | awk 'END{print NR-1}')

    CT=$(cputime_secs "$PID")
    CPU_TRUE=""
    if [ -n "$PREV_CPUTIME" ] && [ -n "$PREV_EPOCH" ] && [ "$EPOCH" -gt "$PREV_EPOCH" ]; then
        CPU_TRUE=$(echo "$CT $PREV_CPUTIME $EPOCH $PREV_EPOCH" \
            | awk '{ printf "%.1f", ($1-$2)/($3-$4)*100 }')
    fi
    PREV_CPUTIME="$CT"; PREV_EPOCH="$EPOCH"

    WINS=""; INDWIN=""
    if [ -n "$WINHELPER" ]; then
        read -r WINS INDWIN <<< "$("$WINHELPER" "$PID" 2>/dev/null | head -1)"
    fi

    REC=$(recording_active)

    echo "$EPOCH,$TS,$PID,${CPU_TRUE},${CPU_PS},${RSS_MB},${THREADS},${WINS},${INDWIN},${REC},${NOTE}" >> "$CSV"

    CPU_NOW="${CPU_TRUE:-$CPU_PS}"
    CPU_INT="${CPU_NOW%%.*}"; CPU_INT="${CPU_INT:-0}"

    # one idle baseline sample, taken only once the app is actually quiet
    if [ "$BASELINE_DONE" -eq 0 ] && [ -n "$CPU_TRUE" ] \
        && [ "$CPU_INT" -lt 20 ] && [ "$REC" = "0" ]; then
        BASELINE_DONE=1
        take_deep_capture "$PID" "$CPU_NOW" "baseline"
        LAST_SAMPLE="$EPOCH"
    fi

    if [ "$CPU_INT" -ge "$SPIKE" ] \
        && [ ! -f "$SAMPLING_FLAG" ] \
        && [ $((EPOCH - LAST_SAMPLE)) -ge "$COOLDOWN" ] \
        && [ "$SAMPLE_COUNT" -lt "$MAX_SAMPLES" ]; then
        SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
        LAST_SAMPLE="$EPOCH"
        TAG="spike"
        [ "$REC" = "1" ] && TAG="recording"
        take_deep_capture "$PID" "$CPU_NOW" "$TAG"
    fi

    # live one-line status
    printf '%s  cpu=%5s%%  mem=%sMB  thr=%s  wins=%s  ind=%s  rec=%s\n' \
        "$TS" "${CPU_NOW}" "$RSS_MB" "$THREADS" "${WINS:--}" "${INDWIN:--}" "$REC"
done

#!/usr/bin/env bash
# Pull training run(s) from Modal Volume to local training_runs/
#
# Usage:
#   ./scripts/pull_run.sh              # list runs, download latest
#   ./scripts/pull_run.sh <run_name>   # download specific run
#   ./scripts/pull_run.sh --list       # list runs only

set -euo pipefail

VOLUME_NAME="minesweeper-runs"
LOCAL_DIR="training_runs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

list_runs() {
    echo "Runs in Modal Volume '$VOLUME_NAME':"
    echo "--------------------------------------"
    modal volume ls "$VOLUME_NAME" / 2>/dev/null || {
        echo "ERROR: Failed to list volume. Make sure modal is installed and authenticated."
        exit 1
    }
}

download_run() {
    local run_name="$1"
    echo "Downloading run '$run_name' from Modal Volume '$VOLUME_NAME'..."
    mkdir -p "$LOCAL_DIR"
    modal volume get "$VOLUME_NAME" "/$run_name" "$LOCAL_DIR/$run_name" 2>/dev/null || \
    modal volume get "$VOLUME_NAME" "$run_name" "$LOCAL_DIR/" 2>/dev/null || {
        echo "ERROR: Failed to download run '$run_name'."
        echo "Try: modal volume ls $VOLUME_NAME /"
        exit 1
    }
    echo "Done. Files saved to: $LOCAL_DIR/$run_name"
}

get_latest_run() {
    # Get the most recently modified directory in the volume
    modal volume ls "$VOLUME_NAME" / 2>/dev/null | grep -v "^$" | tail -1 | awk '{print $NF}' || echo ""
}

# Parse arguments
case "${1:-}" in
    --list|-l)
        list_runs
        ;;
    "")
        # No argument: list and download latest
        list_runs
        echo ""
        latest="$(get_latest_run)"
        if [ -z "$latest" ]; then
            echo "No runs found in volume."
            exit 0
        fi
        echo "Latest run: $latest"
        read -p "Download latest run '$latest'? [Y/n] " -n 1 -r reply
        echo ""
        reply="${reply:-Y}"
        if [[ "$reply" =~ ^[Yy] ]]; then
            download_run "$latest"
        else
            echo "Aborted."
        fi
        ;;
    *)
        download_run "$1"
        ;;
esac

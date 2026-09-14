#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$PROJECT_ROOT/MDSR-PySR/pipeline.sh"

if [[ ! -f "$PIPELINE" ]]; then
    printf 'error: MDSR-PySR is not initialized. Run ./setup.sh first.\n' >&2
    exit 2
fi

if (( $# == 0 )); then
    set -- --help
fi

exec "$PIPELINE" "$@"

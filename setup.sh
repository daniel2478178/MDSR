#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$PROJECT_ROOT/MDSR-PySR/environment.yml"
ENV_NAME="mdsr-pysr"
SKIP_ENV=0

usage() {
    cat <<'EOF'
Usage: ./setup.sh [--skip-env]

Initialize the pinned Git submodules and create or update the mdsr-pysr Conda
environment. Use --skip-env to initialize only the submodules.
EOF
}

case "${1:-}" in
    "")
        ;;
    --skip-env)
        SKIP_ENV=1
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if (( $# > 1 )); then
    usage >&2
    exit 2
fi

git -C "$PROJECT_ROOT" submodule update --init --recursive

if (( SKIP_ENV )); then
    printf 'Submodules initialized. Conda environment setup skipped.\n'
    exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
    printf 'error: conda was not found. Install Conda or rerun with --skip-env.\n' >&2
    exit 2
fi

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fx "$ENV_NAME" >/dev/null; then
    conda env update --name "$ENV_NAME" --file "$ENV_FILE"
else
    conda env create --file "$ENV_FILE"
fi

printf '\nSetup complete. Next run:\n\n'
printf '  conda activate %s\n' "$ENV_NAME"
printf '  ./run.sh all --mode xonly --dry-run\n'

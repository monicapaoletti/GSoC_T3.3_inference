#!/bin/bash
# Start Claude in the development container, from the laptop (host) side.
#
#   ./container/run.sh              start Claude (builds the image the first time)
#   ./container/run.sh --build      rebuild the image first (after changing requirements.in)
#   ./container/run.sh bash         a shell in the same container instead of Claude
#   ./container/run.sh --ssh        forward the host ssh agent, so Claude can reach the
#                                   clusters from inside (see the note below)
#
# --ssh is off by default. Not because anything here needs protecting -- the SC is public
# and everything else is simulated -- but because it is the difference between a container
# that can only develop and one that can submit jobs to doslis and UGent. Turn it on when
# you want that, leave it off when you want cluster runs to stay a host-side decision.
#
# Deliberately a plain `docker run` rather than a VS Code dev container: VS Code forwards
# the ssh agent and git credentials automatically, which would make that choice for you.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
source "$HERE/config.env"

# Must match the project path Claude Code derives from WORKDIR inside the container, so the
# memory written in here is the same memory as on the host.
MEMORY_DIR="$HOME/.claude/projects/-home-monica-Desktop-GSoC-repos-GSoC-T3-3-inference/memory"

BUILD=0
SSH_FORWARD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --build) BUILD=1; shift ;;
        --ssh)   SSH_FORWARD=1; shift ;;
        *)       break ;;
    esac
done

if [ "$BUILD" = 1 ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build --target claude -t "$IMAGE" \
        --build-arg USER_UID="$(id -u)" --build-arg USER_GID="$(id -g)" \
        "$HERE"
fi

MOUNTS=(
    -v "$REPO":"$WORKDIR"
    -v gsoc-t33-claude-config:/home/dev/.claude
)
[ -d "$MEMORY_DIR" ] && MOUNTS+=( -v "$MEMORY_DIR":/home/dev/.claude/projects/-workspaces-GSoC-T3-3-inference/memory )

ENVS=()
if [ "$SSH_FORWARD" = 1 ]; then
    if [ -z "${SSH_AUTH_SOCK:-}" ]; then
        echo "--ssh: no SSH_AUTH_SOCK on the host. Start an agent first:" >&2
        echo "    eval \$(ssh-agent) && ssh-add ~/.ssh/<key>" >&2
        exit 1
    fi
    # The agent socket, not the keys: the container can use them, never read them.
    MOUNTS+=( -v "$SSH_AUTH_SOCK":/ssh-agent )
    MOUNTS+=( -v "$HOME/.ssh/config":/home/dev/.ssh/config:ro )
    ENVS+=( -e SSH_AUTH_SOCK=/ssh-agent )
fi

exec docker run --rm -it \
    --name gsoc-t33-claude \
    --hostname gsoc-t33 \
    -p 127.0.0.1:8888:8888 \
    "${MOUNTS[@]}" \
    "${ENVS[@]}" \
    -w "$WORKDIR" \
    "$IMAGE" "$@"

#!/bin/bash
# Container start. There is no firewall to raise and no synthetic-data tree to create --
# this container isolates nothing -- so this only normalises the ssh agent and then runs
# whatever command was given (default: claude).
set -euo pipefail

# run.sh --ssh forwards the host agent socket. If it wasn't passed, make sure a stale
# variable from the image doesn't make ssh hang looking for a socket that isn't there.
if [ -n "${SSH_AUTH_SOCK:-}" ] && [ ! -S "${SSH_AUTH_SOCK}" ]; then
    unset SSH_AUTH_SOCK
fi

exec "$@"

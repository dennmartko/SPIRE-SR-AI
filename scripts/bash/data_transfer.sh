#!/usr/bin/env bash
# A script to tar a directory and rsync it to a remote server without compression,
# while only prompting for your SSH password once (via connection multiplexing).

set -euo pipefail

# --- Configuration ---
# Local directory to send
SRC_DIR="/mnt/g/data/PhD Projects/SR/120deg2_shark_sides"
# Remote server login
DEST_USER="dkoopmans"
DEST_HOST="snellius.surf.nl"
# Remote destination path
DEST_PATH="/home/dkoopmans/SPIRE-SR-AI/data/processed"
# Temporary tarball path (local)
TMP_TAR="/mnt/g/data/PhD Projects/SR/$(basename "$SRC_DIR").tar.gz"

# SSH multiplexing socket (use absolute path)
CTRL_SOCKET="$HOME/.ssh/cm_${DEST_HOST}_${DEST_USER}"
# Choose a faster SSH cipher (e.g., aes128-ctr or arcfour128)
SSH_CIPHER="aes128-ctr"

# Determine compression command: pigz if available, else gzip
if command -v pigz &>/dev/null; then
  COMPRESS_CMD=(pigz -p 4 -1)
  echo "[+] Using pigz for compression"
else
  COMPRESS_CMD=(gzip -1 -c)
  echo "[!] pigz not found, falling back to gzip"
fi
# Pre-calculate basename for remote extraction
TAR_BASENAME="$(basename "$TMP_TAR")"

# rsync tuning options
RSYNC_OPTS=(
  -a          # archive mode
  -v          # verbose
  -z          # compress data during transit
  -W          # transfer whole files
  --progress
)

# Ensure ControlPath directory exists
mkdir -p "$HOME/.ssh"

# 1) Establish persistent SSH master
echo "[+] Establishing SSH master connection..."
ssh -o ControlMaster=auto \
    -o ControlPath="$CTRL_SOCKET" \
    -o ControlPersist=300s \
    -c "$SSH_CIPHER" \
    -N -f "${DEST_USER}@${DEST_HOST}"

# 2) Create compressed tarball locally if absent
if [[ -s "$TMP_TAR" ]]; then
  echo "[+] Found existing compressed tarball: $TMP_TAR"
else
  echo "[+] Creating compressed tarball: $TMP_TAR"
  mkdir -p "$(dirname "$TMP_TAR")"
  tar -cf - -C "$(dirname "$SRC_DIR")" "$(basename "$SRC_DIR")" \
    | "${COMPRESS_CMD[@]}" > "$TMP_TAR"
  echo "[+] Tarball created. Size: $(du -h "$TMP_TAR" | cut -f1)"
fi

# 3) Rsync compressed tarball to remote
echo "[+] Rsyncing compressed tarball to ${DEST_USER}@${DEST_HOST}:${DEST_PATH}"
rsync "${RSYNC_OPTS[@]}" -e \
  "ssh -o ControlMaster=auto -o ControlPath=$CTRL_SOCKET -c $SSH_CIPHER" \
  "$TMP_TAR" "${DEST_USER}@${DEST_HOST}:${DEST_PATH}/"

# 4) Extract and clean up on remote
echo "[+] Extracting on remote and cleaning up..."
ssh -o ControlMaster=auto -o ControlPath="$CTRL_SOCKET" -c "$SSH_CIPHER" \
    "${DEST_USER}@${DEST_HOST}" << EOF
set -e
cd "$DEST_PATH"
# Choose decompression on remote
if command -v pigz &>/dev/null; then
  echo "[+] Remote using pigz for decompression"
  pigz -d -c "$TAR_BASENAME" | tar -xf -
else
  echo "[!] Remote pigz not found, using gzip"
  gzip -d -c "$TAR_BASENAME" | tar -xf -
fi
# Remove remote tarball
rm -f "$TAR_BASENAME"
echo "[+] Remote extraction complete"
EOF

# 5) Cleanup local tarball
echo "[+] Removing local tarball: $TMP_TAR"
rm -f "$TMP_TAR"

# 6) Close master connection
echo "[+] Closing SSH master connection"
ssh -O exit -o ControlPath="$CTRL_SOCKET" "${DEST_USER}@${DEST_HOST}" || true

echo "All done!"

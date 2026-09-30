#!/usr/bin/env bash
# Consistent SQLite backup of the production DB via the sqlite3 backup API,
# run inside the backend container and streamed to the host.
#
# - Output: $BACKUP_DIR (default /root/backups)/testoreale-YYYYmmdd-HHMMSS.db
# - Keeps the newest $KEEP (default 4) backups, deletes older ones.
# - Writes nothing inside the repo or the ./data volume.
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/root/backups}"
KEEP="${KEEP:-4}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON_LINE="0 3 * * 0 ${REPO_DIR}/scripts/backup-db.sh >> ${BACKUP_DIR}/backup.log 2>&1"

case "$(realpath -m "$BACKUP_DIR")/" in
  "$REPO_DIR"/*) echo "error: BACKUP_DIR must be outside the repo" >&2; exit 1 ;;
esac

umask 077
mkdir -p "$BACKUP_DIR"
ts="$(date -u +%Y%m%d-%H%M%S)"
out="$BACKUP_DIR/testoreale-$ts.db"
tmp="$out.partial"
trap 'rm -f "$tmp"' EXIT

# Backup to a temp file inside the container (not the mounted volume), stream it out, remove it.
cd "$REPO_DIR"
docker compose exec -T backend python - <<'PY' > "$tmp"
import os, sqlite3, sys, tempfile
src = sqlite3.connect("file:" + os.getenv("DB_PATH", "/app/data/testoreale.db") + "?mode=ro", uri=True)
fd, path = tempfile.mkstemp(suffix=".db")
os.close(fd)
try:
    dst = sqlite3.connect(path)
    src.backup(dst)
    ok = dst.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    dst.close()
    if not ok:
        sys.exit("integrity_check failed")
    with open(path, "rb") as f:
        sys.stdout.buffer.write(f.read())
finally:
    os.remove(path)
PY

if [ ! -s "$tmp" ] || [ "$(head -c 15 "$tmp")" != "SQLite format 3" ]; then
  echo "error: backup output is empty or not a SQLite file" >&2
  exit 1
fi
mv "$tmp" "$out"
echo "backup ok: $out ($(stat -c %s "$out") bytes)"

# Rotation: keep the newest $KEEP
ls -1t "$BACKUP_DIR"/testoreale-*.db 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do
  rm -f -- "$old"
  echo "removed old backup: $old"
done

if [ -t 1 ]; then
  echo
  echo "Suggested weekly crontab entry (NOT installed; add it with 'crontab -e'):"
  echo "  $CRON_LINE"
fi

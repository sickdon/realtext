#!/usr/bin/env python3
"""One-shot migration: replace plaintext IPs in `analytics` with HMAC-SHA256(ip, IP_HASH_SALT)[:16].

Usage:
    IP_HASH_SALT=... python3 scripts/migrate_hash_ips.py /path/to/testoreale.db

- Uses the same salt and algorithm as the backend (app/db/database.py:hash_ip).
- Idempotent: values that are already hashed (16 chars, no ':' or '.') and 'na' are skipped.
- `rate_limits` is not touched.
- Prints only counts, never IPs.
- Stdlib only.
"""
import hashlib
import hmac
import os
import sqlite3
import sys

HASHED_WHERE = "(length(ip) = 16 AND instr(ip, ':') = 0 AND instr(ip, '.') = 0) OR ip = 'na'"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: IP_HASH_SALT=... migrate_hash_ips.py <db_path>", file=sys.stderr)
        return 2
    db_path = sys.argv[1]
    salt = os.environ.get("IP_HASH_SALT", "")
    if not salt:
        print("error: IP_HASH_SALT is not set", file=sys.stderr)
        return 2
    if not os.path.isfile(db_path):
        print("error: database file not found", file=sys.stderr)
        return 2

    def hash_ip(ip: str) -> str:
        return hmac.new(salt.encode(), ip.encode(), hashlib.sha256).hexdigest()[:16]

    con = sqlite3.connect(db_path)
    con.create_function("hash_ip", 1, hash_ip, deterministic=True)
    # Overwrite freed/old page content with zeros so plaintext IPs don't linger in the file.
    con.execute("PRAGMA secure_delete = ON")

    def counts() -> tuple[int, int, int]:
        rows, distinct = con.execute("SELECT COUNT(*), COUNT(DISTINCT ip) FROM analytics").fetchone()
        plain = con.execute(f"SELECT COUNT(*) FROM analytics WHERE NOT ({HASHED_WHERE})").fetchone()[0]
        return rows, distinct, plain

    rows_before, distinct_before, plain_before = counts()
    print(f"before: rows={rows_before} distinct_ip={distinct_before} plaintext_rows={plain_before}")

    with con:
        cur = con.execute(f"UPDATE analytics SET ip = hash_ip(ip) WHERE NOT ({HASHED_WHERE})")
        updated = cur.rowcount

    rows_after, distinct_after, plain_after = counts()
    print(f"updated: {updated}")
    print(f"after:  rows={rows_after} distinct_ip={distinct_after} plaintext_rows={plain_after}")

    ok = rows_after == rows_before and plain_after == 0
    if distinct_after != distinct_before:
        # Only possible with a 64-bit prefix collision, or if hashed and plaintext values mixed.
        print("warning: distinct ip count changed", file=sys.stderr)
    con.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

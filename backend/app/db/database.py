import os
import hmac
import hashlib
import logging
from datetime import datetime, timedelta

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "/app/data/testoreale.db")
IP_HASH_SALT = os.getenv("IP_HASH_SALT", "")
ANALYTICS_RETENTION_DAYS = 180


def hash_ip(ip: str) -> str:
    """Pseudonymize an IP for analytics: HMAC-SHA256 with IP_HASH_SALT, first 16 hex chars.

    Without a salt no identifier is stored at all.
    """
    if not IP_HASH_SALT:
        return "na"
    return hmac.new(IP_HASH_SALT.encode(), ip.encode(), hashlib.sha256).hexdigest()[:16]


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                ip TEXT NOT NULL,
                date TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (ip, date)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ip TEXT NOT NULL,
                detected_language TEXT NOT NULL,
                score INTEGER NOT NULL,
                text_length INTEGER NOT NULL
            )
        """)
        await db.commit()
    logger.info("Database initialized at %s", DB_PATH)
    if not IP_HASH_SALT:
        logger.warning("IP_HASH_SALT not set: analytics will store ip='na' (no identifier)")

    from app.services.rate_limiter import cleanup_old_records
    await cleanup_old_records()
    await cleanup_old_analytics()


async def cleanup_old_analytics() -> None:
    cutoff = (datetime.utcnow() - timedelta(days=ANALYTICS_RETENTION_DAYS)).isoformat()
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("DELETE FROM analytics WHERE timestamp < ?", (cutoff,))
            await db.commit()
        logger.info("Deleted %d analytics rows older than %s", cursor.rowcount, cutoff)
    except Exception as e:
        logger.error("Error cleaning up analytics: %s", e)


async def log_analysis(
    ip: str, language: str, score: int, text_length: int
) -> None:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO analytics (timestamp, ip, detected_language, score, text_length) VALUES (?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), hash_ip(ip), language, score, text_length),
            )
            await db.commit()
    except Exception as e:
        logger.error("Error logging analysis: %s", e)


async def get_stats() -> dict:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM analytics")
            row = await cursor.fetchone()
            total_analyses = row[0] if row else 0

            cursor = await db.execute("SELECT AVG(score) FROM analytics")
            row = await cursor.fetchone()
            avg_score = round(row[0], 1) if row and row[0] is not None else 0

            cursor = await db.execute(
                "SELECT detected_language, COUNT(*) FROM analytics GROUP BY detected_language ORDER BY COUNT(*) DESC"
            )
            languages = {r[0]: r[1] for r in await cursor.fetchall()}

            return {
                "total_analyses": total_analyses,
                "average_score": avg_score,
                "languages": languages,
            }
    except Exception as e:
        logger.error("Error getting stats: %s", e)
        return {"total_analyses": 0, "average_score": 0, "languages": {}}

"""Database setup for the distributed DDAR proof-search system.

Connects to the PostgreSQL server, creates the target database if it does not
already exist, and applies the SQL schema from db/schema.sql.

Typical usage example:

  python db.setup_db
"""

import os
import psycopg2
from contextlib import closing

DB_CONFIG = {
    "host": os.environ.get("DDAR_DB_HOST", "localhost"),
    "port": int(os.environ.get("DDAR_DB_PORT", 5432)),
    "dbname": os.environ.get("DDAR_DB_NAME", "alphageometry"),
    "user": os.environ.get("DDAR_DB_USER", "postgres"),
    "password": os.environ.get("DDAR_DB_PASSWORD", ""),
}


def setup_db(config: dict = DB_CONFIG) -> None:
    """Create the database and apply the schema if not already present."""
    dbname = config["dbname"]

    # Connect to the default 'postgres' database to create the target DB.
    admin_config = {**config, "dbname": "postgres"}
    with closing(psycopg2.connect(**admin_config)) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone():
                print(f"Database already exists: {dbname}")
            else:
                cur.execute(f'CREATE DATABASE "{dbname}"')
                print(f"Created database: {dbname}")

    # Apply the schema
    schema_path = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")
    with closing(psycopg2.connect(**config)) as conn:
        with open(schema_path) as f:
            schema = f.read()
        with conn.cursor() as cur:
            cur.execute(schema)
        conn.commit()
    print("Schema applied successfully.")


if __name__ == "__main__":
    setup_db()
    print("Done.")

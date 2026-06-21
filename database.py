import sqlite3

conn=sqlite3.connect(
    "resume.db"
)

cursor=conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS candidates(

name TEXT,
score INTEGER,
result TEXT

)

""")

conn.commit()
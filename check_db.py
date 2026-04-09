import sqlite3

conn = sqlite3.connect("database/users.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

cursor.execute("SELECT * FROM results")
rows1 = cursor.fetchall()

print(rows)
print(rows1)

conn.close()
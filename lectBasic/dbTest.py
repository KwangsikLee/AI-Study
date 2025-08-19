
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

db_file = 'db/friend.db'

def create_connection(db_file):
    """Create a database connection to the SQLite database specified by db_file."""
    conn = None
    try:
        
        conn = sqlite3.connect(db_file)
        print(f"Connected to database: {db_file}")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn

def create_test_table():
    conn = create_connection(db_file)
    if conn:
        try:
            cursor = conn.cursor()

            # Create a table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER,
                    grade TEXT
                )
            ''')
            print("Table 'students' created successfully.")
            
            # Insert sample data
            cursor.execute('''
                INSERT INTO students (name, age, grade)
                VALUES ('Alice', 20, 'A'), ('Bob', 21, 'B'), ('Charlie', 22, 'C')
            ''')
            print("Sample data inserted successfully.")
            
            # Commit the changes
            conn.commit()
            
        except sqlite3.Error as e:
            print(f"Error: {e}")
        
        finally:
            # Close the connection
            conn.close()
            print("Connection closed.")


conn = create_connection(db_file)
cur = conn.cursor()
cur.execute("SELECT * FROM friend_data")

friends = cur.fetchall()

# for friend in friends:
#   print(friend)

# 테이블 컬럼명 가져오기
column_name = [column[0] for column in cur.description]
# print(column_name)
frame = pd.DataFrame.from_records(data=friends, columns=column_name)

# conn.close()
#print(frame)



cur.execute("SELECT COUNT(*)" \
            ", COUNT(CASE WHEN age >= 20 THEN 1 END) as adults" \
            ", COUNT(CASE WHEN age < 20 THEN 1 END) as young" \
" FROM friend_data")

result = cur.fetchone()
column_name = [column[0] for column in cur.description]
print(column_name)

conn.close()

# 막대 그래프 데이터 준비
categories = ['total', 'adults', 'young']
counts = [result[0], result[1], result[2]]
colors = ['Green', 'Blue', 'Yellow']

# 막대 그래프 생성
plt.figure(figsize=(12, 8))

# 서브플롯 1: 기본 막대 그래프
fig, ax = plt.subplot(1, 1, 1)
bars1 = plt.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
plt.title('Age', fontsize=14, fontweight='bold')
plt.ylabel('Total Count')

# 막대 위에 값 표시
for bar, count in zip(bars1, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             str(count), ha='center', va='bottom', fontweight='bold')
    
plt.show()
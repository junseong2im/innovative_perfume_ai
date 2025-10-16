import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

conn = sqlite3.connect('fragrance_ai.db')
cursor = conn.cursor()

# 향료 목록
cursor.execute("SELECT id, name FROM fragrance_notes LIMIT 30")
sample_notes = cursor.fetchall()

print("=== 데이터베이스에 있는 향료 샘플 ===")
for note_id, name in sample_notes:
    print(f"  - {name}")

print("\n=== 테스트할 향료명 ===")
test_notes = ["Neroli", "Bergamot", "Lemon", "Jasmine", "Rose", "Vanilla",
              "Calabrian Bergamot", "Pink Pepper", "Ylang-Ylang"]

for note in test_notes:
    cursor.execute("SELECT id, name FROM fragrance_notes WHERE LOWER(name) LIKE ?", (f"%{note.lower()}%",))
    matches = cursor.fetchall()

    if matches:
        print(f"✓ '{note}' 매칭:")
        for mid, mname in matches[:3]:
            print(f"    - {mname}")
    else:
        print(f"✗ '{note}' - 매칭 없음")

conn.close()

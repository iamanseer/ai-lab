import sqlite3
from ai_lab.tools import DB_PATH, FILES_DIR

EMPLOYEES = [
    ("Asha", "Engineering", 95000, "2022-03-14"),
    ("Rahul", "Engineering", 88000, "2023-07-01"),
    ("Arjun", "Engineering", 105000, "2019-09-30"),
    ("Meera", "Sales", 62000, "2021-11-20"),
    ("Vikram", "Sales", 58000, "2024-01-08"),
    ("Fatima", "HR", 54000, "2020-05-25"),
]

def main() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, "
                 "department TEXT, salary INTEGER, joined DATE)")
    conn.executemany("INSERT INTO employees (name, department, salary, joined) "
                     "VALUES (?, ?, ?, ?)", EMPLOYEES)
    conn.commit()
    conn.close()
    (FILES_DIR / "notes.txt").write_text(
        "Q4 hiring budget: 3 engineers, 1 salesperson.\nOffsite: 14 Nov 2026 in Kochi.\n")
    print(f"Seeded {DB_PATH} and {FILES_DIR / 'notes.txt'}")

if __name__ == "__main__":
    main()
import sqlite3
from typing import List, Tuple, Optional

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    context TEXT NOT NULL,
                    answer TEXT,
                    approved INTEGER DEFAULT 0,
                    processed INTEGER DEFAULT 0
                )
            ''')
            conn.commit()

    def insert_question(self, question: str, context: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO qa_pairs (question, context) VALUES (?, ?)',
                (question, context)
            )
            conn.commit()

    def get_unanswered_questions(self) -> List[Tuple[int, str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT id, question, context FROM qa_pairs WHERE answer IS NULL'
            )
            return cursor.fetchall()

    def update_answer(self, question_id: int, answer: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'UPDATE qa_pairs SET answer = ? WHERE id = ?',
                (answer, question_id)
            )
            conn.commit()

    def mark_answer_failed(self, question_id: int):
        """Mark a question as failed (empty answer, approved=0, processed=1)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'UPDATE qa_pairs SET answer = "", approved = 0, processed = 1 WHERE id = ?',
                (question_id,)
            )
            conn.commit()

    def get_unprocessed_qa_pairs(self) -> List[Tuple[int, str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT id, question, answer FROM qa_pairs WHERE processed = 0 AND answer IS NOT NULL'
            )
            return cursor.fetchall()

    def update_approval_status(self, question_id: int, approved: bool):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'UPDATE qa_pairs SET approved = ?, processed = 1 WHERE id = ?',
                (1 if approved else 0, question_id)
            )
            conn.commit()

    def get_approved_qa_pairs(self) -> List[Tuple[str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT question, answer FROM qa_pairs WHERE approved = 1'
            )
            return cursor.fetchall()

    def mark_as_unprocessed(self, question_id: int):
        """Mark a question as unprocessed so it goes through approval again"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'UPDATE qa_pairs SET processed = 0 WHERE id = ?',
                (question_id,)
            )
            conn.commit()

    def update_question_and_answer(self, question_id: int, new_question: str, new_answer: str):
        """Update both question and answer for a given question_id"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE qa_pairs SET question = ?, answer = ? WHERE id = ?",
                (new_question, new_answer, question_id)
            )
            conn.commit()

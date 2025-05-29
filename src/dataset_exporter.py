import json
from typing import List, Tuple

class DatasetExporter:
    @staticmethod
    def export_to_jsonl(qa_pairs: List[Tuple[str, str]], output_path: str):
        with open(output_path, 'w') as f:
            for question, answer in qa_pairs:
                entry = {"question": question, "answer": answer}
                f.write(json.dumps(entry) + '\n')

        print(f"Exported {len(qa_pairs)} question-answer pairs to {output_path}")

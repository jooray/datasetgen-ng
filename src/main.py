import argparse
import os
import sys
from .config import Config
from .database import Database
from .question_generator import QuestionGenerator
from .answer_generator import AnswerGenerator
from .approval_checker import ApprovalChecker
from .dataset_exporter import DatasetExporter
from .chat_venice_api import ChatVeniceAPI
from openai import InternalServerError, RateLimitError, APITimeoutError, APIConnectionError

def main():
    parser = argparse.ArgumentParser(description="Dataset generator for finetuning")
    parser.add_argument("--dataset", help="Path to the text file (required for phases 1, addcontext, all)")
    parser.add_argument("--config", help="Path to config file",
                       default=os.path.expanduser("~/.datasetgen.json"))
    parser.add_argument("--phase", choices=['1', '2', '3', '4', 'addcontext', 'all'], default='all',
                       help="Which phase to run (1=questions, 2=answers, 3=approval, 4=export, addcontext=import to vector store)")
    parser.add_argument("--output", help="Output path for dataset (phase 4)",
                       default="dataset.jsonl")
    parser.add_argument("--db", help="Database path", default="dataset.db")
    parser.add_argument("--vector-store", help="Vector store path", default="vector_store")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging to stderr")
    parser.add_argument("--reprocess-rejected", action="store_true",
                       help="Reprocess previously rejected question-answer pairs in phase 3")
    parser.add_argument("--with-context", action="store_true",
                       help="Include context information during approval checking (phase 3)")

    args = parser.parse_args()

    # Validate dataset argument for phases that need it
    phases_needing_dataset = ['1', 'addcontext', 'all']
    if args.phase in phases_needing_dataset:
        if not args.dataset:
            print(f"Error: --dataset is required for phase {args.phase}")
            sys.exit(1)
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset file '{args.dataset}' not found")
            sys.exit(1)

    # Read text file only if needed
    text = None
    if args.dataset and os.path.exists(args.dataset):
        with open(args.dataset, 'r') as f:
            text = f.read()

    config = Config(args.config)
    db = Database(args.db)

    api_key = os.getenv('VENICE_API_KEY')
    if not api_key:
        print("Error: VENICE_API_KEY environment variable not set")
        sys.exit(1)

    # Create LLM instance once with retry configuration
    base_llm = ChatVeniceAPI(
        model=config.get_model_name(),
        api_key=api_key,
        base_url=config.get_api_base()
    )
    llm = base_llm.with_retry(
        retry_if_exception_type=(InternalServerError, RateLimitError, APITimeoutError, APIConnectionError),
        wait_exponential_jitter=True,
        stop_after_attempt=3
    )

    prompts = config.get_prompts()

    # Initialize answer generator for vector store operations (only if we have text or need vector store)
    answer_gen = None
    if text or args.phase in ['2', '3'] or (args.phase == 'all'):
        answer_gen = AnswerGenerator(
            text=text or "",  # Empty string if no text provided
            prompt=prompts['answer_generation'],
            llm=llm,
            embedding_model=config.get_embedding_model(),
            verbose=args.verbose,
            vector_store_path=args.vector_store
        )

    if args.phase in ['addcontext']:
        print("Phase addcontext: Importing text to vector store...")
        answer_gen.load_or_create_vector_store(db)
        print("Text imported to vector store")
        return

    if args.phase in ['1', 'all']:
        print("Phase 1: Generating questions...")

        # Ensure vector store is set up for RAG
        answer_gen.load_or_create_vector_store(db)

        generator = QuestionGenerator(
            text=text,
            prompt=prompts['question_generation'],
            llm=llm,
            chunk_size=config.get_question_chunk_size(),
            verbose=args.verbose
        )

        chunks = generator.split_text()
        total_chunks = len(chunks)
        total_questions = 0
        print(f"Found {total_chunks} text chunks to process")

        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{total_chunks}...")
            questions = generator.generate_questions_for_chunk(chunk)
            for question in questions:
                db.insert_question(question, chunk)
                total_questions += 1
            if args.verbose:
                print(f"[VERBOSE] Generated {len(questions)} questions from chunk {i}", file=sys.stderr)
        print(f"Generated {total_questions} questions from {total_chunks} chunks")

    if args.phase in ['2', 'all']:
        print("Phase 2: Generating answers...")

        # Ensure vector store is loaded (may not have text for phase 2 only)
        if not answer_gen.vector_store and os.path.exists(args.vector_store):
            answer_gen.load_existing_vector_store()

        unanswered = db.get_unanswered_questions()
        total_questions = len(unanswered)
        print(f"Found {total_questions} unanswered questions")

        for i, (question_id, question, context) in enumerate(unanswered, 1):
            print(f"Processing question {i}/{total_questions}...")
            answer = answer_gen.generate_answer(question, context)
            if answer is None:
                # Answer generation failed
                db.mark_answer_failed(question_id)
                if args.verbose:
                    print(f"[VERBOSE] Answer generation failed for question {question_id}", file=sys.stderr)
            else:
                db.update_answer(question_id, answer)
                if args.verbose:
                    print(f"[VERBOSE] Successfully generated answer for question {question_id}", file=sys.stderr)
        print(f"Completed answer generation for {total_questions} questions")

    if args.phase in ['3', 'all']:
        print("Phase 3: Checking approvals...")

        # Handle reprocessing of rejected pairs if requested
        if args.reprocess_rejected:
            rejected_pairs = db.get_rejected_qa_pairs()
            if rejected_pairs:
                print(f"Found {len(rejected_pairs)} rejected pairs to reprocess")
                question_ids = [pair[0] for pair in rejected_pairs]
                db.reset_for_reprocessing(question_ids)
                print(f"Reset {len(question_ids)} rejected pairs for reprocessing")
            else:
                print("No rejected pairs found to reprocess")

        # Load vector store if with-context is enabled and we need it for similarity search
        if args.with_context and answer_gen and not answer_gen.vector_store and os.path.exists(args.vector_store):
            answer_gen.load_existing_vector_store()

        checker = ApprovalChecker(
            approval_prompts=prompts['approval_prompts'],
            llm=llm,
            verbose=args.verbose,
            vector_store=answer_gen.vector_store if args.with_context and answer_gen else None
        )

        # Loop until all pairs are processed (handle CHANGE status that creates new unprocessed pairs)
        iteration = 1
        while True:
            # Get unprocessed pairs (always includes context)
            unprocessed = db.get_unprocessed_qa_pairs()

            if not unprocessed:
                break

            total_pairs = len(unprocessed)
            print(f"Approval iteration {iteration}: Found {total_pairs} unprocessed question-answer pairs")

            changes_made = 0
            for i, (question_id, question, answer, context) in enumerate(unprocessed, 1):
                print(f"Processing approval {i}/{total_pairs}...")

                if args.with_context:
                    result = checker.check_approval(question, answer, use_context=True, context=context)
                else:
                    result = checker.check_approval(question, answer, use_context=False)

                if result["status"] == "PASS":
                    db.update_approval_status(question_id, True)
                    if args.verbose:
                        print(f"[VERBOSE] Question {question_id} approved", file=sys.stderr)
                elif result["status"] == "REJECT":
                    rejection_reason = result.get('message', 'No reason provided')
                    db.update_approval_status(question_id, False, rejection_reason)
                    if args.verbose:
                        print(f"[VERBOSE] Question {question_id} rejected: {rejection_reason}", file=sys.stderr)
                elif result["status"] == "CHANGE":
                    # Handle missing keys gracefully - only update what's provided
                    new_question = result.get("question", question)  # Use original if not provided
                    new_answer = result.get("answer", answer)        # Use original if not provided

                    # Update the question and answer in database
                    db.update_question_and_answer(question_id, new_question, new_answer)
                    # Mark as unprocessed so it goes through approval again
                    db.mark_as_unprocessed(question_id)
                    changes_made += 1
                    if args.verbose:
                        print(f"[VERBOSE] Question {question_id} changed: {result.get('justification', 'No justification provided')}", file=sys.stderr)

            print(f"Completed approval iteration {iteration}: {changes_made} pairs were changed and will be re-processed")
            iteration += 1

            # Safety check to prevent infinite loops
            if iteration > 10:
                print("Warning: Reached maximum approval iterations (10). Some pairs may still need processing.")
                break

        print(f"Completed approval processing after {iteration - 1} iterations")

    if args.phase in ['4', 'all']:
        print("Phase 4: Exporting dataset...")
        approved_pairs = db.get_approved_qa_pairs()
        DatasetExporter.export_to_jsonl(approved_pairs, args.output)

if __name__ == "__main__":
    main()

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import Config
from .database import Database
from .question_generator import QuestionGenerator
from .answer_generator import AnswerGenerator
from .approval_checker import ApprovalChecker
from .dataset_exporter import DatasetExporter
from .chat_venice_api import ChatVeniceAPI
from openai import InternalServerError, RateLimitError, APITimeoutError, APIConnectionError

def process_chunk_questions(args):
    """Helper function for chunk processing in phase 1"""
    chunk, chunk_index, total_chunks, generator, db, verbose = args
    if verbose:
        print(f"[VERBOSE] Processing chunk {chunk_index}/{total_chunks}...", file=sys.stderr)

    questions = generator.generate_questions_for_chunk(chunk)
    question_count = 0
    for question in questions:
        db.insert_question(question, chunk)
        question_count += 1

    if verbose:
        print(f"[VERBOSE] Generated {len(questions)} questions from chunk {chunk_index}", file=sys.stderr)

    return question_count

def process_question_answer(args):
    """Helper function for answer generation in phase 2"""
    question_id, question, context, answer_gen, db, verbose = args
    if verbose:
        print(f"[VERBOSE] Processing question {question_id}...", file=sys.stderr)

    answer = answer_gen.generate_answer(question, context)
    if answer is None:
        db.mark_answer_failed(question_id)
        if verbose:
            print(f"[VERBOSE] Answer generation failed for question {question_id}", file=sys.stderr)
        return False
    else:
        db.update_answer(question_id, answer)
        if verbose:
            print(f"[VERBOSE] Successfully generated answer for question {question_id}", file=sys.stderr)
        return True

def process_approval_check(args):
    """Helper function for approval checking in phase 3"""
    question_id, question, answer, context, checker, db, with_context, verbose = args
    if verbose:
        print(f"[VERBOSE] Processing approval for question {question_id}...", file=sys.stderr)

    if with_context:
        result = checker.check_approval(question, answer, use_context=True, context=context)
    else:
        result = checker.check_approval(question, answer, use_context=False)

    if result["status"] == "PASS":
        db.update_approval_status(question_id, True)
        if verbose:
            print(f"[VERBOSE] Question {question_id} approved", file=sys.stderr)
        return {"status": "PASS", "question_id": question_id}
    elif result["status"] == "REJECT":
        rejection_reason = result.get('message', 'No reason provided')
        db.update_approval_status(question_id, False, rejection_reason)
        if verbose:
            print(f"[VERBOSE] Question {question_id} rejected: {rejection_reason}", file=sys.stderr)
        return {"status": "REJECT", "question_id": question_id}
    elif result["status"] == "CHANGE":
        new_question = result.get("question", question)
        new_answer = result.get("answer", answer)
        db.update_question_and_answer(question_id, new_question, new_answer)
        db.mark_as_unprocessed(question_id)
        if verbose:
            print(f"[VERBOSE] Question {question_id} changed: {result.get('justification', 'No justification provided')}", file=sys.stderr)
        return {"status": "CHANGE", "question_id": question_id}

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
    parser.add_argument("--threads", type=int, default=1,
                       help="Number of threads for parallel processing (default: 1)")

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

        # Prepare arguments for processing
        chunk_args = [
            (chunk, i + 1, total_chunks, generator, db, args.verbose)
            for i, chunk in enumerate(chunks)
        ]

        if args.threads > 1:
            print(f"Processing chunks in parallel with {args.threads} threads...")

            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                future_to_chunk = {executor.submit(process_chunk_questions, arg): arg[1] for arg in chunk_args}

                for future in as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        question_count = future.result()
                        total_questions += question_count
                        print(f"Completed chunk {chunk_index}/{total_chunks}")
                    except Exception as exc:
                        print(f"Chunk {chunk_index} generated an exception: {exc}")
        else:
            # Sequential processing using the same helper function
            for arg in chunk_args:
                chunk_index = arg[1]
                print(f"Processing chunk {chunk_index}/{total_chunks}...")
                try:
                    question_count = process_chunk_questions(arg)
                    total_questions += question_count
                except Exception as exc:
                    print(f"Chunk {chunk_index} generated an exception: {exc}")

        print(f"Generated {total_questions} questions from {total_chunks} chunks")

    if args.phase in ['2', 'all']:
        print("Phase 2: Generating answers...")

        # Ensure vector store is loaded (may not have text for phase 2 only)
        if not answer_gen.vector_store and os.path.exists(args.vector_store):
            answer_gen.load_existing_vector_store()

        unanswered = db.get_unanswered_questions()
        total_questions = len(unanswered)
        print(f"Found {total_questions} unanswered questions")

        # Prepare arguments for processing
        answer_args = [
            (question_id, question, context, answer_gen, db, args.verbose)
            for question_id, question, context in unanswered
        ]

        if args.threads > 1:
            print(f"Processing answers in parallel with {args.threads} threads...")

            completed_count = 0
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                future_to_question = {executor.submit(process_question_answer, arg): arg[0] for arg in answer_args}

                for future in as_completed(future_to_question):
                    question_id = future_to_question[future]
                    try:
                        success = future.result()
                        completed_count += 1
                        print(f"Processed question {completed_count}/{total_questions}")
                    except Exception as exc:
                        print(f"Question {question_id} generated an exception: {exc}")
        else:
            # Sequential processing using the same helper function
            for i, arg in enumerate(answer_args, 1):
                question_id = arg[0]
                print(f"Processing question {i}/{total_questions}...")
                try:
                    process_question_answer(arg)
                except Exception as exc:
                    print(f"Question {question_id} generated an exception: {exc}")

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
            unprocessed = db.get_unprocessed_qa_pairs()

            if not unprocessed:
                break

            total_pairs = len(unprocessed)
            print(f"Approval iteration {iteration}: Found {total_pairs} unprocessed question-answer pairs")

            # Prepare arguments for processing
            approval_args = [
                (question_id, question, answer, context, checker, db, args.with_context, args.verbose)
                for question_id, question, answer, context in unprocessed
            ]

            changes_made = 0
            if args.threads > 1:
                print(f"Processing approvals in parallel with {args.threads} threads...")

                completed_count = 0
                with ThreadPoolExecutor(max_workers=args.threads) as executor:
                    future_to_question = {executor.submit(process_approval_check, arg): arg[0] for arg in approval_args}

                    for future in as_completed(future_to_question):
                        question_id = future_to_question[future]
                        try:
                            result = future.result()
                            if result["status"] == "CHANGE":
                                changes_made += 1
                            completed_count += 1
                            print(f"Processed approval {completed_count}/{total_pairs}")
                        except Exception as exc:
                            print(f"Approval for question {question_id} generated an exception: {exc}")
            else:
                # Sequential processing using the same helper function
                for i, arg in enumerate(approval_args, 1):
                    question_id = arg[0]
                    print(f"Processing approval {i}/{total_pairs}...")
                    try:
                        result = process_approval_check(arg)
                        if result["status"] == "CHANGE":
                            changes_made += 1
                    except Exception as exc:
                        print(f"Approval for question {question_id} generated an exception: {exc}")

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

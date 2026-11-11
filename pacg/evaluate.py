"""Local exact-match evaluation and optional Base fixed-response reference generation."""
import argparse
import json
from collections import defaultdict

import numpy as np

from .reference import messages, read_rows


def summarize(rows):
    questions = defaultdict(list)
    for row in rows:
        questions[(str(row["benchmark"]), str(row["question_id"]))].append(row["accuracy"])
    benchmarks = defaultdict(list)
    for (name, _), values in questions.items():
        benchmarks[name].append(float(np.mean(values)))
    scores = {name: 100 * float(np.mean(values)) for name, values in sorted(benchmarks.items())}
    return {"accuracy_percent": scores, "unweighted_benchmark_mean": float(np.mean(list(scores.values())))
            if scores else None, "questions": len(questions), "responses": len(rows),
            "metric": "mean response accuracy -> question macro -> unweighted benchmark mean; not pass@k"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="JSONL: sample_id, question_id, benchmark, question, image, answer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.n < 1 or args.batch_size < 1:
        parser.error("n and batch-size must be positive")
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from verl.utils.dataset import process_image
    from mathruler.grader import extract_boxed_content, grade_answer
    rows = read_rows(args.input)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    engine = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
                 trust_remote_code=True, max_model_len=6144, gpu_memory_utilization=.8,
                 limit_mm_per_prompt={"image": 1}, seed=args.seed)
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=1.0,
                            max_tokens=args.max_tokens, seed=args.seed)
    results = []
    with open(args.output, "x") as dest:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [{"prompt": processor.apply_chat_template(messages(row["question"]),
                         tokenize=False, add_generation_prompt=True),
                        "multi_modal_data": {"image": process_image(row["image"], 200704, 1003520)}} for row in batch]
            outputs = engine.generate(prompts, params)
            for row, output in zip(batch, outputs):
                for i, sample in enumerate(output.outputs):
                    result = {**row, "sample_id": f'{row["sample_id"]}:{i}', "response": sample.text,
                              "accuracy": float(grade_answer(extract_boxed_content(sample.text), row["answer"])),
                              "generation": {"model": args.model, "seed": args.seed, "n": args.n,
                                             "temperature": args.temperature, "max_tokens": args.max_tokens}}
                    results.append(result)
                    dest.write(json.dumps(result, ensure_ascii=False) + "\n")
                dest.flush()
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()


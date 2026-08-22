"""Evaluate a base Mistral LM or MLPMemory on WebQA and TriviaQA."""

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence

import regex
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from models import MLPMemory, MistralMLPModel


TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, choices=("webqa", "triviaqa"))
    parser.add_argument("--data-path", required=True, help="Path to the task test.jsonl")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--mode", choices=("base", "mlpmemory"), default="base")
    parser.add_argument("--knn-generator-path")
    parser.add_argument("--lmbda", type=float, default=0.75)
    parser.add_argument("--knn-temp", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--max-eval-samples", type=int, help="Optional smoke-test limit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(TORCH_DTYPES), default="bfloat16")
    args = parser.parse_args()

    if args.mode == "mlpmemory" and not args.knn_generator_path:
        parser.error("--knn-generator-path is required when --mode=mlpmemory")
    if not 0.0 <= args.lmbda <= 1.0:
        parser.error("--lmbda must be in [0, 1]")
    if args.knn_temp <= 0:
        parser.error("--knn-temp must be positive")
    if args.eval_batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("--eval-batch-size and --max-new-tokens must be positive")
    if args.max_eval_samples is not None and args.max_eval_samples <= 0:
        parser.error("--max-eval-samples must be positive")
    return args


def load_jsonl(path: str) -> List[Dict[str, object]]:
    samples = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if "question" not in sample or "answer" not in sample:
                raise ValueError(f"Missing question/answer at {path}:{line_number}")
            if not isinstance(sample["answer"], list):
                sample["answer"] = [sample["answer"]]
            samples.append(sample)
    return samples


def build_prompt(question: str) -> str:
    # This is the exact Mistral prompt used by the verified neuralKNN runs.
    return f"Answer the questions:\n\nQuestion: {question}? The answer is:"


class MultiTokenEOSCriteria(StoppingCriteria):
    """Stop after every item in the batch has generated the same stop string."""

    def __init__(self, sequence: str, tokenizer, initial_input_length: int, batch_size: int):
        self.sequence = sequence
        self.tokenizer = tokenizer
        self.initial_input_length = initial_input_length
        self.lookback_length = len(tokenizer.encode(sequence, add_special_tokens=False)) + 2
        self.done = [False] * batch_size

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[:, self.initial_input_length :]
        generated_ids = generated_ids[:, -self.lookback_length :]
        generated_text = self.tokenizer.batch_decode(generated_ids)
        for index, is_done in enumerate(self.done):
            if not is_done:
                self.done[index] = self.sequence in generated_text[index]
        return all(self.done)


def stop_criteria(tokenizer, input_length: int, batch_size: int) -> StoppingCriteriaList:
    return StoppingCriteriaList(
        [
            MultiTokenEOSCriteria(sequence, tokenizer, input_length, batch_size)
            for sequence in ("\n", ".", ",")
        ]
    )


class SimpleTokenizer:
    alpha_numeric = r"[\p{L}\p{N}\p{M}]+"
    non_whitespace = r"[^\p{Z}\p{C}]"

    def __init__(self):
        self.pattern = regex.compile(
            f"({self.alpha_numeric})|({self.non_whitespace})",
            flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE,
        )

    def tokenize(self, text: str, uncased: bool = False) -> List[str]:
        tokens = [match.group() for match in self.pattern.finditer(text)]
        return [token.lower() for token in tokens] if uncased else tokens


ANSWER_TOKENIZER = SimpleTokenizer()


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def has_answer(answers: Sequence[str], text: str) -> bool:
    text_tokens = ANSWER_TOKENIZER.tokenize(normalize_unicode(text), uncased=True)
    for answer in answers:
        answer_tokens = ANSWER_TOKENIZER.tokenize(normalize_unicode(str(answer)), uncased=True)
        for start in range(len(text_tokens) - len(answer_tokens) + 1):
            if answer_tokens == text_tokens[start : start + len(answer_tokens)]:
                return True
    return False


def load_model(args: argparse.Namespace):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    device = torch.device(args.device)
    dtype = TORCH_DTYPES[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        add_eos_token=False,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no pad, unk, or eos token")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.eval()

    if args.mode == "base":
        return base_model, tokenizer, device

    if base_model.config.model_type != "mistral":
        raise ValueError("MLPMemory evaluation currently supports a Mistral base model only")
    config = AutoConfig.from_pretrained(args.knn_generator_path)
    knn_generator = MistralMLPModel.from_pretrained(
        args.knn_generator_path,
        config=config,
        input_dim=config.hidden_size,
        output_dim=config.hidden_size,
        torch_dtype=dtype,
    ).to(device)
    knn_generator.eval()
    model = MLPMemory(
        base_lm=base_model,
        knn_generator=knn_generator,
        lmbda=args.lmbda,
        knn_temp=args.knn_temp,
    ).eval()
    return model, tokenizer, device


@torch.no_grad()
def generate_answers(
    model,
    tokenizer,
    device: torch.device,
    prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    answers = []
    batches = range(0, len(prompts), batch_size)
    batch_count = (len(prompts) + batch_size - 1) // batch_size
    for start in tqdm(batches, desc="Generating", total=batch_count):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        input_length = input_ids.shape[1]
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stop_criteria(tokenizer, input_length, len(batch_prompts)),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        decoded = tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=False)
        answers.extend(text.strip() for text in decoded)
    return answers


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data_path)
    if args.max_eval_samples is not None:
        samples = samples[: args.max_eval_samples]
    if not samples:
        raise ValueError(f"No samples found in {args.data_path}")

    model, tokenizer, device = load_model(args)
    prompts = [build_prompt(str(sample["question"])) for sample in samples]
    prompt_lengths = tokenizer(prompts, return_length=True).length
    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
    generated_answers = generate_answers(
        model,
        tokenizer,
        device,
        prompts,
        args.eval_batch_size,
        args.max_new_tokens,
    )

    sample_scores = [
        float(has_answer(sample["answer"], generated_answer))
        for sample, generated_answer in zip(samples, generated_answers)
    ]
    substring_match = round(sum(sample_scores) / len(sample_scores), 4)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "first_prompt.txt").write_text(prompts[0], encoding="utf-8")
    with (results_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sample, prediction, score in zip(samples, generated_answers, sample_scores):
            row = {
                "id": sample.get("id"),
                "question": sample["question"],
                "answers": sample["answer"],
                "prediction": prediction,
                "substring_match": score,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "dataset": args.data,
        "batch_size": args.eval_batch_size,
        "include_retrieval": False,
        "avg_prompt_length": avg_prompt_length,
        "model": args.model_name_or_path,
        "substring_match": substring_match,
    }
    with (results_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4, ensure_ascii=False)

    print(json.dumps(result, indent=4, ensure_ascii=False))
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()

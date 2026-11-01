"""Generate DCC labels and retain every raw LLM response before validation.

Apply the appendix annotation prompt and validate event fields and relationships
against the input captions and timestamps.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from schema import normalize_input, validate_annotation, write_json, require_new_paths

MODEL_ID = 'meta-llama/Llama-3.3-70B-Instruct'
PROMPT_VERSION = 'dcc-appendix-reference-v1'
SYSTEM_PROMPT = 'You are an expert in analyzing video narratives and identifying cause-effect relationships in sequences of events.'
SCHEMA = '{\n    "duration": float,\n    "events": [\n        {\n            "event_id": int,\n            "timestamp": [start, end],\n            "description": "text",\n            "chain": {\n                "is_cause": boolean,\n                "is_effect": boolean,\n                "cause_event": [event_ids],\n                "effect_event": [event_ids]\n            },\n            "event_type": "cause,effect,causal,independent"\n        }\n    ]\n}'


def build_prompt(video_data):
    """
    Constructs the user-turn prompt.

    Follows the LLM Annotation Generation Prompt in Appendix C of the paper
    verbatim, substituting {input_json} and {N} with the actual video data.
    """
    video_data = normalize_input(video_data)
    input_json = json.dumps(video_data, indent=2)
    n_events = len(video_data.get("timestamps", []))
    return (
        "Convert the following video caption data into a structured narrative "
        "format that shows cause-effect relationships between events.\n"
        "Input video caption data:\n"
        f"{input_json}\n\n"
        "Generate a single JSON object as output with exactly this structure:\n"
        f"{SCHEMA}\n\n"
        "Guidelines for analysis:\n"
        "0. Count total number of events in input first\n"
        "1. Label each event as \"independent\", \"cause\", \"effect\" or \"causal\":\n"
        "   - \"cause\": Directly leads to or influences other events\n"
        "   - \"effect\": Results from or is influenced by other events\n"
        "   - \"causal\": Both causes and effects (two-way relationships)\n"
        "   - \"independent\": No causal relationships with other events\n"
        "2. Look for causal indicators\n"
        "3. Track relationships using event IDs\n"
        "4. Enforce temporal constraints:\n"
        "   - cause_event can ONLY reference prior event IDs\n"
        "   - effect_event can ONLY reference later event IDs\n"
        f"5. Do not add more than {n_events} events\n"
        "6. Output ONLY the JSON object, no additional text\n"
        "Provide only the JSON object output."
    )


def parse_response(text):
    start = text.find('{')
    if start < 0:
        raise ValueError('Response contains no JSON object')
    parsed, _ = json.JSONDecoder().raw_decode(text, start)
    return parsed


def process_response(text, source):
    parsed = parse_response(text)
    return validate_annotation(parsed, source)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default=MODEL_ID, help='HF ID or local checkpoint directory')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--device-map', default='auto', help='auto shards one model across visible GPUs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-videos', type=int, default=0)
    p.add_argument('--dry-run', action='store_true', help='Write prompts only, without loading any model')
    args = p.parse_args()
    if args.batch_size < 1 or args.max_videos < 0:
        p.error('Batch size must be positive and max-videos nonnegative')
    data = json.loads(Path(args.input).read_text())
    data = {vid: normalize_input(record) for vid, record in data.items()}
    items = list(data.items())[:args.max_videos or None]
    if not items:
        raise ValueError('No videos to annotate')
    output = Path(args.output)
    raw_path = output.with_suffix('.raw.jsonl')
    require_new_paths(output, raw_path, output.with_suffix('.manifest.json'),
                      output.with_suffix('.partial.json'))
    if args.dry_run:
        write_json(output, {'dry_run': True, 'prompt_version': PROMPT_VERSION,
                            'prompts': {vid: build_prompt(record) for vid, record in items}})
        return
    import torch
    from transformers import pipeline, set_seed
    set_seed(args.seed)
    pipe = pipeline('text-generation', model=args.model, device_map=args.device_map,
                    torch_dtype=torch.bfloat16, model_kwargs={'low_cpu_mem_usage': True})
    pipe.tokenizer.padding_side = 'left'
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    output.parent.mkdir(parents=True, exist_ok=True)
    annotations, failures = {}, {}
    with raw_path.open('x') as raw:
        for offset in range(0, len(items), args.batch_size):
            batch = items[offset:offset+args.batch_size]
            prompts = [build_prompt(record) for _, record in batch]
            messages = [[{'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': prompt}] for prompt in prompts]
            responses = pipe(messages, max_new_tokens=4096, do_sample=True, temperature=0.1,
                             return_full_text=False, batch_size=args.batch_size)
            if len(responses) != len(batch):
                raise RuntimeError('Model response count differs from input batch')
            for (vid, source), prompt, response in zip(batch, prompts, responses):
                text = response[0]['generated_text']
                entry = {'video_id': vid, 'model': args.model, 'seed': args.seed,
                         'prompt_version': PROMPT_VERSION, 'prompt': prompt, 'system_prompt': SYSTEM_PROMPT,
                         'raw_response': text}
                raw.write(json.dumps(entry, allow_nan=False) + '\n')
                raw.flush()
                try:
                    annotations[vid] = process_response(text, source)
                    entry['status'] = 'valid'
                except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
                    failures[vid] = str(exc)
                    entry.update(status='invalid', error=str(exc))
            print(f'Processed {min(offset+args.batch_size, len(items))}/{len(items)}; invalid={len(failures)}', flush=True)
    write_json(output.with_suffix('.manifest.json'), {
        'model': args.model, 'seed': args.seed, 'prompt_version': PROMPT_VERSION,
        'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'raw_response_file': raw_path.name, 'videos': len(items), 'valid_videos': len(annotations),
        'invalid_videos': failures})
    if failures:
        write_json(output.with_suffix('.partial.json'), annotations)
        raise RuntimeError(f'{len(failures)} invalid annotations; final output was not created. Inspect raw records.')
    write_json(output, annotations)


if __name__ == '__main__':
    main()

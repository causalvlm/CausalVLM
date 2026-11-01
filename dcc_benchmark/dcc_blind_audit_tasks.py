"""Prepare full-video annotation tasks with blank labels for every forward pair.
"""
import argparse
import csv
import json
import random
from pathlib import Path
from schema import normalize_input, write_json


def sample_tasks(inputs, dataset, sample_size, seed):
    if sample_size < 1 or sample_size > len(inputs):
        raise ValueError('Requested audit sample must fit the available videos')
    normalized = {vid: normalize_input(record) for vid, record in inputs.items()}
    strata = {}
    for vid, record in normalized.items():
        strata.setdefault(len(record['timestamps']), []).append(vid)
    # Proportional event-count strata, using largest remainders to meet N exactly.
    rng = random.Random(seed)
    allocations = {n: int(sample_size*len(ids)/len(inputs)) for n, ids in strata.items()}
    remainder = sample_size-sum(allocations.values())
    order = sorted(strata, key=lambda n: (-(sample_size*len(strata[n])/len(inputs)-allocations[n]), n))
    for n in order[:remainder]:
        allocations[n] += 1
    selected = []
    for n in sorted(strata):
        selected.extend(rng.sample(sorted(strata[n]), allocations[n]))
    rng.shuffle(selected)
    videos, pairs = [], []
    for vid in selected:
        record = normalized[vid]
        events = [{'event_id': i+1, 'timestamp': span, 'description': caption}
                  for i, (span, caption) in enumerate(zip(record['timestamps'], record['sentences']))]
        videos.append({'dataset': dataset, 'video_id': vid, 'duration': record['duration'], 'events': events})
        for i in range(len(events)):
            for j in range(i+1, len(events)):
                pairs.append({'pair_id': f'{dataset}:{vid}:{i+1}:{j+1}', 'dataset': dataset,
                              'video_id': vid, 'source_event_id': i+1, 'target_event_id': j+1})
    return videos, pairs


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--activitynet', required=True, help='Human caption input from prepare_inputs.py, not LLM annotations')
    p.add_argument('--youcook2', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--videos-per-dataset', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    videos, pairs = [], []
    for name, filename in [('activitynet', a.activitynet), ('youcook2', a.youcook2)]:
        vs, ps = sample_tasks(json.loads(Path(filename).read_text()), name, a.videos_per_dataset, a.seed)
        videos.extend(vs)
        pairs.extend(ps)
    write_json(out/'videos.json', videos)
    write_json(out/'sampling.json', {'seed': a.seed, 'videos': len(videos), 'eligible_pairs': len(pairs),
                                   'allocation': 'proportional event-count strata; largest remainders'})
    with (out/'annotation_template.csv').open('w', newline='') as handle:
        fields = ['pair_id', 'dataset', 'video_id', 'source_event_id', 'target_event_id', 'edge_label']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    (out/'INSTRUCTIONS.txt').write_text(
        'Watch the full original video and label every listed pair as 0 (no edge) or 1 (edge).\n'
        'An edge represents an observed prerequisite dependency: the earlier event enables the later event in this video.\n'
        'Temporal succession alone is insufficient. Do not add transitive closure. Overlapping events follow onset order.\n'
        'Do not consult LLM labels, scores, or explanations. Give each annotator a separate copy of the template.\n')

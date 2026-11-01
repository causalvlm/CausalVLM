"""Apply completed human pair ratings to DCC labels."""
import argparse
import copy
import csv
import json
from pathlib import Path
from schema import validate_annotations, write_json


def apply_ratings(data, tasks, ratings, dataset, threshold):
    validate_annotations(data)
    if not 1 <= threshold <= 5:
        raise ValueError('Retention threshold must lie within the 1–5 rating scale')
    decisions = {}
    for task in tasks:
        if task['dataset'] != dataset:
            continue
        values = ratings.get(task['pair_id'], [])
        if len(values) != 3 or any(type(v) is not int or not 1 <= v <= 5 for v in values):
            raise ValueError(f"{task['pair_id']}: complete all three ratings before filtering")
        if task.get('clip_ok') is not True:
            raise ValueError(f"{task['pair_id']}: the annotation clip was not created successfully")
        key = (task['video_id'], task['cause_event_id'], task['effect_event_id'])
        if key in decisions:
            raise ValueError('Duplicate human task for the same event pair')
        decisions[key] = sum(values)/3 >= threshold
    if not decisions:
        raise ValueError(f'No completed tasks for dataset {dataset}')
    selected_videos = {key[0] for key in decisions}
    expected = {(vid, e['event_id'], dst) for vid in selected_videos
                for e in data.get(vid, {}).get('events', []) for dst in e['chain']['effect_event']}
    if expected != decisions.keys():
        raise ValueError('Tasks must cover every proposed pair in each selected video')
    result = copy.deepcopy(data)
    seen = set()
    for vid, record in result.items():
        if not isinstance(record, dict):
            raise ValueError(f'{vid}: invalid input annotation')
        events = record['events']
        original = {(event['event_id'], dst) for event in events for dst in event['chain']['effect_event']}
        kept = set()
        for src, dst in original:
            key = (vid, src, dst)
            if key in decisions:
                seen.add(key)
            if decisions.get(key, True):
                kept.add((src, dst))
        for event in events:
            i = event['event_id']
            causes = sorted(src for src, dst in kept if dst == i)
            effects = sorted(dst for src, dst in kept if src == i)
            event['chain'] = {'is_cause': bool(effects), 'is_effect': bool(causes),
                              'cause_event': causes, 'effect_event': effects}
            event['event_type'] = 'causal' if causes and effects else 'cause' if effects else 'effect' if causes else 'independent'
    if seen != decisions.keys():
        raise ValueError('Human tasks reference pairs absent from the input annotations')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--tasks', required=True)
    p.add_argument('--ratings', required=True, help='Completed CSV with pair_id,rater_1,rater_2,rater_3')
    p.add_argument('--dataset', required=True, choices=['activitynet', 'youcook2'])
    p.add_argument('--threshold', required=True, type=float, help='Explicit experiment retention cutoff, from 1 to 5')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    ratings = {}
    tasks = json.loads(Path(a.tasks).read_text())
    selected_ids = {task['pair_id'] for task in tasks if task['dataset'] == a.dataset}
    with open(a.ratings, newline='') as handle:
        for row in csv.DictReader(handle):
            if row['pair_id'] not in selected_ids:
                continue
            if row['pair_id'] in ratings:
                raise ValueError('Duplicate pair ID in rating CSV')
            ratings[row['pair_id']] = [int(row[f'rater_{i}']) for i in (1, 2, 3)]
    write_json(a.output, apply_ratings(json.loads(Path(a.input).read_text()),
               tasks, ratings, a.dataset, a.threshold))

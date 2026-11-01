"""Validate DCC event fields, intervals, and reciprocal causal references."""
from __future__ import annotations
import json
import math
import os
import tempfile
from pathlib import Path


def normalize_input(record):
    duration = float(record['duration'])
    timestamps, captions = record['timestamps'], record['sentences']
    if not timestamps or not math.isfinite(duration) or duration <= 0 or len(timestamps) != len(captions):
        raise ValueError('Invalid duration or event/caption count')
    events = []
    for span, caption in zip(timestamps, captions):
        if len(span) != 2 or not isinstance(caption, str) or not caption.strip():
            raise ValueError('Events need a two-element interval and nonempty caption')
        start, end = map(float, span)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError('Event intervals must lie within the video duration')
        events.append(([start, end], caption))
    # Preserve input order for events with equal onsets.
    events.sort(key=lambda item: item[0][0])
    return {'duration': duration, 'timestamps': [e[0] for e in events],
            'sentences': [e[1] for e in events]}


def validate_annotation(annotation, source):
    source = normalize_input(source)
    if not isinstance(annotation, dict) or not isinstance(annotation.get('events'), list):
        raise ValueError('Annotation must contain an events list')
    events, n = annotation['events'], len(source['timestamps'])
    if len(events) != n:
        raise ValueError(f'Expected exactly {n} events, received {len(events)}')
    if not math.isclose(float(annotation['duration']), source['duration'], abs_tol=1e-6):
        raise ValueError('Generated duration differs from input')
    if any(type(e.get('event_id')) is not int for e in events):
        raise ValueError('Event IDs must be integers')
    if [e['event_id'] for e in events] != list(range(1, n+1)):
        raise ValueError('Event IDs must be unique and numbered in onset order')
    outgoing, incoming = set(), set()
    for i, event in enumerate(events, 1):
        span = event['timestamp']
        if len(span) != 2 or any(not math.isclose(float(a), b, abs_tol=1e-6)
                                 for a, b in zip(span, source['timestamps'][i-1])):
            raise ValueError('Generated timestamps differ from input')
        if event['description'] != source['sentences'][i-1]:
            raise ValueError('Generated description differs from the human caption')
        chain = event['chain']
        for field in ('cause_event', 'effect_event'):
            values = chain[field]
            if not isinstance(values, list) or any(type(v) is not int for v in values):
                raise ValueError('Edge references must be lists of integer event IDs')
            if len(set(values)) != len(values) or any(v < 1 or v > n for v in values):
                raise ValueError('Duplicate or unknown edge endpoint')
        if any(j >= i for j in chain['cause_event']) or any(j <= i for j in chain['effect_event']):
            raise ValueError('Self/backward edges are not eligible')
        outgoing.update((i, j) for j in chain['effect_event'])
        incoming.update((j, i) for j in chain['cause_event'])
        cause, effect = bool(chain['effect_event']), bool(chain['cause_event'])
        if type(chain['is_cause']) is not bool or type(chain['is_effect']) is not bool:
            raise ValueError('Role flags must be JSON booleans')
        role = 'causal' if cause and effect else 'cause' if cause else 'effect' if effect else 'independent'
        if chain['is_cause'] != cause or chain['is_effect'] != effect or event['event_type'] != role:
            raise ValueError('Event role disagrees with its edges')
    if outgoing != incoming:
        raise ValueError('Incoming and outgoing edge lists must agree')
    return annotation


def validate_annotations(data):
    """Validate every input record and report the affected video on failure."""
    if not isinstance(data, dict) or not data:
        raise ValueError('Expected a nonempty mapping of video IDs to annotations')
    for vid, record in data.items():
        try:
            if not isinstance(vid, str) or not vid or any(c in vid for c in '/\\\n\r'):
                raise ValueError('Video ID must be a nonempty filename component')
            events = record['events']
            validate_annotation(record, {'duration': record['duration'],
                'timestamps': [e['timestamp'] for e in events],
                'sentences': [e['description'] for e in events]})
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
            raise ValueError(f'{vid}: invalid DCC annotation: {exc}') from exc
    return data


def normalize_scores(scores):
    """Use the four-score arithmetic mean specified in the appendix."""
    fields = ('plausibility', 'visual_evidence', 'strength', 'transition')
    if not isinstance(scores, dict):
        raise ValueError('Scores must be a JSON object')
    for key in (*fields, 'overall'):
        value = scores.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{key}: expected a finite numeric score in [0, 1]')
    result = dict(scores)
    result['model_reported_overall'] = scores.get('model_reported_overall', scores['overall'])
    result['overall'] = sum(scores[key] for key in fields) / len(fields)
    return result


def require_new_paths(*paths):
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Output and report paths must be distinct')
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f'Choose a new output path: {path}')


def write_text(path, text):
    """Publish a complete file atomically without replacing any existing file."""
    path = Path(path)
    require_new_paths(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')

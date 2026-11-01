"""Convert released per-event DCC JSONL into validated video-level annotations."""
import argparse
import json
import math
from pathlib import Path

from schema import validate_annotations, write_json


FIELDS = {
    'video_id', 'duration', 'event_id', 'start', 'end', 'description',
    'event_type', 'is_cause', 'is_effect', 'cause_event', 'effect_event',
}


def read_labels(path):
    data = {}
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != FIELDS:
                    raise ValueError('Expected the released DCC JSONL fields')
                vid, duration = row['video_id'], row['duration']
                if not isinstance(vid, str) or not vid:
                    raise ValueError('video_id must be a nonempty string')
                if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
                    raise ValueError('duration must be a finite positive number')
                if type(row['event_id']) is not int:
                    raise ValueError('event_id must be an integer')
                for field in ('start', 'end'):
                    if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
                        raise ValueError(f'{field} must be a finite number')
                record = data.setdefault(vid, {'duration': duration, 'events': []})
                if record['duration'] != duration:
                    raise ValueError('Rows for one video must have the same duration')
                record['events'].append({
                    'event_id': row['event_id'],
                    'timestamp': [row['start'], row['end']],
                    'description': row['description'],
                    'event_type': row['event_type'],
                    'chain': {key: row[key] for key in
                              ('is_cause', 'is_effect', 'cause_event', 'effect_event')},
                })
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
    for record in data.values():
        record['events'].sort(key=lambda event: event['event_id'])
    return validate_annotations(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Released per-event JSONL split')
    parser.add_argument('--output', required=True, help='New video-level annotation JSON')
    args = parser.parse_args()
    write_json(args.output, read_labels(args.input))


if __name__ == '__main__':
    main()

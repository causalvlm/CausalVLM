"""Export validated nested DCC labels to the released per-event JSONL format."""
import argparse
import json
from pathlib import Path
from schema import validate_annotations, write_text


def flatten(data):
    validate_annotations(data)
    return [{'video_id': vid, 'duration': record['duration'], 'event_id': e['event_id'],
             'start': e['timestamp'][0], 'end': e['timestamp'][1],
             'description': e['description'], 'event_type': e['event_type'],
             'is_cause': e['chain']['is_cause'], 'is_effect': e['chain']['is_effect'],
             'cause_event': e['chain']['cause_event'], 'effect_event': e['chain']['effect_event']}
            for vid, record in data.items() for e in record['events']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Final nested annotation JSON')
    parser.add_argument('--output', required=True, help='New per-event JSONL file')
    args = parser.parse_args()
    rows = flatten(json.loads(Path(args.input).read_text()))
    write_text(args.output, ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))

"""Convert ActivityNet caption splits or a YouCook2 database to generation input."""
import argparse
import json
from pathlib import Path
from schema import normalize_input, write_json


def convert(data, split=None):
    records = data.get('database', data)
    result = {}
    for video_id, record in records.items():
        if split and record.get('subset') != split:
            continue
        if 'annotations' in record:
            record = {'duration': record['duration'],
                      'timestamps': [a['segment'] for a in record['annotations']],
                      'sentences': [a['sentence'] for a in record['annotations']]}
        result[video_id] = normalize_input(record)
    if not result:
        raise ValueError('No videos selected; check the input and split name')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', help='Exact YouCook2 subset name, e.g. training or validation')
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    write_json(a.output, convert(json.loads(Path(a.input).read_text()), a.split))

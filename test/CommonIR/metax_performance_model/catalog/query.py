"""Read the source-to-LLVM catalog; conditions are descriptive text."""
from pathlib import Path
import re
import yaml

FIELDS = {'mapping_id', 'source', 'conditions', 'llvm', 'source_reference'}


def load_catalog(path=None):
    data = yaml.safe_load(Path(path or Path(__file__).with_name('catalog.yaml')).read_text())
    if not isinstance(data, dict) or set(data) != {'flagtree_commit', 'mappings'
                                                   } or not re.fullmatch('[0-9a-f]{40}', str(data['flagtree_commit'])):
        raise ValueError('catalog requires one commit and mappings')
    if not isinstance(data['mappings'], list): raise ValueError('mappings must be a list')
    ids = set()
    for row in data['mappings']:
        if not isinstance(row, dict) or set(row) != FIELDS or not all(row[k] for k in FIELDS):
            raise ValueError('incomplete mapping record')
        if not isinstance(row['mapping_id'], str) or not isinstance(row['llvm'], str):
            raise ValueError('mapping ID and LLVM must be text')
        if row['mapping_id'] in ids:
            raise ValueError('duplicate mapping: ' + row['mapping_id'])
        ids.add(row['mapping_id'])
        if set(row['source']) != {'apis', 'ir_ops'} or not row['source']['ir_ops']:
            raise ValueError('mapping requires source operations')
        for values in (row['conditions'], row['source']['apis'], row['source']['ir_ops']):
            if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
                raise ValueError('conditions and source names must be text lists')
        for reference in row['source_reference']:
            path = Path(reference['path'])
            if path.is_absolute() or '..' in path.parts or not (reference.get('symbol') or reference.get('line')):
                raise ValueError('source reference requires a relative path and locator')
    return data


def query_catalog(catalog, source=None, llvm=None, text=None):
    """Match a source name exactly; LLVM and free text use substring lookup."""
    return [
        r for r in catalog['mappings']
        if (source is None or source in r['source']['apis'] + r['source']['ir_ops']) and (
            llvm is None or llvm in r['llvm']) and (text is None or text.casefold() in str(r).casefold())
    ]


def table(rows):
    lines = ['| mapping_id | source op/API | conditions | LLVM | source reference |', '|---|---|---|---|---|']
    for r in rows:
        cells = [
            r['mapping_id'], '; '.join(r['source']['apis'] + r['source']['ir_ops']), '; '.join(r['conditions']),
            r['llvm'],
            '; '.join(x['path'] + ':' + str(x.get('symbol', x.get('line', ''))) for x in r['source_reference'])
        ]
        lines.append('| ' + ' | '.join(c.replace('|', '\\|').replace('\n', '<br>') for c in cells) + ' |')
    return '\n'.join(lines)

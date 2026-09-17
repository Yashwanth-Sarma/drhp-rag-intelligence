"""Read OKF v0.2 as untrusted knowledge, never as executable instructions.

Unknown metadata is retained. Verification declarations are not authentication.
This reader implements the concept/lifecycle subset, not computation execution.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import hashlib
import re
import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError('OKF metadata keys must be unique strings.')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _json_metadata(value, depth=0):
    if depth > 20:
        raise ValueError('OKF metadata nesting exceeds the application limit.')
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k:_json_metadata(v, depth+1) for k,v in value.items()}
    if isinstance(value, list):
        return [_json_metadata(v, depth+1) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError('FinSight metadata must use JSON-compatible YAML values.')


def _instant(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('OKF timestamps require an explicit UTC offset.')
    return parsed


def read_concept(path, root, now=None):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or path.suffix != '.md':
        raise ValueError('Concept must be a markdown file within the bundle.')
    if path.name in {'index.md', 'log.md'}:
        raise ValueError('Reserved OKF file is not a concept.')
    if path.stat().st_size > 131072:
        raise ValueError('Concept exceeds the application size limit.')
    raw = path.read_text(encoding='utf-8-sig')
    parts = re.split(r'^---\s*$', raw, maxsplit=2, flags=re.MULTILINE)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError('Missing OKF frontmatter.')
    # Aliases can expand recursive structures; FinSight accepts a restricted YAML profile.
    if any(isinstance(t, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)) for t in yaml.scan(parts[1])):
        raise ValueError('YAML aliases and anchors are disabled in FinSight.')
    meta = yaml.load(parts[1], Loader=UniqueLoader)
    if not isinstance(meta, dict) or not isinstance(meta.get('type'), str) or not meta['type'].strip():
        raise ValueError('OKF requires a nonempty type.')
    events = meta.get('verified', [])
    if isinstance(events, dict):
        events = [events]
    if not isinstance(events, list):
        raise ValueError('Invalid verification events.')
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get('by'), str) or not event['by']:
            raise ValueError('Invalid verification actor.')
        _instant(event.get('at'))
    generated = meta.get('generated')
    if generated is not None:
        if not isinstance(generated, dict) or not isinstance(generated.get('by'), str):
            raise ValueError('Invalid generated metadata.')
        if 'at' in generated:
            _instant(generated['at'])
    sources = meta.get('sources', [])
    if not isinstance(sources, list) or any(not isinstance(s, dict) or not isinstance(s.get('resource'), str) or not s['resource'] for s in sources):
        raise ValueError('Each OKF source requires a resource.')
    source_ids = [s['id'] for s in sources if 'id' in s]
    if any(not isinstance(s, str) for s in source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError('Source IDs must be unique strings.')
    cited = set(re.findall(r'\[\^([^\]]+)\]', parts[2]))
    missing = sorted(cited - set(source_ids))
    tier = 'human-reviewed' if any(e['by'].startswith('human:') for e in events) else 'machine-confirmed' if events else 'unverified'
    now = now or datetime.now(timezone.utc)
    stale = 'stale_after' in meta and now >= _instant(meta['stale_after'])
    status = meta.get('status', 'stable')
    eligible = status == 'stable' and not stale and not missing
    return dict(id=path.relative_to(root).with_suffix('').as_posix(), metadata=_json_metadata(meta),
                body=parts[2].strip(), sha256=hashlib.sha256(raw.encode()).hexdigest(),
                declared_trust=tier, verification_authenticated=False, stale=stale,
                eligible=eligible, missing_source_ids=missing, role='analytical_context_not_issuer_evidence')


def load_bundle(root):
    root = Path(root)
    concepts, errors = [], []
    paths = sorted(root.rglob('*.md'))
    if len(paths) > 2000:
        raise ValueError('Bundle exceeds the application file limit.')
    for path in paths:
        if path.name in {'index.md', 'log.md'}:
            continue
        try:
            concepts.append(read_concept(path, root))
        except (ValueError, OSError, yaml.YAMLError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    return dict(concepts=concepts, errors=errors)


def retrieve_context(root, question, limit=4):
    bundle = load_bundle(root)
    tokens = set(re.findall(r'\w+', question.lower()))
    scored = []
    for concept in bundle['concepts']:
        if not concept['eligible']:
            continue
        text = str(concept['metadata']) + ' ' + concept['body']
        overlap = len(tokens & set(re.findall(r'\w+', text.lower())))
        if overlap:
            scored.append((overlap, concept))
    scored.sort(key=lambda pair: (-pair[0], pair[1]['id']))
    return dict(concepts=[c for _, c in scored[:limit]], errors=bundle['errors'])

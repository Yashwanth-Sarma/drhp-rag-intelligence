"""Bounded neighbouring source blocks, never inferred table/header relationships."""
import json
import math


def _position(block):
    try:
        geometry = json.loads(block['bbox'])
        rect = geometry['rect']
        if len(rect) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in rect):
            return None
        return rect[1], rect[0], block['id']
    except (TypeError, ValueError, KeyError):
        return None


def source_context(store, evidence_id, radius=2, max_chars=12000):
    if not 0 <= radius <= 5 or not 0 <= max_chars <= 50000:
        raise ValueError('Context expansion exceeds the supported bounds.')
    anchor = store.evidence(evidence_id)
    if not anchor:
        raise ValueError('Evidence does not exist.')
    result = dict(anchor_id=evidence_id, document_id=anchor['document_id'], page=anchor['page'],
                  neighbors=[], omitted_ids=[], relation='same_page_spatial_neighbors',
                  limitation='Spatial neighbours are not verified headings, table headers or supporting evidence for the anchor. Each block requires its own citation.')
    if anchor['status']=='legacy_unverified' or _position(anchor) is None:
        return dict(result, status='geometry_unavailable')
    rows = store.rows('SELECT id,bbox FROM evidence WHERE document_id=? AND page=?',
                      (anchor['document_id'], anchor['page']))
    positioned = [(position, row['id']) for row in rows if (position := _position(row)) is not None]
    ordered = [eid for _,eid in sorted(positioned)]
    index = ordered.index(evidence_id)
    # Nearest neighbours get the character budget first; display in page order.
    candidates = [(abs(i-index), i, eid) for i,eid in enumerate(ordered)
                  if eid!=evidence_id and abs(i-index)<=radius]
    selected, used = [], 0
    for _, page_order, eid in sorted(candidates):
        block = store.evidence(eid)
        if used+len(block['text']) > max_chars:
            result['omitted_ids'].append(eid)
            continue
        used += len(block['text'])
        selected.append((page_order, block))
    result['neighbors'] = [block for _,block in sorted(selected)]
    result['status'] = 'partial' if result['omitted_ids'] else 'available'
    result['characters'] = used
    return result

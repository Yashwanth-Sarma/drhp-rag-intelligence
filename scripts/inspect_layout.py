"""Record reproducible parser coverage and known failures without copying filings."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.runtime'), str(ROOT)]
from finsight.store import Store
from finsight.structure import build_layout

PAGES = {
    'a5b8cce713093f98ee7819e0a9fe26db': [32, 33, 70, 71, 72],
    'f8d05e110527a3b14987ae5a7fd2a11e': [39, 59, 80, 81, 82],
    'e1886aea64996c056c36a85c38ceb662': [33, 34, 35, 36],
}

if __name__ == '__main__':
    store = Store(ROOT / 'data')
    pages = []
    for did, numbers in PAGES.items():
        for p in build_layout(store, did, numbers):
            tables = [dict(id=t['id'], bbox=t['bbox'], rows=len(t['cells']),
                           columns=max(map(len, t['cells']), default=0),
                           evidence_ids=t['evidence_ids'],
                           cells_sha256=hashlib.sha256(json.dumps(t['cells']).encode()).hexdigest())
                      for t in p['tables']]
            pages.append(dict(document_id=did, document_sha256=p['document_sha256'],
                page=p['page'], parser_version=p['parser_version'], version=p['version'],
                native_lines=len(p['lines']), heading_candidates=len(p['headings']), tables=tables))
    result = dict(pages=pages, visual_review=dict(
        scope='Agent inspection of rendered Ola p32, Swiggy p81, Hyundai p36 only; not expert annotation.',
        observations=[
            'Ola p32 has a risk heading, a loss/cash-flow table and a period header. Native lines preserve these; financial meaning is unvalidated.',
            'Swiggy p81 has annual/interim columns, currency scale, parentheses and continuation to p82. Cross-page relationships remain unlinked.',
            'Hyundai p36 has one visually continuous litigation table. Ruled-line detection returns four fragments and splits header text into rows. Retain this failure; do not reconstruct financial observations from these candidates.']),
        limitation='Fourteen selected pages, one native parser, three visually inspected pages. No full-corpus structure coverage, parser comparison, or complete Stage 02 acceptance.')
    path = ROOT / 'evals/results/source-layout-v1.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(pages=len(pages), tables=sum(len(p['tables']) for p in pages), output=str(path))))

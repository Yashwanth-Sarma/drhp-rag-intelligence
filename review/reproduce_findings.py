"""Run isolated legacy pure functions; no provider calls or legacy imports."""
import ast
import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

@dataclasses.dataclass
class Document:
    page_content: str
    metadata: dict

def load_nodes(relative, names):
    source = (ROOT / 'legacy-review' / relative).read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    namespace = dict(re=re, Document=Document, Optional=Optional,
                     dataclass=dataclasses.dataclass, field=dataclasses.field,
                     logger=logging.getLogger('audit'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), namespace)
    return namespace

clean = load_nodes('src/parsers/data_loader.py', {'clean_text'})['clean_text']
evidence = load_nodes('src/evidence/evidence_assembler.py', {
    'EvidenceItem', '_categorize_chunk', '_compute_chunk_confidence',
    '_find_supporting_chunks', '_detect_contradictions'})
sample = Document('Company B revenue was 100 in FY2023.', {
    'company_name': 'Company B', 'year': '2023', 'doc_type': 'DRHP', 'chunk_id': 'b'})
wrong_claim = 'Company A revenue was 999 in FY2025.'
support = evidence['_find_supporting_chunks'](wrong_claim, [sample])
first = Document('Revenue was 100.', {'company_name':'A', 'year':'2022', 'doc_type':'DRHP', 'chunk_id':'a'})
second = Document('Revenue was 200.', {'company_name':'A', 'year':'2023', 'doc_type':'Annual_Report', 'chunk_id':'c'})
results = {
    'scope': 'Isolated pure-function reproductions, not end-to-end application tests',
    'numeric_cleaning': {'input':'Revenue\n1000\n0\n(5)\nNote text', 'output':clean('Revenue\n1000\n0\n(5)\nNote text')},
    'wrong_company_wrong_number_support': {'claim':wrong_claim, 'source':sample.page_content,
        'support_count':len(support), 'display_confidence':support[0].confidence if support else None},
    'different_years_flagged_as_conflict': evidence['_detect_contradictions']([first, second]),
}
assert results['numeric_cleaning']['output'] == 'Revenue\nNote text'
assert results['wrong_company_wrong_number_support']['support_count'] == 1
assert len(results['different_years_flagged_as_conflict']) == 1
output = ROOT / 'review' / 'reproduced-findings.json'
output.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))

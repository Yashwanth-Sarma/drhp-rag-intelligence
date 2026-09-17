"""Deterministic evidence boundary for a future Qwen synthesis stage.

These checks establish citation integrity, not semantic entailment or truth.
Only verbatim extracts can receive 'validated_extract' status automatically.
"""
from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    evidence_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    quote: str = Field(min_length=1, max_length=12000)


class DraftClaim(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1, max_length=12000)
    citations: list[Citation] = Field(min_length=1, max_length=8)


def check_claim(store, claim, allowed_evidence_ids, company, document_id=None, as_of=None):
    """Retrieve originals again; never trust source text echoed by the model."""
    claim = DraftClaim.model_validate(claim)
    errors, sources = [], []
    for citation in claim.citations:
        source = store.evidence(citation.evidence_id)
        if citation.evidence_id not in allowed_evidence_ids or source is None:
            errors.append('citation_not_in_retrieved_context')
            continue
        sources.append(source)
        if source['company'] != company or (document_id and source['document_id'] != document_id):
            errors.append('issuer_or_document_mismatch')
        if as_of and (not source['filing_date'] or source['filing_date'] > str(as_of)):
            errors.append('outside_date_scope')
        if source['status'] == 'legacy_unverified' or not store.document(source['document_id'])['sha256']:
            errors.append('unverified_source')
        if citation.quote not in source['text']:
            errors.append('quote_not_in_source')
    exact_extract = len(claim.citations) == 1 and claim.text == claim.citations[0].quote
    return dict(status='rejected' if errors else 'validated_extract' if exact_extract else 'needs_semantic_review',
                errors=sorted(set(errors)), claim=claim.model_dump(),
                limitation='Citation integrity only. Verbatim disclosures may contain issuer assertions; paraphrases require semantic review.')

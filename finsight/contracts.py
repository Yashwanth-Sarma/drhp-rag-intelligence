"""Versioned contracts. Source availability is never a correctness score."""
import re
from datetime import date
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

Identifier = Annotated[str, Field(pattern=r'^[a-f0-9]{32}$')]
DocumentKind = Literal['DRHP', 'RHP', 'Annual report', 'Call transcript', 'Other filing']
EXTRACTION_VERSION = 'native-blocks-v2'

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

class FilingMetadata(Contract):
    company: str = Field(min_length=2, max_length=120)
    name: str = Field(min_length=1, max_length=200)
    kind: DocumentKind = 'DRHP'
    filing_date: date | None = None

    @field_validator('company', 'name')
    @classmethod
    def no_controls(cls, value):
        if re.search(r'[\x00-\x1f\x7f]', value):
            raise ValueError('Control characters are not permitted.')
        return value

class Query(Contract):
    question: str = Field(min_length=2, max_length=1500)
    company: str | None = Field(default=None, min_length=2, max_length=120)
    document_id: Identifier | None = None
    as_of: date | None = None

class ReportRequest(Contract):
    company: str = Field(min_length=2, max_length=120)
    document_id: Identifier | None = None
    as_of: date | None = None

class Comparison(Contract):
    first: Identifier
    second: Identifier
    mode: Literal['peer', 'growth'] = 'peer'

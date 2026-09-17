import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.runtime'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import pymupdf
from fastapi.testclient import TestClient
from finsight.api import create_app
from finsight.store import Store

@pytest.fixture
def store(tmp_path): return Store(tmp_path/'store')

@pytest.fixture
def pdf_bytes():
    with pymupdf.open() as pdf:
        p=pdf.new_page()
        p.insert_text((72,72),'Revenue from operations\nFY2024\n1000\n0\n(5)\nINR million')
        p.insert_text((72,200),'Risk factors: dependency on suppliers may affect operations.')
        pdf.new_page()
        pdf.set_page_labels([{'startpage':0,'prefix':'','style':'r','firstpagenum':1}])
        return pdf.tobytes()

@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path/'api'),headers={'X-FinSight-Client':'local-ui'}) as c:
        yield c

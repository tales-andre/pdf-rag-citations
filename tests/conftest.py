import pytest

from pdfrag.config import ROOT, Settings
from pdfrag.service import build_service


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(docs_dir=ROOT / "sample_docs")


@pytest.fixture(scope="session")
def service(settings):
    return build_service(settings)


@pytest.fixture(scope="session")
def index(service):
    return service.pipeline.index

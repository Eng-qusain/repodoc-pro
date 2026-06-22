import asyncio
from datetime import datetime
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, mock_open

from core.domain.entities.entities import ExportJob, ExportMode, FileCategory, FileInfo, Language
from core.services.export_orchestrator import ExportOrchestratorService


class DummyStats:
    def __init__(self):
        self.total_files = 2
        self.total_directories = 1
        self.total_lines = 100
        self.total_size_bytes = 1024
        self.language_distribution = {"Python": 1.0}
        self.extension_distribution = {".py": 2}
        self.largest_files = []
        self.average_file_size = 512
        self.average_line_count = 50


class DummyScan:
    def __init__(self, files):
        self.project_name = "test_project"
        self.flat_files = files
        self.stats = DummyStats()


@pytest.fixture
def mock_dependencies(mocker):
    return {
        "scanner": mocker.MagicMock(),
        "code_parser": mocker.MagicMock(),
        "csv_parser": mocker.MagicMock(),
        "excel_parser": mocker.MagicMock(),
        "image_parser": mocker.MagicMock(),
        "ai_documenter": mocker.MagicMock(),
    }


@pytest.fixture
def orchestrator(mock_dependencies):
    return ExportOrchestratorService(
        scanner=mock_dependencies["scanner"],
        code_parser=mock_dependencies["code_parser"],
        csv_parser=mock_dependencies["csv_parser"],
        excel_parser=mock_dependencies["excel_parser"],
        image_parser=mock_dependencies["image_parser"],
        ai_documenter=mock_dependencies["ai_documenter"],
    )


@pytest.mark.asyncio
async def test_run_export_exception_handling(orchestrator, mock_dependencies):
    """Targets lines 144-148: Exception block in _run_export pipeline."""
    mock_dependencies["scanner"].scan = AsyncMock(side_effect=RuntimeError("Scan failed"))
    
    progress_calls = []
    def cb(job_id, progress, message):
        progress_calls.append((progress, message))

    job_id = await orchestrator.start_export(
        project_path="/tmp/fake",
        output_path="/tmp/out.pdf",
        mode=ExportMode.SINGLE,
        options={},
        progress_callback=cb
    )
    
    # Wait for the background task to execution error block
    await asyncio.sleep(0.1)
    
    job = orchestrator.get_job(job_id)
    assert job.status == "failed"
    assert "Scan failed" in job.error
    assert progress_calls[-1][0] == -1


@pytest.mark.asyncio
async def test_generate_ai_summaries_exception(orchestrator, mock_dependencies):
    """Targets lines 174-175: Exception block inside _generate_ai_summaries."""
    orchestrator._ai_documenter.is_available = True
    orchestrator._ai_documenter.document_file = AsyncMock(side_effect=Exception("AI API Error"))

    file_info = FileInfo(
        name="main.py",
        path=Path("/tmp/main.py"),
        relative_path="main.py",
        category=FileCategory.SOURCE,
        is_binary=False,
        size_bytes=10,
        line_count=2,
        last_modified=datetime.utcnow(),
        language=Language.PYTHON,
        extension=".py"
    )

    # Mock file read behavior
    mock_dependencies["scanner"].scan = AsyncMock(return_value=DummyScan([file_info]))
    
    # Mocking build engine to prevent disk writes
    orchestrator._build_pdfs = AsyncMock(return_value=["/tmp/out.pdf"])

    with pytest.monkey_patch().context() as m:
        m.setattr("pathlib.Path.read_text", lambda self, errors=None: "print('hello')")
        job_id = await orchestrator.start_export(
            project_path="/tmp/fake",
            output_path="/tmp/out.pdf",
            mode=ExportMode.SINGLE,
            options={"include_ai": True}
        )
        await asyncio.sleep(0.1)

    job = orchestrator.get_job(job_id)
    assert job.status == "completed"  # Pipeline should survive AI failure gracefully


@pytest.mark.asyncio
async def test_build_alternative_modes(orchestrator, mock_dependencies, tmp_path):
    """Targets lines 196-221: _build_folder_pdfs, _build_per_file_pdfs, _build_package."""
    file1 = FileInfo(
        name="auth.py",
        path=tmp_path / "auth.py",
        relative_path="src/auth.py",
        category=FileCategory.SOURCE,
        is_binary=False,
        size_bytes=10,
        line_count=2,
        last_modified=datetime.utcnow(),
        language=Language.PYTHON,
        extension=".py"
    )
    
    scan = DummyScan([file1])
    mock_dependencies["scanner"].scan = AsyncMock(return_value=scan)
    
    # Mocking basic PDFBuilder methods inside orchestrator factory
    builder_mock = MagicMock()
    builder_mock.build.return_value = str(tmp_path / "out.pdf")
    orchestrator._create_builder = MagicMock(return_value=builder_mock)

    # Test Folder mode (Mode B)
    job_id_folder = await orchestrator.start_export(
        project_path=str(tmp_path),
        output_path=str(tmp_path / "output_dir"),
        mode=ExportMode.FOLDER,
        options={"mode": "folder"}
    )
    await asyncio.sleep(0.1)
    assert orchestrator.get_job(job_id_folder).status == "completed"

    # Test File mode (Mode C)
    job_id_file = await orchestrator.start_export(
        project_path=str(tmp_path),
        output_path=str(tmp_path / "output_dir"),
        mode=ExportMode.FILE,
        options={"mode": "file"}
    )
    await asyncio.sleep(0.1)
    assert orchestrator.get_job(job_id_file).status == "completed"

    # Test Package mode (Mode D)
    job_id_package = await orchestrator.start_export(
        project_path=str(tmp_path),
        output_path=str(tmp_path / "output_dir"),
        mode=ExportMode.PACKAGE,
        options={"mode": "package"}
    )
    await asyncio.sleep(0.1)
    assert orchestrator.get_job(job_id_package).status == "completed"


@pytest.mark.asyncio
async def test_add_file_to_builder_extensions(orchestrator, mock_dependencies, tmp_path):
    """Targets lines 331-348, 357-371: CSV parser integrations, configurations, and errors inside _add_file_to_builder."""
    
    # Setup mocks for parsers
    mock_dependencies["csv_parser"].parse.return_value = {
        "headers": ["id", "name"],
        "rows": [["1", "test"]],
        "stats": {"rows": 1}
    }

    builder_mock = MagicMock()
    
    # 1. Test CSV Processing Paths
    csv_file = FileInfo(
        name="data.csv",
        path=tmp_path / "data.csv",
        relative_path="data.csv",
        category=FileCategory.DATA,
        is_binary=False,
        size_bytes=100,
        line_count=2,
        last_modified=datetime.utcnow(),
        language=None,
        extension=".csv"
    )
    await orchestrator._add_file_to_builder(builder_mock, csv_file, {}, {"max_csv_rows": 10})
    builder_mock.add_csv_preview.assert_called_once()

    # 2. Test Image Processing Paths
    img_file = FileInfo(
        name="logo.png",
        path=tmp_path / "logo.png",
        relative_path="logo.png",
        category=FileCategory.DATA,
        is_binary=True,
        size_bytes=5000,
        line_count=0,
        last_modified=datetime.utcnow(),
        language=None,
        extension=".png"
    )
    await orchestrator._add_file_to_builder(builder_mock, img_file, {}, {})
    builder_mock.add_image.assert_called_once()

    # 3. Test Configuration File Paths
    config_file = FileInfo(
        name="config.yaml",
        path=tmp_path / "config.yaml",
        relative_path="config.yaml",
        category=FileCategory.CONFIG,
        is_binary=False,
        size_bytes=50,
        line_count=5,
        last_modified=datetime.utcnow(),
        language=None,
        extension=".yaml"
    )
    
    with pytest.monkey_patch().context() as m:
        m.setattr("pathlib.Path.read_text", lambda self, errors=None: "setting: true")
        await orchestrator._add_file_to_builder(builder_mock, config_file, {}, {})
        assert builder_mock.add_source_file.call_count == 1

    # 4. Target Exception Logger Path (lines 370-371)
    bad_file = FileInfo(
        name="broken.py",
        path=tmp_path / "broken.py",
        relative_path="broken.py",
        category=FileCategory.SOURCE,
        is_binary=False,
        size_bytes=10,
        line_count=1,
        last_modified=datetime.utcnow(),
        language=Language.PYTHON,
        extension=".py"
    )
    # Raising an unhandled execution path inside the file read sequence to hit the catch block
    with pytest.monkey_patch().context() as m:
        m.setattr("pathlib.Path.read_text", MagicMock(side_effect=IOError("Permission Denied")))
        await orchestrator._add_file_to_builder(builder_mock, bad_file, {}, {})
        # Should catch gracefully and log warning instead of crashing
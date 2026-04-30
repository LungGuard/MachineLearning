import logging
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from common.constants import ApiConstants,DownloadLimits
from common.constants.status_codes import StatusCode
from common.dto import PipelineResults
from middleware.eureka_config import deregister_eureka, register_eureka
from middleware.gateway_auth import GatewayAuthMiddleware
from middleware.middleware_config import EurekaConfig
from pipeline import MainPipeline

logger = logging.getLogger(__name__)

download_limits= DownloadLimits()
eureka_config = EurekaConfig()

DOWNLOAD_TIMEOUT = httpx.Timeout(connect=download_limits.connect,
                                 read=download_limits.read,
                                 write=download_limits.write, 
                                 pool=download_limits.pool)


# Lazy pipeline singleton. The actual model loading is intentionally not wired
# here — plug the detection / regression / classification model constructors
# into _build_pipeline() once the loading code lives somewhere importable.
_pipeline: Optional[MainPipeline] = None


def _build_pipeline() -> MainPipeline:
    raise NotImplementedError(
        "MainPipeline model wiring is not set up. Construct MainPipeline with "
        "concrete detection_model / regression_model / classification_model "
        "instances here."
    )


def get_pipeline() -> MainPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    register_eureka(config=eureka_config)
    yield
    deregister_eureka()


app = FastAPI(
    title=ApiConstants.SERVICE_NAME,
    lifespan=lifespan,
)
app.add_middleware(GatewayAuthMiddleware)


class AnalyzeRequest(BaseModel):
    scan_id: str
    presigned_url: str


@app.post("/api/models/analyze", response_model=PipelineResults)
def analyze(
    req: AnalyzeRequest,
    pipeline: MainPipeline = Depends(get_pipeline),
) -> PipelineResults:
    with tempfile.TemporaryDirectory(prefix=f"scan_{req.scan_id}_") as workdir:
        work_path = Path(workdir)
        zip_path = work_path / "scan.zip"
        extract_dir = work_path / "extracted"
        extract_dir.mkdir()

        _download(req.presigned_url, zip_path)
        _safe_extract(zip_path, extract_dir)
        dicom_dir = _locate_dicom_dir(extract_dir)

        try:
            return pipeline(dicom_dir)
        except NotImplementedError:
            raise
        except Exception as e:
            logger.exception("Pipeline failure for scan %s", req.scan_id)
            raise HTTPException(status_code=StatusCode.INTERNAL_SERVER_ERROR
                                , detail=f"pipeline failure: {e}")


def _download(url: str, dest: Path) -> None:
    try:
        with httpx.stream(method="GET", url=url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as response:
            if response.status_code >= 400:
                raise HTTPException(
                    status_code=StatusCode.BAD_GATEWAY,
                    detail=f"failed to download scan: status {response.status_code}",
                )
            with dest.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
    except httpx.HTTPError as e:
        logger.warning("Download failed: %s", e)
        raise HTTPException(status_code=StatusCode.BAD_GATEWAY, detail=f"failed to download scan: {e}")



def _safe_extract(zip_path: Path, dest: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            # Hard ceiling on uncompressed ZIP contents. A presigned PUT does not constrain
            # size on the client side, so we enforce it here as a zip-bomb guard.
            total_uncompressed = sum(m.file_size for m in archive.infolist())
            if total_uncompressed > ApiConstants.MAX_UNCOMPRESSED_BYTES:
                raise HTTPException(
                    status_code=StatusCode.UNPROCESSABLE_CONTENT,
                    detail=f"scan archive exceeds size limit: {total_uncompressed} bytes",
                )

            dest_root = dest.resolve()
            for member in archive.infolist():
                # Zip-slip guard: refuse entries that resolve outside dest.
                target = (dest / member.filename).resolve()
                if not str(target).startswith(str(dest_root)):
                    raise HTTPException(
                        status_code=StatusCode.UNPROCESSABLE_CONTENT,
                        detail=f"unsafe zip entry: {member.filename}",
                    )
            archive.extractall(dest)
    except zipfile.BadZipFile as e:
        raise HTTPException(status_code=StatusCode.UNPROCESSABLE_CONTENT, detail=f"invalid zip archive: {e}")


def _locate_dicom_dir(root: Path) -> Path:
    # Scans are often archived under nested folders like PatientX/Series1/.
    # Return the deepest directory that directly contains .dcm files.
    candidates = sorted(
        {p.parent for p in root.rglob("*.dcm")},
        key=lambda p: len(p.parts),
        reverse=True,
    )
    if not candidates:
        raise HTTPException(
            status_code=StatusCode.UNPROCESSABLE_CONTENT,
            detail="archive does not contain any .dcm files",
        )
    return candidates[0]

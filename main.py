import functools
import logging
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch as _torch

_orig_torch_load = _torch.load


@functools.wraps(_orig_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


_torch.load = _patched_torch_load

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from pydantic import HttpUrl

from ClassificationModel.constants.constants.dataset import DatasetConstants
from ClassificationModel.constants.enums.cancer_type import CancerType
from ClassificationModel.src.models.classification_model import CancerClassificationModel
from DetectionModel.src.models.detection_model import NodulesDetectionModel
from DetectionModel.src.models.efficientnet_nodule_model import EfficientNetNoduleModel
from common.constants import (
    ApiConstants,
    DownloadLimits,
    ErrorSlugs,
    PipelineMessages,
    ScanConstants,
)
from common.constants.status_codes import StatusCode
from common.dto import PipelineResults, error_response
from common.logging_config import configure_logging
from middleware.actuator import build_actuator_router
from middleware.eureka_config import deregister_eureka, register_eureka
from middleware.gateway_auth import GatewayAuthMiddleware
from middleware.middleware_config import EurekaConfig
from paths import ProjectPaths
from pipeline import MainPipeline

configure_logging()
logger = logging.getLogger(__name__)

download_limits = DownloadLimits()
eureka_config = EurekaConfig()

DOWNLOAD_TIMEOUT = httpx.Timeout(
    connect=download_limits.connect,
    read=download_limits.read,
    write=download_limits.write,
    pool=download_limits.pool,
)

_pipeline: Optional[MainPipeline] = None

_CLASSIFICATION_CLASS_NAMES: list[str] = sorted(t.value for t in CancerType)


def _build_pipeline() -> MainPipeline:
    detection_model = NodulesDetectionModel(
        pretrained_weights=str(ProjectPaths.DETECTION_BEST_WEIGHTS),
    )
    detection_model.eval()

    regression_model = EfficientNetNoduleModel.load_from_checkpoint(
        ProjectPaths.REGRESSION_BEST_CHECKPOINT,
        map_location="cpu",
        strict=False,
    )
    regression_model.eval()

    classification_model = CancerClassificationModel(
        dataset={
            DatasetConstants.NUM_CLASSES_KEY: len(_CLASSIFICATION_CLASS_NAMES),
            DatasetConstants.CLASS_NAMES_KEY: _CLASSIFICATION_CLASS_NAMES,
        },
        input_shape=(*DatasetConstants.IMAGE_SIZE, DatasetConstants.CHANNELS),
        checkpoint_path=ProjectPaths.CLASSIFICATION_BEST_CHECKPOINT,
    )

    return MainPipeline(
        detection_model=detection_model,
        regression_model=regression_model,
        classification_model=classification_model,
    )


def get_pipeline() -> MainPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    await register_eureka(config=eureka_config)
    yield
    await deregister_eureka()


app = FastAPI(
    title=ApiConstants.SERVICE_NAME,
    lifespan=lifespan,
)
app.add_middleware(GatewayAuthMiddleware)
app.include_router(build_actuator_router(get_pipeline))


_STATUS_SLUGS: dict[int, str] = {
    StatusCode.BAD_REQUEST: "bad_request",
    StatusCode.UNAUTHORIZED: "unauthorized",
    StatusCode.FORBIDDEN: "forbidden",
    StatusCode.NOT_FOUND: "not_found",
    StatusCode.UNPROCESSABLE_CONTENT: ErrorSlugs.VALIDATION,
    StatusCode.INTERNAL_SERVER_ERROR: ErrorSlugs.INTERNAL,
    StatusCode.BAD_GATEWAY: "bad_gateway",
    StatusCode.SERVICE_UNAVAILABLE: ErrorSlugs.SERVICE_UNAVAILABLE,
}


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request, exc):
    msgs = "; ".join(
        f"{'.'.join(str(l) for l in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )
    return error_response(StatusCode.UNPROCESSABLE_CONTENT, ErrorSlugs.VALIDATION, msgs)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request, exc):
    slug = _STATUS_SLUGS.get(exc.status_code, ErrorSlugs.DEFAULT)
    return error_response(exc.status_code, slug, exc.detail)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):
    logger.exception("Unhandled error on %s", request.url.path)
    return error_response(
        StatusCode.INTERNAL_SERVER_ERROR,
        ErrorSlugs.INTERNAL,
        "internal server error",
    )


class AnalyzeRequest(BaseModel):
    scan_id: str = Field(
        ...,
        min_length=ScanConstants.ID_MIN_LENGTH,
        max_length=ScanConstants.ID_MAX_LENGTH,
        pattern=ScanConstants.ID_PATTERN,
    )
    presigned_url: HttpUrl


@app.post("/api/models/analyze", response_model=PipelineResults)
def analyze(req: AnalyzeRequest) -> PipelineResults:
    try:
        pipeline = get_pipeline()
    except NotImplementedError:
        raise HTTPException(
            status_code=StatusCode.SERVICE_UNAVAILABLE,
            detail=PipelineMessages.UNAVAILABLE,
        )

    with tempfile.TemporaryDirectory(
        prefix=ScanConstants.DIR_PREFIX_TEMPLATE.format(scan_id=req.scan_id)
    ) as workdir:
        work_path = Path(workdir)
        zip_path = work_path / ScanConstants.ZIP_NAME
        extract_dir = work_path / ScanConstants.EXTRACT_DIR_NAME
        extract_dir.mkdir()

        _download(str(req.presigned_url), zip_path)
        _safe_extract(zip_path, extract_dir)
        dicom_dir = _locate_dicom_dir(extract_dir)

        try:
            return pipeline(dicom_dir)
        except Exception:
            logger.exception("Pipeline failure for scan %s", req.scan_id)
            raise HTTPException(
                status_code=StatusCode.INTERNAL_SERVER_ERROR,
                detail=PipelineMessages.FAILURE_TEMPLATE.format(scan_id=req.scan_id),
            )


def _download(url: str, dest: Path) -> None:
    try:
        with httpx.stream("GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as response:
            if response.status_code >= 400:
                raise HTTPException(
                    status_code=StatusCode.BAD_GATEWAY,
                    detail=f"failed to download scan: status {response.status_code}",
                )
            with dest.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
    except httpx.HTTPError as exc:
        logger.warning("Download failed: %s", exc)
        raise HTTPException(
            status_code=StatusCode.BAD_GATEWAY,
            detail=f"failed to download scan: {exc}",
        )


def _safe_extract(zip_path: Path, dest: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            # Zip-bomb guard: presigned URLs don't constrain uncompressed size.
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
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=StatusCode.UNPROCESSABLE_CONTENT,
            detail=f"invalid zip archive: {exc}",
        )


def _locate_dicom_dir(root: Path) -> Path:
    if candidates := sorted(
        {p.parent for p in root.rglob(ScanConstants.DICOM_GLOB)},
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        return candidates[0]
    else:
        raise HTTPException(
            status_code=StatusCode.UNPROCESSABLE_CONTENT,
            detail="archive does not contain any .dcm files",
        )

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from api.custom_components.custom_pipe import CustomPipeline
from api.config import PIPELINE_YAML_PATH, FILE_UPLOAD_PATH, INDEXING_FU_PIPELINE_NAME

logger = logging.getLogger(__name__)
router = APIRouter()

try:
    INDEXING_PIPELINE = CustomPipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_FU_PIPELINE_NAME)
except KeyError:
    INDEXING_PIPELINE = None
    logger.info("Indexing Pipeline not found in the YAML configuration. File Upload API will not be available.")


os.makedirs(FILE_UPLOAD_PATH, exist_ok=True)  # create directory for uploading files


class Response(BaseModel):
    status: str


@router.post("/file-upload", response_model=Response)
def file_upload(
    file: UploadFile = File(...),
    meta: Optional[str] = Form("null"),  # JSON serialized string
    remove_numeric_tables: Optional[bool] = Form(None),
    remove_whitespace: Optional[bool] = Form(None),
    remove_empty_lines: Optional[bool] = Form(None),
    remove_header_footer: Optional[bool] = Form(None),
    valid_languages: Optional[List[str]] = Form(None),
    split_by: Optional[str] = Form(None),
    split_length: Optional[int] = Form(None),
    split_overlap: Optional[int] = Form(None),
    split_respect_sentence_boundary: Optional[bool] = Form(False)  # Set to False temporarly because of unresolved issue https://github.com/deepset-ai/haystack/issues/1038
):
    """File Upload endpoint.

    Receives a document as input from any file type, extracts its text content, 
    preprocesses it, gets the corresponding embeddings and adds it to the document
    store.
    """
    if not INDEXING_PIPELINE:
        raise HTTPException(status_code=501, detail="Indexing Pipeline is not configured.")
    try:
        file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        meta = json.loads(meta) or {}
        meta["name"] = file.filename

        INDEXING_PIPELINE.run(
            file_path=file_path,
            meta=meta,
            remove_numeric_tables=remove_numeric_tables,
            remove_whitespace=remove_whitespace,
            remove_empty_lines=remove_empty_lines,
            remove_header_footer=remove_header_footer,
            valid_languages=valid_languages,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary
        )
    except:
        return {'status': 'Fail'}
    else:
        return {'status': 'Success'}
    finally:
        file.file.close()

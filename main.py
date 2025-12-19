from __future__ import annotations

import logging
import secrets
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator
from uipath.platform import UiPath
from uipath.platform.documents import (
    ActionPriority,
    ClassificationResult,
    ExtractionResponse,
    ProjectType,
    ValidateExtractionAction,
    ValidateClassificationAction,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()
sdk = UiPath()


class Input(BaseModel):
    """Input contract that UiPath coded agents expect."""

    bucket_name: str = Field(
        description="UiPath storage bucket name that contains the document."
    )
    blob_file_path: str = Field(
        description="Path of the file inside the storage bucket."
    )
    folder_path: Optional[str] = Field(
        default=None,
        description="Optional Orchestrator folder path where the bucket resides.",
    )
    project_type: str = Field(
        description="Project flavor. Use PRETRAINED, MODERN, or IXP."
    )
    project_name: Optional[str] = Field(
        default=None,
        description="DU Modern or IXP project name. Required when project_type is not PRETRAINED.",
    )
    project_tag: Optional[str] = Field(
        default=None,
        description="Published tag (for example staging, live). Required when project_type is not PRETRAINED.",
    )
    document_type_name: Optional[str] = Field(
        default=None,
        description="Required when project_type is MODERN or PRETRAINED and classification_results are not provided.",
    )
    validate_extraction: bool = Field(
        default=False,
        description="When true, create a validation action for the extraction results.",
    )
    validate_classification: bool = Field(
        default=False,
        description="When true, use validation results for document type",
    )
    action_title: Optional[str] = Field(
        default=None,
        description="Title for the validation action.",
    )
    action_priority: Optional[str] = Field(
        default="MEDIUM",
        description=(
            "Priority for the validation action (LOW, MEDIUM, HIGH, CRITICAL)."
        ),
    )
    action_catalog: Optional[str] = Field(
        default="default_du_actions",
        description="Action catalog name for the validation action.",
    )
    action_folder: Optional[str] = Field(
        default="Shared",
        description="Folder where the validation action should be created.",
    )
    storage_bucket_name: Optional[str] = Field(
        default="du_storage_bucket",
        description="Storage bucket name used for validation actions.",
    )
    storage_bucket_directory_path: Optional[str] = Field(
        default="/",
        description="Directory path in the storage bucket for validation actions.",
    )
    classification_results: Optional[List[ClassificationResult]] = Field(
        default=None,
        description="Optional classification results to reuse for extraction instead of providing project and file inputs.",
    )
    validated_classification_action: Optional[ValidateClassificationAction] = Field(
        default=None,
        description="Validation action used to retrieve validated classification results when validate_classification is true.",
    )

    @model_validator(mode="after")
    def validate_settings(self) -> "Input":
        if self.validate_classification:
            if not self.validated_classification_action:
                raise ValueError(
                    "validated_classification_action is required when validate_classification is true."
                )
            classification_supplied = True
        else:
            classification_supplied = self.classification_results is not None
            if classification_supplied and not self.classification_results:
                raise ValueError(
                    "classification_results cannot be empty when provided."
                )

        if not classification_supplied:
            missing = [
                name
                for name, value in {
                    "bucket_name": self.bucket_name,
                    "blob_file_path": self.blob_file_path,
                    "project_type": self.project_type,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(
                    f"Missing required fields when classification_results are not provided: {', '.join(missing)}"
                )

            project_type_enum = _parse_project_type(self.project_type)
            if project_type_enum != ProjectType.PRETRAINED:
                project_missing = [
                    name
                    for name, value in {
                        "project_name": self.project_name,
                        "project_tag": self.project_tag,
                    }.items()
                    if not value
                ]
                if project_missing:
                    raise ValueError(
                        f"{', '.join(project_missing)} are required for {project_type_enum.value} projects when classification_results are not provided."
                    )

            if (
                project_type_enum in {ProjectType.MODERN, ProjectType.PRETRAINED}
                and not self.document_type_name
            ):
                raise ValueError(
                    "document_type_name is required for DU Modern and PRETRAINED extraction when classification_results are not provided."
                )

        if self.validate_extraction:
            missing_fields = [
                name
                for name, value in {
                    "action_catalog": self.action_catalog,
                    "action_folder": self.action_folder,
                    "storage_bucket_name": self.storage_bucket_name,
                    "storage_bucket_directory_path": self.storage_bucket_directory_path,
                }.items()
                if not value
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields for validation action when validate_extraction is true: {', '.join(missing_fields)}"
                )

        return self


Input.model_rebuild()


class Output(BaseModel):
    """Structured output returned to UiPath."""

    bucket_name: Optional[str]
    blob_file_path: Optional[str]
    project_type: Optional[str]
    project_name: Optional[str] = None
    project_tag: Optional[str] = None
    document_type_id: Optional[str]
    document_id: Optional[str]
    extraction_results: Dict[str, Any]
    validation_action: Optional[ValidateExtractionAction] = None


Output.model_rebuild()


def _parse_project_type(project_type: Optional[str]) -> ProjectType:
    if not project_type:
        raise ValueError(
            "project_type is required when classification_results are not provided."
        )
    normalized = project_type.strip().upper()
    try:
        return ProjectType[normalized]
    except KeyError as exc:
        allowed = ", ".join(ProjectType.__members__.keys())
        raise ValueError(
            f"Invalid project_type '{project_type}'. Allowed: {allowed}."
        ) from exc


def _parse_action_priority(priority: Optional[str]) -> ActionPriority:
    if not priority:
        return ActionPriority.MEDIUM
    normalized = priority.strip().upper()
    try:
        return ActionPriority[normalized]
    except KeyError as exc:
        allowed = ", ".join(ActionPriority.__members__.keys())
        raise ValueError(
            f"Invalid action_priority '{priority}'. Allowed: {allowed}."
        ) from exc


def _build_action_title(blob_file_path: str, override: Optional[str]) -> str:
    """Create a unique validation action title using the file name and a short hex suffix."""
    if override:
        return override
    filename = Path(blob_file_path).name
    suffix = f"{secrets.randbelow(16**3):03X}"
    return f"Validate Classification - {filename} - {suffix}"


async def _download_from_bucket(
    bucket_name: str,
    blob_file_path: str,
    destination_path: Path,
    folder_path: Optional[str],
) -> Path:
    """Download a file from UiPath storage buckets asynchronously."""
    await sdk.buckets.download_async(
        name=bucket_name,
        blob_file_path=blob_file_path,
        destination_path=str(destination_path),
        folder_path=folder_path,
    )
    return destination_path


async def _extract_document(
    project_type: Optional[ProjectType],
    project_name: Optional[str],
    project_tag: Optional[str],
    document_type_name: Optional[str],
    file_path: Optional[Path],
    classification_result: Optional[ClassificationResult],
) -> ExtractionResponse:
    if classification_result:
        return await sdk.documents.extract_async(
            classification_result=classification_result,
        )

    if not file_path:
        raise ValueError(
            "file_path is required when classification_results are not provided."
        )

    return await sdk.documents.extract_async(
        project_type=project_type,
        project_name=project_name,
        tag=project_tag,
        document_type_name=document_type_name,
        file_path=str(file_path),
    )


async def _run_async(input_data: Input) -> Output:
    classification_result: Optional[ClassificationResult] = None
    if (
        input_data.validate_classification
        and input_data.validated_classification_action
    ):
        logger.info(
            "Fetching validated classification results from action %s",
            input_data.validated_classification_action.operation_id,
        )
        validated_results = (
            await sdk.documents.get_validate_classification_result_async(
                validation_action=input_data.validated_classification_action
            )
        )
        if not validated_results:
            raise ValueError("Validated classification action returned no results.")
        classification_result = validated_results[0]
    elif input_data.classification_results:
        classification_result = input_data.classification_results[0]
    project_type_enum: Optional[ProjectType] = None
    extraction_response: ExtractionResponse

    if classification_result:
        logger.info("Reusing supplied classification results for extraction.")
        extraction_response = await _extract_document(
            project_type=None,
            project_name=None,
            project_tag=None,
            document_type_name=None,
            file_path=None,
            classification_result=classification_result,
        )
    else:
        project_type_enum = _parse_project_type(input_data.project_type)
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination_path = Path(tmp_dir) / Path(input_data.blob_file_path).name  # type: ignore[arg-type]
            logger.info(
                "Downloading %s from bucket %s",
                input_data.blob_file_path,
                input_data.bucket_name,
            )
            downloaded_path = await _download_from_bucket(
                input_data.bucket_name,  # type: ignore[arg-type]
                input_data.blob_file_path,  # type: ignore[arg-type]
                destination_path,
                input_data.folder_path,
            )

            logger.info(
                "Extracting document with project_type=%s", project_type_enum.value
            )
            extraction_response = await _extract_document(
                project_type_enum,
                input_data.project_name,
                input_data.project_tag,
                input_data.document_type_name,
                downloaded_path,
                classification_result=None,
            )

    validation_action = None
    if input_data.validate_extraction:
        priority_enum = _parse_action_priority(input_data.action_priority)
        action_title = _build_action_title(
            input_data.blob_file_path, input_data.action_title
        )
        logger.info("Creating validation action for extraction results")
        validation_action = await sdk.documents.create_validate_extraction_action_async(
            action_title=action_title,
            action_priority=priority_enum,
            action_catalog=input_data.action_catalog,
            action_folder=input_data.action_folder,
            storage_bucket_name=input_data.storage_bucket_name,
            storage_bucket_directory_path=input_data.storage_bucket_directory_path,
            extraction_response=extraction_response,
        )

    project_type_value = (
        project_type_enum.value
        if project_type_enum
        else extraction_response.project_type.value
        if extraction_response.project_type
        else None
    )

    return Output(
        bucket_name=input_data.bucket_name,
        blob_file_path=input_data.blob_file_path,
        project_type=project_type_value,
        project_name=input_data.project_name,
        project_tag=input_data.project_tag,
        document_type_id=extraction_response.document_type_id,
        document_id=extraction_response.extraction_result.document_id,
        extraction_results=extraction_response.extraction_result.model_dump(
            by_alias=True
        ),
        validation_action=validation_action,
    )


async def main(input_data: Input | Dict[str, Any]) -> Output:
    """Entry point for UiPath coded agents."""
    try:
        validated = (
            input_data
            if isinstance(input_data, Input)
            else Input.model_validate(input_data)
        )
    except ValidationError as exc:
        raise ValueError(f"Invalid input: {exc}") from exc

    return await _run_async(validated)

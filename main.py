from __future__ import annotations

import logging
import secrets
import tempfile
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID

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


class FileResource(BaseModel):
    model_config = {"populate_by_name": True}

    id: str = Field(..., alias="ID", description="Resource ID (UUID string).")
    full_name: Optional[str] = Field(None, alias="FullName", description="File name.")
    mime_type: Optional[str] = Field(None, alias="MimeType", description="MIME type.")
    metadata: Optional[Dict[str, Any]] = Field(
        None, alias="Metadata", description="Metadata map."
    )


class PipelineExtractionProject(BaseModel):
    model_config = {"populate_by_name": True}

    name: str = Field(..., description="Extraction project name.")
    project_type: str = Field(
        ..., description="Extraction project type (PRETRAINED, MODERN, IXP)."
    )
    project_tag: Optional[str] = Field(
        default=None, description="Published tag (for example staging, live)."
    )
    id: Optional[str] = Field(
        default=None, description="Extraction project ID (UUID string)."
    )
    extractor_id: Optional[str] = Field(
        default=None, description="Extractor identifier for the project."
    )


class PipelineConfig(BaseModel):
    model_config = {"populate_by_name": True}

    project_id: Optional[str] = Field(
        default=None, description="Classification project ID."
    )
    project_name: Optional[str] = Field(
        default=None, description="Classification project name."
    )
    project_type: Optional[str] = Field(
        default=None, description="Classification project type."
    )
    classifier_id: Optional[str] = Field(
        default=None, description="Classifier identifier."
    )
    extraction_projects: Dict[str, PipelineExtractionProject] = Field(
        default_factory=dict,
        description=(
            "Map of document type keys to extraction projects. Keys must match "
            "classification_results document types."
        ),
    )
    validate_classification: Optional[bool] = Field(
        default=None, description="Pipeline setting for classification validation."
    )
    validate_extraction: Optional[bool] = Field(
        default=None, description="Pipeline setting for extraction validation."
    )
    validate_extraction_later: Optional[bool] = Field(
        default=None, description="Pipeline setting for delayed extraction validation."
    )
    perform_classification: Optional[bool] = Field(
        default=None, description="Pipeline setting for classification."
    )
    perform_extraction: Optional[bool] = Field(
        default=None, description="Pipeline setting for extraction."
    )


class Input(BaseModel):
    """Input contract that UiPath coded agents expect."""

    model_config = {"populate_by_name": True}

    file_resource: FileResource = Field(
        ..., description="UiPath IResource descriptor for a Data Service file."
    )
    project_type: Optional[str] = Field(
        default=None,
        description="Project flavor. Use PRETRAINED, MODERN, or IXP when extracting without classification_results.",
    )
    project_name: Optional[str] = Field(
        default=None,
        description="DU Modern or IXP project name. Required when project_type is not PRETRAINED.",
    )
    project_tag: Optional[str] = Field(
        default=None,
        description="Published tag (for example staging, live). Required when project_type is not PRETRAINED.",
    )
    pipeline_json: Annotated[
        Optional[PipelineConfig],
        Field(
            alias="pipelineJSON",
            description=(
                "Pipeline configuration containing extraction_projects mapped by document type."
            ),
        ),
    ] = None
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
                    "file_resource": self.file_resource,
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

            if project_type_enum == ProjectType.PRETRAINED:
                if not self.pipeline_json or not self.pipeline_json.extraction_projects:
                    raise ValueError(
                        "pipeline_json.extraction_projects is required for PRETRAINED extraction when classification_results are not provided."
                    )
                if len(self.pipeline_json.extraction_projects) != 1:
                    raise ValueError(
                        "pipeline_json.extraction_projects must contain exactly one entry for PRETRAINED extraction when classification_results are not provided."
                    )
        else:
            if not self.pipeline_json or not self.pipeline_json.extraction_projects:
                raise ValueError(
                    "pipeline_json.extraction_projects is required when classification_results are provided or validate_classification is true."
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


def _classification_document_type_key(
    classification_result: ClassificationResult,
) -> str:
    for attr in ("document_type_id", "document_type_name", "document_type"):
        value = getattr(classification_result, attr, None)
        if value:
            return value
    raise ValueError("Classification result did not include a document type key.")


def _resolve_extraction_project(
    pipeline_json: PipelineConfig, document_type_key: str
) -> tuple[str, PipelineExtractionProject]:
    extraction_projects = pipeline_json.extraction_projects or {}
    if document_type_key in extraction_projects:
        return document_type_key, extraction_projects[document_type_key]
    normalized = document_type_key.strip().lower()
    for key, project in extraction_projects.items():
        if key.strip().lower() == normalized:
            return key, project
    available = ", ".join(extraction_projects.keys())
    raise ValueError(
        f"No extraction project configured for document type '{document_type_key}'. Available: {available}"
    )


def _resolve_pretrained_document_type(pipeline_json: Optional[PipelineConfig]) -> str:
    if not pipeline_json or not pipeline_json.extraction_projects:
        raise ValueError(
            "pipeline_json.extraction_projects is required to derive document_type_name for PRETRAINED extraction."
        )
    if len(pipeline_json.extraction_projects) != 1:
        raise ValueError(
            "pipeline_json.extraction_projects must contain exactly one entry to derive document_type_name for PRETRAINED extraction."
        )
    return next(iter(pipeline_json.extraction_projects.keys()))


def _file_resource_filename(file_resource: FileResource) -> str:
    """Resolve a stable filename for the file resource."""
    if file_resource.full_name:
        return Path(file_resource.full_name).name
    return f"resource-{file_resource.id}"


def _parse_resource_id(resource_id: str) -> UUID:
    try:
        return UUID(resource_id)
    except ValueError as exc:
        raise ValueError(
            f"file_resource.ID must be a valid UUID string. Received: {resource_id}"
        ) from exc


def _build_action_title(file_name: str, override: Optional[str]) -> str:
    """Create a unique validation action title using the file name and a short hex suffix."""
    if override:
        return override
    filename = Path(file_name).name
    suffix = f"{secrets.randbelow(16**3):03X}"
    return f"Validate Extraction - {filename} - {suffix}"


async def _download_file_resource(
    file_resource: FileResource,
    destination_path: Path,
) -> Path:
    """Download a Data Service file resource asynchronously."""
    await sdk.attachments.download_async(
        key=_parse_resource_id(file_resource.id),
        destination_path=str(destination_path),
    )
    return destination_path


async def _extract_document(
    project_type: ProjectType,
    project_name: Optional[str],
    project_tag: Optional[str],
    document_type_name: Optional[str],
    file_path: Path,
) -> ExtractionResponse:
    return await sdk.documents.extract_async(
        project_type=project_type,
        project_name=project_name,
        tag=project_tag,
        document_type_name=document_type_name,
        file_path=str(file_path),
    )


async def _run_async(input_data: Input) -> Output:
    classification_result: Optional[ClassificationResult] = None
    file_name = _file_resource_filename(input_data.file_resource)
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
    project_name: Optional[str] = None
    project_tag: Optional[str] = input_data.project_tag
    document_type_name: Optional[str] = None
    extraction_response: ExtractionResponse

    if classification_result:
        if not input_data.pipeline_json:
            raise ValueError(
                "pipeline_json is required when classification_results are provided."
            )
        document_type_key = _classification_document_type_key(classification_result)
        matched_key, extraction_project = _resolve_extraction_project(
            input_data.pipeline_json, document_type_key
        )
        project_type_enum = _parse_project_type(extraction_project.project_type)
        project_name = extraction_project.name
        project_tag = extraction_project.project_tag or project_tag
        if project_type_enum == ProjectType.PRETRAINED:
            document_type_name = matched_key
    else:
        project_type_enum = _parse_project_type(input_data.project_type)
        project_name = input_data.project_name
        if project_type_enum == ProjectType.PRETRAINED:
            document_type_name = _resolve_pretrained_document_type(
                input_data.pipeline_json
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        destination_path = Path(tmp_dir) / file_name
        logger.info(
            "Downloading file resource %s",
            input_data.file_resource.id,
        )
        downloaded_path = await _download_file_resource(
            input_data.file_resource,
            destination_path,
        )

        logger.info("Extracting document with project_type=%s", project_type_enum.value)
        extraction_response = await _extract_document(
            project_type_enum,
            project_name,
            project_tag,
            document_type_name,
            downloaded_path,
        )

    validation_action = None
    if input_data.validate_extraction:
        priority_enum = _parse_action_priority(input_data.action_priority)
        action_title = _build_action_title(
            file_name,
            input_data.action_title,
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
        project_type=project_type_value,
        project_name=project_name,
        project_tag=project_tag,
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

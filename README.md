# ixpy-extract

Async UiPath coded agent that downloads a Data Service file resource, performs extraction with the Documents service (Modern, IXP, or Pretrained with a document type), and optionally creates validation actions. It mirrors the structure and validation logic of the companion classification agent for consistent inputs/outputs.

## Prerequisites
- Python 3.12+ with the dependencies from `pyproject.toml` (install via `uv sync` or `pip install -e .`).
- UiPath cloud credentials available to the SDK (environment variables such as `UIPATH_CLIENT_ID`, `UIPATH_CLIENT_SECRET`, `UIPATH_SCOPE`, `UIPATH_URL`, or an access token).
- Access to the Data Service file resource and DU project (Modern/IXP/Pretrained with document type).

## Entry point
- Function: `main.py:main`
- Type: async; accepts `Input` Pydantic model or compatible dict.

## Input schema (summary)
- `file_resource` (object, required): UiPath IResource descriptor for a Data Service file (`ID` required; `FullName`, `MimeType`, `Metadata` optional).
- `project_type` (str): `PRETRAINED`, `MODERN`, or `IXP` (required unless reusing classification results).
- `project_name` (str, optional): Required for `MODERN`/`IXP` when not providing classification results.
- `project_tag` (str, optional): Required for `MODERN`/`IXP` when not providing classification results.
- `document_type_name` (str, optional): Required for `MODERN` and `PRETRAINED` when not providing classification results.
- `classification_results` (list[ClassificationResult], optional): Reuse raw classification results directly.
- `validate_classification` (bool, default `False`): When true, fetch validated classification results instead of raw ones.
- `validated_classification_action` (ValidateClassificationAction, optional): Required when `validate_classification` is true; the agent will block until the action is completed and will reuse the validated results for extraction.
- `validate_extraction` (bool, default `False`): When true, create a validation action for extraction results.
- `action_title`, `action_priority`, `action_catalog`, `action_folder`, `storage_bucket_name`, `storage_bucket_directory_path`: Validation action settings (required when `validate_extraction` is true).

All validation rules are enforced in `Input.validate_settings` to keep orchestrations simple.

## Output schema (summary)
- `project_type`, `project_name`, `project_tag`
- `document_type_id`, `document_id`
- `extraction_results`: Raw extraction result dictionary.
- `validation_action`: `ValidateExtractionAction` when created.

## Workflow
1) If `validate_classification` is true, the agent calls `get_validate_classification_result_async` with `validated_classification_action` and waits for completion, then uses the validated classification result.
2) If classification results are provided directly, they are reused; otherwise the file resource is downloaded and extracted using the provided project info.
3) Optionally creates a validation action for the extraction output.
4) Returns structured `Output`.

## Quick example
```python
import asyncio
from main import Input, main

payload = Input(
    file_resource={
        "ID": "<file resource UUID>",
        "FullName": "Sample Invoice.pdf"
    },
    project_type="PRETRAINED",
    document_type_name="Invoices"
)

result = asyncio.run(main(payload))
print(result.extraction_results)
```

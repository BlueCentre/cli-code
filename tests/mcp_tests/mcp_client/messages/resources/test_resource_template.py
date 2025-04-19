import pytest

from mcp_code.mcp_client.messages.resources.resource_template import ResourceTemplate

# Test cases for ResourceTemplate


def test_resource_template_instantiation_required_only():
    """Test instantiation with only required fields."""
    template = ResourceTemplate(uriTemplate="file:///templates/{name}.txt", name="FileTemplate")
    assert template.uriTemplate == "file:///templates/{name}.txt"
    assert template.name == "FileTemplate"
    assert template.description is None
    assert template.mimeType is None
    assert template.model_dump() == {
        "uriTemplate": "file:///templates/{name}.txt",
        "name": "FileTemplate",
        "description": None,
        "mimeType": None,
    }


def test_resource_template_instantiation_all_fields():
    """Test instantiation with all fields provided."""
    template = ResourceTemplate(
        uriTemplate="https://example.com/api/v1/items/{id}",
        name="APITemplate",
        description="Template for API items.",
        mimeType="application/json",
    )
    assert template.uriTemplate == "https://example.com/api/v1/items/{id}"
    assert template.name == "APITemplate"
    assert template.description == "Template for API items."
    assert template.mimeType == "application/json"
    assert template.model_dump() == {
        "uriTemplate": "https://example.com/api/v1/items/{id}",
        "name": "APITemplate",
        "description": "Template for API items.",
        "mimeType": "application/json",
    }


def test_resource_template_instantiation_some_optional():
    """Test instantiation with some optional fields."""
    template = ResourceTemplate(uriTemplate="git://repo.git#branch/{path}", name="GitTemplate", mimeType="text/plain")
    assert template.uriTemplate == "git://repo.git#branch/{path}"
    assert template.name == "GitTemplate"
    assert template.description is None
    assert template.mimeType == "text/plain"
    assert template.model_dump() == {
        "uriTemplate": "git://repo.git#branch/{path}",
        "name": "GitTemplate",
        "description": None,
        "mimeType": "text/plain",
    }


# Add tests for potential validation errors if any constraints exist

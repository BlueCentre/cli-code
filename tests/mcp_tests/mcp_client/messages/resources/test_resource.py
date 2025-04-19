import pytest

from mcp_code.mcp_client.messages.resources.resource import Resource

# Test cases for Resource


def test_resource_instantiation_required_only():
    """Test instantiation with only required fields."""
    resource = Resource(uri="file:///path/to/doc.txt", name="Document")
    assert resource.uri == "file:///path/to/doc.txt"
    assert resource.name == "Document"
    assert resource.description is None
    assert resource.mimeType is None
    assert resource.model_dump() == {
        "uri": "file:///path/to/doc.txt",
        "name": "Document",
        "description": None,
        "mimeType": None,
    }


def test_resource_instantiation_all_fields():
    """Test instantiation with all fields provided."""
    resource = Resource(
        uri="https://example.com/image.jpg", name="ExampleImage", description="An example image.", mimeType="image/jpeg"
    )
    assert resource.uri == "https://example.com/image.jpg"
    assert resource.name == "ExampleImage"
    assert resource.description == "An example image."
    assert resource.mimeType == "image/jpeg"
    assert resource.model_dump() == {
        "uri": "https://example.com/image.jpg",
        "name": "ExampleImage",
        "description": "An example image.",
        "mimeType": "image/jpeg",
    }


def test_resource_instantiation_some_optional():
    """Test instantiation with some optional fields."""
    resource = Resource(uri="git://repo.git#main/README.md", name="ProjectReadme", mimeType="text/markdown")
    assert resource.uri == "git://repo.git#main/README.md"
    assert resource.name == "ProjectReadme"
    assert resource.description is None
    assert resource.mimeType == "text/markdown"
    assert resource.model_dump() == {
        "uri": "git://repo.git#main/README.md",
        "name": "ProjectReadme",
        "description": None,
        "mimeType": "text/markdown",
    }


# Add tests for potential validation errors if any constraints exist

import pytest

from mcp_code.mcp_client.messages.resources.resource_content import ResourceContent

# Test cases for ResourceContent


def test_resource_content_instantiation_required_only():
    """Test instantiation with only the required uri field."""
    content = ResourceContent(uri="file:///path/to/resource.txt")
    assert content.uri == "file:///path/to/resource.txt"
    assert content.mimeType is None
    assert content.text is None
    assert content.blob is None
    assert content.model_dump() == {"uri": "file:///path/to/resource.txt", "mimeType": None, "text": None, "blob": None}


def test_resource_content_instantiation_with_text():
    """Test instantiation with text content."""
    content = ResourceContent(uri="file:///data/info.txt", mimeType="text/plain", text="This is the content.")
    assert content.uri == "file:///data/info.txt"
    assert content.mimeType == "text/plain"
    assert content.text == "This is the content."
    assert content.blob is None
    assert content.model_dump() == {
        "uri": "file:///data/info.txt",
        "mimeType": "text/plain",
        "text": "This is the content.",
        "blob": None,
    }


def test_resource_content_instantiation_with_blob():
    """Test instantiation with blob content."""
    # Example base64 encoded string (e.g., echo -n "binary" | base64 -> YmluYXJ5)
    blob_data = "YmluYXJ5"
    content = ResourceContent(uri="file:///data/image.png", mimeType="image/png", blob=blob_data)
    assert content.uri == "file:///data/image.png"
    assert content.mimeType == "image/png"
    assert content.text is None
    assert content.blob == blob_data
    assert content.model_dump() == {
        "uri": "file:///data/image.png",
        "mimeType": "image/png",
        "text": None,
        "blob": blob_data,
    }


def test_resource_content_instantiation_with_all():
    """Test instantiation with all optional fields (though text+blob is unusual)."""
    blob_data = "YmluYXJ5"
    content = ResourceContent(
        uri="file:///mixed.dat", mimeType="application/octet-stream", text="Fallback text", blob=blob_data
    )
    assert content.uri == "file:///mixed.dat"
    assert content.mimeType == "application/octet-stream"
    assert content.text == "Fallback text"
    assert content.blob == blob_data
    assert content.model_dump() == {
        "uri": "file:///mixed.dat",
        "mimeType": "application/octet-stream",
        "text": "Fallback text",
        "blob": blob_data,
    }


# Add tests for potential validation (e.g., maybe text and blob shouldn't both be set)
# if such validation rules exist.

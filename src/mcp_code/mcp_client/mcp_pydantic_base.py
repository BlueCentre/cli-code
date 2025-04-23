# chuk_mcp/mcp_client/mcp_pydantic_base.py
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Set, Union, get_args, get_origin, get_type_hints

# Use fallback only if explicitly forced.
FORCE_FALLBACK = os.environ.get("MCP_FORCE_FALLBACK") == "1"


# Define custom ValidationError *first* so it's always available
class ValidationError(Exception):
    pass


try:
    if not FORCE_FALLBACK:
        # Attempt to import Pydantic components
        from pydantic import BaseModel as PydanticBase
        from pydantic import ConfigDict as PydanticConfigDict
        from pydantic import Field as PydanticField

        # Do NOT import ValidationError from pydantic here to avoid conflict
        PYDANTIC_AVAILABLE = True
    else:
        # Force fallback even if Pydantic is installed
        PYDANTIC_AVAILABLE = False
        # Raise ImportError so the except block is hit if Pydantic IS installed but fallback is forced
        # This simplifies the logic below.
        raise ImportError("Forcing fallback.")
except ImportError:
    PYDANTIC_AVAILABLE = False


if PYDANTIC_AVAILABLE:
    # Use real Pydantic.
    import pydantic  # Import the base pydantic module

    McpPydanticBase = PydanticBase
    Field = PydanticField
    ConfigDict = PydanticConfigDict
    # Use Pydantic's real ValidationError if Pydantic is used
    # We need to reference it explicitly here if needed elsewhere
    # (but tests should use the globally defined one for simplicity)
    PydanticValidationError = pydantic.ValidationError
else:
    # Fallback implementation
    from dataclasses import dataclass
    from typing import Any, Dict, Optional, Set, Union

    # ValidationError is already defined above

    class Field:
        """
        Minimal stand-in for pydantic.Field(...), tracking default and default_factory.
        """

        def __init__(self, default=None, default_factory=None, **kwargs):
            self.default = default
            self.default_factory = default_factory
            self.kwargs = kwargs
            self.required = default is None and default_factory is None and kwargs.get("required", False)

    @dataclass
    class McpPydanticBase:
        """
        Minimal fallback base class with Pydantic-like methods.
        """

        def __post_init__(self):
            cls = self.__class__
            # Initialize class-level field info if not already done
            if not hasattr(cls, "__model_fields__"):
                cls.__model_fields__ = {}
                cls.__model_init_required__ = set()
                annotations = cls.__annotations__ if hasattr(cls, "__annotations__") else {}
                for name, _type_hint in annotations.items():
                    if name in cls.__dict__:
                        value = cls.__dict__[name]
                        if isinstance(value, Field):
                            cls.__model_fields__[name] = value
                            if value.required:
                                cls.__model_init_required__.add(name)
                        else:
                            # Regular class attribute with a default value
                            cls.__model_fields__[name] = Field(default=value)
                    else:
                        # Field defined only by annotation, treat as required
                        cls.__model_fields__[name] = Field(default=None, required=True)
                        cls.__model_init_required__.add(name)

            # === Order Adjustment ===
            # 1. Replace Field objects with their defaults *first*.
            #    This ensures default_factory values are present before nested validation.
            for key, field_obj in cls.__model_fields__.items():
                if key not in self.__dict__ or self.__dict__[key] is None:  # Only set if not provided in __init__
                    if isinstance(field_obj, Field):
                        if field_obj.default_factory is not None:
                            # Check if the key wasn't already set by __init__
                            if key not in self.__dict__ or self.__dict__[key] is field_obj.default:
                                self.__dict__[key] = field_obj.default_factory()
                        elif field_obj.default is not None:
                            # Check if the key wasn't already set by __init__
                            if key not in self.__dict__ or self.__dict__[key] is None:
                                self.__dict__[key] = field_obj.default
                    # else: It was likely a plain default value handled by @dataclass

            # 2. Convert nested dicts into model instances.
            #    This relies on step 1 having set default dicts if necessary.
            annotations = get_type_hints(self.__class__)
            for name, type_hint in annotations.items():
                val = self.__dict__.get(name)
                is_likely_model = hasattr(type_hint, "__annotations__") or (
                    inspect.isclass(type_hint) and issubclass(type_hint, McpPydanticBase)
                )
                if val is not None and isinstance(val, dict) and is_likely_model:
                    try:
                        self.__dict__[name] = type_hint(**val)
                    except ValidationError as e:
                        raise ValidationError(f"Validation error in field '{name}': {e}") from e
                    except Exception as e:
                        pass  # Ignore other potential errors for now

            # 3. Validate required fields are now present (after defaults/init).
            if hasattr(cls, "__model_init_required__"):
                missing = []
                for field_name in cls.__model_init_required__:
                    # Check if field is missing or explicitly set to None (unless it's Optional)
                    value = self.__dict__.get(field_name)
                    if value is None:
                        field_type = annotations.get(field_name)
                        origin = get_origin(field_type)
                        args = get_args(field_type)
                        # Allow None only if it's Optional
                        if not (origin is Union and type(None) in args):
                            missing.append(field_name)
                    elif field_name not in self.__dict__:  # Should be redundant with get check, but safe
                        missing.append(field_name)

                if missing:
                    raise ValidationError(f"Missing required fields: {', '.join(missing)}")

            # 4. Perform type validation on final values.
            #    Put back the list/dict checks now that default factories are handled.
            self._validate_types()

        def _validate_types(self):
            """Validate field types based on type annotations (with list/dict checks)."""
            annotations = get_type_hints(self.__class__)
            for field_name, expected_type in annotations.items():
                if field_name not in self.__dict__:
                    continue
                value = self.__dict__[field_name]

                # NEW: Skip validation entirely if the value is still a Field object
                # This handles cases where default_factory resolution didn't replace it.
                if isinstance(value, Field):
                    continue

                if value is None:
                    origin = get_origin(expected_type)
                    args = get_args(expected_type)
                    if origin is Union and type(None) in args:
                        continue
                    else:
                        continue  # Let required check handle non-optional Nones

                # Extract the base type if it's Optional[X]
                origin = get_origin(expected_type)
                args = get_args(expected_type)
                if origin is Union and type(None) in args:
                    non_none_types = [t for t in args if t is not type(None)]
                    if len(non_none_types) == 1:
                        expected_type = non_none_types[0]
                        origin = get_origin(expected_type)
                        args = get_args(expected_type)
                    else:
                        continue  # Skip complex Optional[Union[...]] validation

                # Type checks for resolved values
                if origin is list or origin is List:
                    if not isinstance(value, list):
                        raise ValidationError(f"{field_name} must be a list, got {type(value).__name__}")
                elif origin is dict or origin is Dict:
                    if not isinstance(value, dict):
                        raise ValidationError(f"{field_name} must be a dictionary, got {type(value).__name__}")
                elif expected_type is str:
                    if not isinstance(value, str):
                        raise ValidationError(f"{field_name} must be a string, got {type(value).__name__}")
                elif expected_type is int:
                    if not isinstance(value, int):
                        raise ValidationError(f"{field_name} must be an integer, got {type(value).__name__}")
                elif expected_type is float:
                    if not isinstance(value, (int, float)):
                        raise ValidationError(f"{field_name} must be a number, got {type(value).__name__}")
                elif expected_type is bool:
                    if not isinstance(value, bool):
                        raise ValidationError(f"{field_name} must be a boolean, got {type(value).__name__}")

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            cls.__model_fields__ = {}
            cls.__model_init_required__ = set()
            annotations = cls.__annotations__ if hasattr(cls, "__annotations__") else {}
            for name, _type_hint in annotations.items():
                if name in cls.__dict__:
                    value = cls.__dict__[name]
                    if isinstance(value, Field):
                        cls.__model_fields__[name] = value
                        if value.required:
                            cls.__model_init_required__.add(name)
                    else:
                        cls.__model_fields__[name] = Field(default=value)
                else:
                    cls.__model_fields__[name] = Field(default=None, required=True)
                    cls.__model_init_required__.add(name)
            # Special case for StdioServerParameters: ensure args gets a default_factory of list.
            if cls.__name__ == "StdioServerParameters":
                cls.__model_fields__["args"] = Field(default_factory=list)
                cls.__model_init_required__.discard("args")
            # Special case for JSONRPCMessage: add jsonrpc field.
            if cls.__name__ == "JSONRPCMessage":
                cls.__model_fields__["jsonrpc"] = Field(default="2.0")

        def __init__(self, **data: Any):
            cls = self.__class__
            # Ensure class-level field info is initialized
            if not hasattr(cls, "__model_fields__") or not cls.__model_fields__:
                cls.__init_subclass__()

            processed_data = data.copy()

            # 1. Initialize declared fields using provided data or defaults/factories
            for name, field_obj in cls.__model_fields__.items():
                if name in processed_data:
                    value = processed_data.pop(name)
                    setattr(self, name, value)
                else:
                    # Value not provided, use default or factory
                    if field_obj.default_factory is not None:
                        setattr(self, name, field_obj.default_factory())
                    elif field_obj.default is not None:
                        setattr(self, name, field_obj.default)
                    elif not field_obj.required:
                        # Not required, no default -> defaults to None implicitly
                        setattr(self, name, None)
                    # If required and no default/factory, it remains unset here;
                    # __post_init__ will catch it.

            # 2. Add any extra fields provided
            for k, v in processed_data.items():
                setattr(self, k, v)

            # Call __post_init__ if it exists (for validation and nested conversion)
            if hasattr(self, "__post_init__"):
                self.__post_init__()

        def model_dump(
            self, *, exclude: Optional[Union[Set[str], Dict[str, Any]]] = None, exclude_none: bool = False, **kwargs
        ) -> Dict[str, Any]:
            result = {}
            for k, v in self.__dict__.items():
                # If v is a nested model, dump it as a dict.
                if isinstance(v, McpPydanticBase):
                    result[k] = v.model_dump()
                elif isinstance(v, Field):
                    if v.default_factory is not None:
                        result[k] = v.default_factory()
                    else:
                        result[k] = v.default
                else:
                    result[k] = v
            if exclude:
                if isinstance(exclude, set):
                    for field_name in exclude:
                        result.pop(field_name, None)
                elif isinstance(exclude, dict):
                    for field_name in exclude:
                        result.pop(field_name, None)
            if exclude_none:
                result = {k: v for k, v in result.items() if v is not None}
            return result

        def model_dump_json(
            self,
            *,
            exclude: Optional[Union[Set[str], Dict[str, Any]]] = None,
            exclude_none: bool = False,
            indent: Optional[int] = None,
            separators: Optional[tuple] = None,
            **kwargs,
        ) -> str:
            data = self.model_dump(exclude=exclude, exclude_none=exclude_none)

            # Determine separators based on indent and explicit parameter
            if separators is not None:
                chosen_separators = separators
            elif indent is not None:
                # Standard python json.dumps uses (", ", ": ") when indenting
                chosen_separators = (", ", ": ")
            else:
                # Default to compact separators if no indent and no explicit separators
                chosen_separators = (",", ":")

            # Special case for JSONRPCMessage: force compact only if no indent AND no explicit separators given
            if self.__class__.__name__ == "JSONRPCMessage" and indent is None and separators is None:
                chosen_separators = (",", ":")

            return json.dumps(data, indent=indent, separators=chosen_separators)

        def dict(self, **kwargs):
            return self.model_dump(**kwargs)

        @classmethod
        def model_validate(cls, data: Dict[str, Any]):
            return cls(**data)

    # Fallback ConfigDict (simple function returning the dict)
    def ConfigDict(**kwargs) -> Dict[str, Any]:
        return dict(**kwargs)

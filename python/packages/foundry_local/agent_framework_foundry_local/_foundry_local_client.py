# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from importlib import import_module
from typing import Any, ClassVar, Generic, Literal, cast, overload

from agent_framework import (
    ChatAndFunctionMiddlewareTypes,
    ChatMiddlewareLayer,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    CompactionStrategy,
    FunctionInvocationConfiguration,
    FunctionInvocationLayer,
    Message,
    ResponseStream,
    TokenizerProtocol,
)
from agent_framework._settings import load_settings
from agent_framework.observability import ChatTelemetryLayer
from agent_framework_openai._chat_completion_client import RawOpenAIChatCompletionClient
from openai import AsyncOpenAI
from pydantic import BaseModel

from ._feature_usage import FeatureIndex

FoundryLocalManager: Any
_FoundryLocalConfiguration: Any
_LegacyDeviceType: Any
_use_modern_foundry_local_sdk: bool

try:
    from foundry_local_sdk import Configuration as _ModernFoundryLocalConfiguration
    from foundry_local_sdk import FoundryLocalManager as _ModernFoundryLocalManager

    FoundryLocalManager = cast(Any, _ModernFoundryLocalManager)
    _FoundryLocalConfiguration = cast(Any, _ModernFoundryLocalConfiguration)
    _LegacyDeviceType = None
    _use_modern_foundry_local_sdk = True
except ModuleNotFoundError as exc:
    if exc.name != "foundry_local_sdk":
        raise

    _legacy_sdk = import_module("foundry_local")
    _legacy_models = import_module("foundry_local.models")
    FoundryLocalManager = _legacy_sdk.FoundryLocalManager
    _LegacyDeviceType = _legacy_models.DeviceType
    _FoundryLocalConfiguration = None
    _use_modern_foundry_local_sdk = False

if sys.version_info >= (3, 13):
    from typing import TypeVar  # pragma: no cover
else:
    from typing_extensions import TypeVar  # pragma: no cover
if sys.version_info >= (3, 11):
    from typing import TypedDict  # pragma: no cover
else:
    from typing_extensions import TypedDict  # pragma: no cover


__all__ = [
    "FoundryLocalChatOptions",
    "FoundryLocalClient",
    "FoundryLocalSettings",
]

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel | None, default=None)


# region Foundry Local Chat Options TypedDict


class FoundryLocalChatOptions(ChatOptions[ResponseModelT], Generic[ResponseModelT], total=False):
    """Azure Foundry Local (local model deployment) chat options dict.

    Extends base ChatOptions for local model inference via Foundry Local.
    Foundry Local provides an OpenAI-compatible API, so most standard
    OpenAI chat completion options are supported.

    See: https://github.com/Azure/azure-ai-foundry-model-inference

    Keys:
        # Inherited from ChatOptions (supported via OpenAI-compatible API):
        model: The model identifier or alias (e.g., 'phi-4-mini').
        temperature: Sampling temperature (0-2).
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum tokens to generate.
        stop: Stop sequences.
        tools: List of tools available to the model.
        tool_choice: How the model should use tools.
        frequency_penalty: Frequency penalty (-2.0 to 2.0).
        presence_penalty: Presence penalty (-2.0 to 2.0).
        seed: Random seed for reproducibility.

        # Options with limited support (depends on the model):
        response_format: Response format specification.
            Not all local models support JSON mode.
        logit_bias: Token bias dictionary.
            May not be supported by all models.

        # Options not supported in Foundry Local:
        user: Not used locally.
        store: Not applicable for local inference.
        metadata: Not applicable for local inference.

        # Foundry Local-specific options:
        extra_body: Additional request body parameters to pass to the model.
            Can be used for model-specific options not covered by standard API.

    Note:
        The actual options supported depend on the specific model being used.
        Some models (like Phi-4) may not support all OpenAI API features.
        Options not supported by the model will typically be ignored.
    """

    # Foundry Local-specific options
    extra_body: dict[str, Any]
    """Additional request body parameters for model-specific options."""

    # ChatOptions fields not applicable for local inference
    user: None  # type: ignore[misc]
    """Not used for local model inference."""

    store: None  # type: ignore[misc]
    """Not applicable for local inference."""


FoundryLocalChatOptionsT = TypeVar(
    "FoundryLocalChatOptionsT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="FoundryLocalChatOptions",
    covariant=True,
)


# endregion


def _device_name(device: Any) -> str:
    """Return a normalized device name from a string or SDK enum."""
    value = getattr(device, "value", None)
    if value is None:
        value = getattr(device, "name", device)
    return str(value).casefold()


def _resolve_modern_model(manager: Any, model_identifier: str, device: Any | None = None) -> Any | None:
    """Resolve a model by alias or variant ID using the current SDK catalog API."""
    catalog = manager.catalog
    model = catalog.get_model(model_identifier)
    is_variant_id = model is None
    if model is None:
        model = catalog.get_model_variant(model_identifier)
    if model is None or device is None:
        return model

    requested_device = _device_name(device)
    if requested_device == "auto":
        return model

    device_markers = {
        "gpu": ("gpu", "cuda", "webgpu", "winml", "directml"),
        "cpu": ("cpu",),
    }.get(requested_device, (requested_device,))
    matching_variant = next(
        (
            variant
            for variant in model.variants
            if any(marker in str(getattr(variant, "id", "")).casefold() for marker in device_markers)
        ),
        None,
    )
    if matching_variant is None:
        raise ValueError(
            f"Model with ID or alias '{model_identifier}:{_device_name(device).upper()}' not found in Foundry Local."
        )

    if is_variant_id:
        return model

    model.select_variant(matching_variant)
    return model


class _FoundryLocalManagerCompatibility:
    """Expose the previous manager convenience methods over the modern SDK."""

    def __init__(self, manager: Any, base_url: str, api_key: str) -> None:
        self._manager = manager
        self.endpoint = base_url
        self.api_key = api_key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def list_catalog_models(self) -> list[Any]:
        """Return models from the current SDK catalog."""
        return self._manager.catalog.list_models()

    def get_model_info(self, model: str, device: Any | None = None) -> Any:
        """Resolve a model by alias or ID, optionally selecting a device variant."""
        model_info = _resolve_modern_model(self._manager, model, device)
        if model_info is None:
            raise ValueError(f"Model with ID or alias '{model}' not found in Foundry Local.")
        return model_info

    def download_model(self, model: str, device: Any | None = None) -> None:
        """Download a model through the current SDK model API."""
        self.get_model_info(model, device).download()

    def load_model(self, model: str, device: Any | None = None) -> None:
        """Load a model through the current SDK model API."""
        self.get_model_info(model, device).load()


class FoundryLocalSettings(TypedDict, total=False):
    """Foundry local model settings.

    Settings are resolved in this order: explicit keyword arguments, values from an
    explicitly provided .env file, then environment variables with the prefix
    'FOUNDRY_LOCAL_'.

    Keys:
        model: The name of the model deployment to use.
            (Env var FOUNDRY_LOCAL_MODEL)
    """

    model: str | None


class FoundryLocalClient(
    FunctionInvocationLayer[FoundryLocalChatOptionsT],
    ChatMiddlewareLayer[FoundryLocalChatOptionsT],
    ChatTelemetryLayer[FoundryLocalChatOptionsT],
    RawOpenAIChatCompletionClient[FoundryLocalChatOptionsT],
    Generic[FoundryLocalChatOptionsT],
):
    """Foundry Local Chat completion class with middleware, telemetry, and function invocation support."""

    _FEATURE_USAGE_INDEX: ClassVar[int | None] = FeatureIndex.FOUNDRY_LOCAL

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = ...,
        options: ChatOptions[ResponseModelT],
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    ) -> Awaitable[ChatResponse[ResponseModelT]]: ...

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = ...,
        options: FoundryLocalChatOptionsT | ChatOptions[None] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    ) -> Awaitable[ChatResponse[Any]]: ...

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        options: FoundryLocalChatOptionsT | ChatOptions[Any] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse[Any]]: ...

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: FoundryLocalChatOptionsT | ChatOptions[Any] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
        """Get a response from the Foundry Local chat client with all standard layers enabled."""
        super_get_response = cast(
            "Callable[..., Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]]",
            super().get_response,
        )
        effective_client_kwargs = dict(client_kwargs) if client_kwargs is not None else {}
        if middleware is not None:
            effective_client_kwargs["middleware"] = middleware
        return super_get_response(
            messages=messages,
            stream=stream,
            options=options,
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            function_invocation_kwargs=function_invocation_kwargs,
            client_kwargs=effective_client_kwargs,
        )

    def __init__(
        self,
        model: str | None = None,
        *,
        bootstrap: bool = True,
        timeout: float | None = None,
        prepare_model: bool = True,
        device: Any | None = None,
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str = "utf-8",
    ) -> None:
        """Initialize a FoundryLocalClient.

        Keyword Args:
            model: The Foundry Local model ID or alias to use. If not provided,
                it will be loaded from the FoundryLocalSettings.
            bootstrap: Whether to initialize the Foundry Local SDK when needed. Default is True.
                With the modern SDK, ``bootstrap=False`` requires an already initialized
                ``FoundryLocalManager``; the embedded HTTP endpoint is started for the client.
            timeout: Optional timeout for requests to Foundry Local.
                This timeout is applied to requests sent through the OpenAI-compatible API.
            prepare_model: Whether to download the model into the cache, and load the model into
                the inferencing service upon initialization. Default is True.
                If false, make sure the model is available before the first request. Depending
                on the installed SDK, the first request may load the model and take longer.
            device: The device type to use for model inference.
                The device selects the appropriate model variant. A legacy ``DeviceType``
                value or a device name such as ``"cpu"`` or ``"gpu"`` may be used.
                If omitted, Foundry Local chooses.
            additional_properties: Additional properties stored on the client instance.
            middleware: Optional sequence of ChatAndFunctionMiddlewareTypes to apply to requests.
            function_invocation_configuration: Optional configuration for function invocation support.
            env_file_path: If provided, the .env settings are read from this file path location.
            env_file_encoding: The encoding of the .env file, defaults to 'utf-8'.

        Examples:

            .. code-block:: python

                # Create a FoundryLocalClient with a specific model ID:
                from agent_framework.foundry import FoundryLocalClient

                client = FoundryLocalClient(model="phi-4-mini")

                agent = client.as_agent(
                    name="LocalAgent",
                    instructions="You are a helpful agent.",
                    tools=get_weather,
                )
                response = await agent.run("What's the weather like in Seattle?")

                # Or you can set the model id in the environment:
                os.environ["FOUNDRY_LOCAL_MODEL"] = "phi-4-mini"
                client = FoundryLocalClient()

                # The FoundryLocalManager is available via the `manager` property.
                # For instance, to find out which models are available:
                for model in client.manager.list_catalog_models():
                    print(f"- {model.alias} - id={model.id}")

                # Other options include specifying the device type:
                client = FoundryLocalClient(
                    model="phi-4-mini",
                    device="gpu",
                )
                # and choosing if the model should be prepared on initialization:
                client = FoundryLocalClient(
                    model="phi-4-mini",
                    prepare_model=False,
                )
                # Beware, in this case the first request to generate a completion
                # will take a long time as the model is loaded then.
                # Alternatively, you could call the `download_model` and `load_model` methods
                # on the `manager` property manually.
                client.manager.download_model("phi-4-mini", device="cpu")
                client.manager.load_model("phi-4-mini", device="cpu")

                # You can also use the CLI:
                `foundry model load phi-4-mini --device Auto`

                # Using custom ChatOptions with type safety:
                from typing import TypedDict
                from agent_framework.foundry import FoundryLocalChatOptions

                class MyOptions(FoundryLocalChatOptions, total=False):
                    my_custom_option: str

                client: FoundryLocalClient[MyOptions] = FoundryLocalClient(model="phi-4-mini")
                response = await client.get_response("Hello", options={"my_custom_option": "value"})

        Raises:
            ValueError: If the specified model ID, alias, or device variant is not found.
                Sometimes a model might be available but if you have specified a device
                type that is not supported by the model, it will not be found.

        """
        settings = load_settings(
            FoundryLocalSettings,
            env_prefix="FOUNDRY_LOCAL_",
            required_fields=["model"],
            model=model,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        model_setting: str = settings["model"]  # type: ignore[assignment]

        if _use_modern_foundry_local_sdk:
            manager = FoundryLocalManager.instance
            if manager is None:
                if not bootstrap:
                    raise ValueError(
                        "bootstrap=False requires FoundryLocalManager to be initialized before creating the client."
                    )
                if _FoundryLocalConfiguration is None:
                    raise RuntimeError("The installed Foundry Local SDK does not provide Configuration.")
                FoundryLocalManager.initialize(_FoundryLocalConfiguration(app_name="agent-framework"))
                manager = FoundryLocalManager.instance
            if manager is None:
                raise RuntimeError("Foundry Local SDK initialization did not create a manager instance.")

            model_info = _resolve_modern_model(manager, model_setting, device)
            if model_info is None:
                requested_model = model_setting
                if device is not None:
                    requested_model += f":{_device_name(device).upper()}"
                message = f"Model with ID or alias '{requested_model}' not found in Foundry Local."
                raise ValueError(message)
            if prepare_model:
                model_info.download()
                model_info.load()

            if not manager.urls:
                manager.start_web_service()
            if not manager.urls:
                raise RuntimeError("Foundry Local web service did not provide an endpoint.")

            base_url = f"{manager.urls[0].rstrip('/')}/v1"
            api_key = "local"
            async_client_kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key}
            if timeout is not None:
                async_client_kwargs["timeout"] = timeout
            async_client = AsyncOpenAI(**async_client_kwargs)
            manager_compatibility = _FoundryLocalManagerCompatibility(manager, base_url, api_key)
        else:
            manager = FoundryLocalManager(bootstrap=bootstrap, timeout=timeout)
            legacy_device = device
            if isinstance(device, str) and _LegacyDeviceType is not None:
                try:
                    legacy_device = _LegacyDeviceType[device.upper()]
                except KeyError:
                    legacy_device = _LegacyDeviceType(device)
            model_info = manager.get_model_info(model_setting, device=legacy_device)
            if model_info is None:
                if device is not None:
                    message = (
                        f"Model with ID or alias '{model_setting}:{_device_name(device).upper()}' "
                        "not found in Foundry Local."
                    )
                else:
                    message = (
                        f"Model with ID or alias '{model_setting}' for your current device not found in Foundry Local."
                    )
                raise ValueError(message)
            if prepare_model:
                manager.download_model(model_info.id, device=legacy_device)
                manager.load_model(model_info.id, device=legacy_device)
            async_client_kwargs: dict[str, Any] = {"base_url": manager.endpoint, "api_key": manager.api_key}
            if timeout is not None:
                async_client_kwargs["timeout"] = timeout
            async_client = AsyncOpenAI(**async_client_kwargs)
            manager_compatibility = manager

        super().__init__(
            model=model_info.id,
            async_client=async_client,
            additional_properties=additional_properties,
            middleware=middleware,
            function_invocation_configuration=function_invocation_configuration,
        )
        self.manager = manager_compatibility

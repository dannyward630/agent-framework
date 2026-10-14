# Copyright (c) Microsoft. All rights reserved.

import inspect
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from agent_framework import Agent, SupportsChatGetResponse
from agent_framework._settings import load_settings
from agent_framework.exceptions import SettingNotFoundError
from agent_framework.foundry import FoundryLocalClient

from agent_framework_foundry_local import _foundry_local_client as foundry_local_client
from agent_framework_foundry_local._feature_usage import FeatureIndex
from agent_framework_foundry_local._foundry_local_client import FoundryLocalSettings


@pytest.fixture
def legacy_foundry_local_sdk(monkeypatch: Any) -> None:
    """Exercise the SDK 0.5 compatibility path independently of the installed SDK."""
    monkeypatch.setattr(foundry_local_client, "_use_modern_foundry_local_sdk", False)


def _legacy_device(name: str) -> Any:
    legacy_device_type = foundry_local_client._LegacyDeviceType
    return getattr(legacy_device_type, name) if legacy_device_type is not None else name.lower()


# Settings Tests


def test_foundry_local_owns_foundry_local_feature() -> None:
    assert FoundryLocalClient._FEATURE_USAGE_INDEX is FeatureIndex.FOUNDRY_LOCAL


def test_foundry_local_settings_init_from_env(foundry_local_unit_test_env: dict[str, str]) -> None:
    """Test FoundryLocalSettings initialization from environment variables."""
    settings = load_settings(FoundryLocalSettings, env_prefix="FOUNDRY_LOCAL_")

    assert settings["model"] == foundry_local_unit_test_env["FOUNDRY_LOCAL_MODEL"]


def test_foundry_local_settings_init_with_explicit_values() -> None:
    """Test FoundryLocalSettings initialization with explicit values."""
    settings = load_settings(
        FoundryLocalSettings,
        env_prefix="FOUNDRY_LOCAL_",
        model="custom-model-id",
    )

    assert settings["model"] == "custom-model-id"


@pytest.mark.parametrize("exclude_list", [["FOUNDRY_LOCAL_MODEL"]], indirect=True)
def test_foundry_local_settings_missing_model(foundry_local_unit_test_env: dict[str, str]) -> None:
    """Test FoundryLocalSettings when model is missing raises error."""
    with pytest.raises(SettingNotFoundError, match="Required setting 'model'"):
        load_settings(
            FoundryLocalSettings,
            env_prefix="FOUNDRY_LOCAL_",
            required_fields=["model"],
        )


def test_foundry_local_settings_explicit_overrides_env(foundry_local_unit_test_env: dict[str, str]) -> None:
    """Test that explicit values override environment variables."""
    settings = load_settings(FoundryLocalSettings, env_prefix="FOUNDRY_LOCAL_", model="override-model-id")

    assert settings["model"] == "override-model-id"
    assert settings["model"] != foundry_local_unit_test_env["FOUNDRY_LOCAL_MODEL"]


# Client Initialization Tests


def test_foundry_local_client_init(mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None) -> None:
    """Test FoundryLocalClient initialization with mocked manager."""
    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ):
        client = FoundryLocalClient(model="test-model-id")

        assert client.model == "test-model-id"
        assert client.manager is mock_foundry_local_manager
        assert isinstance(client, SupportsChatGetResponse)


def test_agent_accepts_foundry_local_client(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None
) -> None:
    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ):
        client = FoundryLocalClient(model="test-model-id")

    agent = Agent(client=client, instructions="test agent")
    assert agent.client is client


def test_foundry_local_client_get_response_uses_explicit_runtime_buckets() -> None:
    """Foundry Local should expose explicit runtime buckets instead of raw kwargs."""
    signature = inspect.signature(FoundryLocalClient.get_response)

    assert "client_kwargs" in signature.parameters
    assert "function_invocation_kwargs" in signature.parameters
    assert all(parameter.kind != inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def test_foundry_local_client_init_with_bootstrap_false(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None
) -> None:
    """Test FoundryLocalClient initialization with bootstrap=False."""
    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ) as mock_manager_class:
        FoundryLocalClient(model="test-model-id", bootstrap=False)

        mock_manager_class.assert_called_once_with(
            bootstrap=False,
            timeout=None,
        )


def test_foundry_local_client_init_with_timeout(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None
) -> None:
    """Test FoundryLocalClient initialization with custom timeout."""
    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ) as mock_manager_class:
        FoundryLocalClient(model="test-model-id", timeout=60.0)

        mock_manager_class.assert_called_once_with(
            bootstrap=True,
            timeout=60.0,
        )


def test_foundry_local_client_init_model_not_found(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None
) -> None:
    """Test FoundryLocalClient initialization when model is not found."""
    mock_foundry_local_manager.get_model_info.return_value = None

    with (
        patch(
            "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
            return_value=mock_foundry_local_manager,
        ),
        pytest.raises(ValueError, match="not found in Foundry Local"),
    ):
        FoundryLocalClient(model="unknown-model")


def test_foundry_local_client_uses_model_info_id(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None
) -> None:
    """Test that client uses the model ID from model_info, not the alias."""
    mock_model_info = MagicMock()
    mock_model_info.id = "resolved-model-id"
    mock_foundry_local_manager.get_model_info.return_value = mock_model_info

    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ):
        client = FoundryLocalClient(model="model-alias")

        assert client.model == "resolved-model-id"


def test_foundry_local_client_init_from_env(
    foundry_local_unit_test_env: dict[str, str],
    mock_foundry_local_manager: MagicMock,
    legacy_foundry_local_sdk: None,
) -> None:
    """Test FoundryLocalClient initialization using environment variables."""
    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ):
        client = FoundryLocalClient()

        assert client.model == foundry_local_unit_test_env["FOUNDRY_LOCAL_MODEL"]


def test_foundry_local_client_init_with_device(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None
) -> None:
    """Test FoundryLocalClient initialization with device parameter."""
    device = _legacy_device("CPU")
    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ):
        FoundryLocalClient(model="test-model-id", device=device)

        mock_foundry_local_manager.get_model_info.assert_called_once_with(
            "test-model-id",
            device=device,
        )
        mock_foundry_local_manager.download_model.assert_called_once_with(
            "test-model-id",
            device=device,
        )
        mock_foundry_local_manager.load_model.assert_called_once_with(
            "test-model-id",
            device=device,
        )


def test_foundry_local_client_init_model_not_found_with_device(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None
) -> None:
    """Test FoundryLocalClient error message includes device when model not found with device specified."""
    device = _legacy_device("GPU")
    mock_foundry_local_manager.get_model_info.return_value = None

    with (
        patch(
            "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
            return_value=mock_foundry_local_manager,
        ),
        pytest.raises(ValueError, match="unknown-model:GPU.*not found"),
    ):
        FoundryLocalClient(model="unknown-model", device=device)


def test_foundry_local_client_init_with_prepare_model_false(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None
) -> None:
    """Test FoundryLocalClient initialization with prepare_model=False skips download and load."""
    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ):
        FoundryLocalClient(model="test-model-id", prepare_model=False)

        mock_foundry_local_manager.download_model.assert_not_called()
        mock_foundry_local_manager.load_model.assert_not_called()


def test_foundry_local_client_init_calls_download_and_load(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None
) -> None:
    """Test FoundryLocalClient initialization calls download_model and load_model by default."""
    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ):
        FoundryLocalClient(model="test-model-id")

        mock_foundry_local_manager.download_model.assert_called_once_with(
            "test-model-id",
            device=None,
        )
        mock_foundry_local_manager.load_model.assert_called_once_with(
            "test-model-id",
            device=None,
        )


def _mock_modern_manager(model: MagicMock) -> MagicMock:
    manager = MagicMock()
    manager.urls = None
    manager.catalog.get_model.return_value = model
    manager.catalog.get_model_variant.return_value = None

    def start_web_service() -> None:
        manager.urls = ["http://127.0.0.1:53123"]

    manager.start_web_service.side_effect = start_web_service
    return manager


def test_foundry_local_client_modern_sdk_initializes_openai_transport() -> None:
    """The modern SDK uses its catalog and web service with the existing OpenAI transport."""
    model = MagicMock()
    model.id = "resolved-model-id"
    manager = _mock_modern_manager(model)

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "_FoundryLocalConfiguration") as config_class,
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        patch.object(foundry_local_client, "AsyncOpenAI") as async_openai_class,
    ):
        manager_class.instance = None
        manager_class.initialize.side_effect = lambda _: setattr(manager_class, "instance", manager)

        client = FoundryLocalClient(model="model-alias", timeout=30.0)

        config_class.assert_called_once_with(app_name="agent-framework")
        manager.catalog.get_model.assert_called_once_with("model-alias")
        manager.catalog.get_model_variant.assert_not_called()
        model.download.assert_called_once_with()
        model.load.assert_called_once_with()
        manager.start_web_service.assert_called_once_with()
        async_openai_class.assert_called_once_with(
            base_url="http://127.0.0.1:53123/v1",
            api_key="local",
            timeout=30.0,
        )
        assert client.model == "resolved-model-id"
        assert client.manager.endpoint == "http://127.0.0.1:53123/v1"
        assert client.manager.api_key == "local"


def test_foundry_local_client_modern_sdk_selects_device_variant() -> None:
    """A requested device selects the corresponding catalog model variant."""
    model = MagicMock()
    model.id = "model-alias-generic-gpu:1"
    cpu_variant = MagicMock()
    cpu_variant.id = "model-alias-generic-cpu:1"
    gpu_variant = MagicMock()
    gpu_variant.id = "model-alias-generic-gpu:1"
    model.variants = [cpu_variant, gpu_variant]
    manager = _mock_modern_manager(model)

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        patch.object(foundry_local_client, "AsyncOpenAI"),
    ):
        manager_class.instance = manager

        client = FoundryLocalClient(model="model-alias", device="gpu", prepare_model=False)

        model.select_variant.assert_called_once_with(gpu_variant)
        model.download.assert_not_called()
        model.load.assert_not_called()
        assert client.model == "model-alias-generic-gpu:1"


def test_foundry_local_client_modern_sdk_accepts_variant_id() -> None:
    """A variant ID falls back from alias lookup to the SDK's variant lookup."""
    model = MagicMock()
    model.id = "model-alias-generic-cpu:1"
    manager = _mock_modern_manager(model)
    manager.catalog.get_model.return_value = None
    manager.catalog.get_model_variant.return_value = model

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        patch.object(foundry_local_client, "AsyncOpenAI"),
    ):
        manager_class.instance = manager

        client = FoundryLocalClient(model="model-alias-generic-cpu:1", prepare_model=False)

        manager.catalog.get_model_variant.assert_called_once_with("model-alias-generic-cpu:1")
        assert client.model == "model-alias-generic-cpu:1"


def test_foundry_local_client_modern_sdk_accepts_variant_id_with_matching_device() -> None:
    """A variant ID can be combined with its device without selecting on a variant object."""
    model = MagicMock()
    model.id = "model-alias-generic-cpu:1"
    model.variants = [model]
    manager = _mock_modern_manager(model)
    manager.catalog.get_model.return_value = None
    manager.catalog.get_model_variant.return_value = model

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        patch.object(foundry_local_client, "AsyncOpenAI"),
    ):
        manager_class.instance = manager

        client = FoundryLocalClient(
            model="model-alias-generic-cpu:1",
            device="cpu",
            prepare_model=False,
        )

        model.select_variant.assert_not_called()
        assert client.model == "model-alias-generic-cpu:1"


def test_foundry_local_client_modern_sdk_retains_manager_helpers() -> None:
    """The modern manager compatibility view retains the old helper methods."""
    model = MagicMock()
    model.id = "model-alias"
    manager = _mock_modern_manager(model)

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        patch.object(foundry_local_client, "AsyncOpenAI"),
    ):
        manager_class.instance = manager

        client = FoundryLocalClient(model="model-alias", prepare_model=False)

        assert client.manager.list_catalog_models() == manager.catalog.list_models.return_value
        assert client.manager.catalog is manager.catalog
        client.manager.download_model("model-alias")
        client.manager.load_model("model-alias")
        assert manager.catalog.get_model.call_count == 3
        model.download.assert_called_once_with()
        model.load.assert_called_once_with()


def test_foundry_local_client_modern_sdk_bootstrap_false_requires_initialized_manager() -> None:
    """bootstrap=False does not initialize a new modern SDK manager."""
    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        pytest.raises(ValueError, match="bootstrap=False requires FoundryLocalManager"),
    ):
        manager_class.instance = None

        FoundryLocalClient(model="model-alias", bootstrap=False)

        manager_class.initialize.assert_not_called()


def test_foundry_local_client_modern_sdk_auto_device_uses_sdk_selection() -> None:
    """The modern SDK keeps its automatic variant choice when device is set to auto."""
    model = MagicMock()
    model.id = "model-alias-selected-by-sdk"
    manager = _mock_modern_manager(model)

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        patch.object(foundry_local_client, "AsyncOpenAI"),
    ):
        manager_class.instance = manager

        client = FoundryLocalClient(model="model-alias", device="auto", prepare_model=False)

        model.select_variant.assert_not_called()
        assert client.model == "model-alias-selected-by-sdk"


def test_foundry_local_client_modern_sdk_rejects_unavailable_device() -> None:
    """A device request without a matching catalog variant fails with a useful error."""
    model = MagicMock()
    model.id = "model-alias"
    model.variants = [MagicMock(id="model-alias-generic-cpu:1")]
    manager = _mock_modern_manager(model)

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        pytest.raises(ValueError, match="model-alias:GPU.*not found in Foundry Local"),
    ):
        manager_class.instance = manager

        FoundryLocalClient(model="model-alias", device="gpu")


def test_foundry_local_client_modern_sdk_rejects_missing_model() -> None:
    """An unknown alias or variant ID raises ValueError during client creation."""
    model = MagicMock()
    model.id = "unused-model"
    manager = _mock_modern_manager(model)
    manager.catalog.get_model.return_value = None
    manager.catalog.get_model_variant.return_value = None

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        pytest.raises(ValueError, match="missing-model.*not found in Foundry Local"),
    ):
        manager_class.instance = manager

        FoundryLocalClient(model="missing-model")


def test_foundry_local_client_modern_sdk_rejects_missing_compatibility_model() -> None:
    """The manager compatibility helper reports an unknown model as ValueError."""
    model = MagicMock()
    model.id = "model-alias"
    manager = _mock_modern_manager(model)

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        patch.object(foundry_local_client, "AsyncOpenAI"),
        pytest.raises(ValueError, match="missing-model.*not found in Foundry Local"),
    ):
        manager_class.instance = manager
        client = FoundryLocalClient(model="model-alias", prepare_model=False)
        manager.catalog.get_model.return_value = None
        manager.catalog.get_model_variant.return_value = None

        client.manager.get_model_info("missing-model")


def test_foundry_local_client_modern_sdk_requires_configuration_to_initialize() -> None:
    """An unavailable Configuration API produces an actionable initialization error."""
    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "_FoundryLocalConfiguration", None),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        pytest.raises(RuntimeError, match="does not provide Configuration"),
    ):
        manager_class.instance = None

        FoundryLocalClient(model="model-alias")


def test_foundry_local_client_modern_sdk_requires_initialized_manager_after_init() -> None:
    """Initialization that does not produce a singleton manager fails clearly."""
    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "_FoundryLocalConfiguration"),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        pytest.raises(RuntimeError, match="did not create a manager instance"),
    ):
        manager_class.instance = None

        FoundryLocalClient(model="model-alias")

        manager_class.initialize.assert_called_once()


def test_foundry_local_client_modern_sdk_requires_web_service_url() -> None:
    """The modern OpenAI transport requires the embedded service to return a URL."""
    model = MagicMock()
    model.id = "model-alias"
    manager = _mock_modern_manager(model)
    manager.start_web_service.side_effect = None

    with (
        patch.object(foundry_local_client, "_use_modern_foundry_local_sdk", True),
        patch.object(foundry_local_client, "FoundryLocalManager") as manager_class,
        pytest.raises(RuntimeError, match="did not provide an endpoint"),
    ):
        manager_class.instance = manager

        FoundryLocalClient(model="model-alias")


def test_foundry_local_client_legacy_path_converts_string_device(
    mock_foundry_local_manager: MagicMock, legacy_foundry_local_sdk: None, monkeypatch: Any
) -> None:
    """String device selectors map to the legacy SDK's enum values."""

    class LegacyDeviceType(Enum):
        CPU = "cpu"

    monkeypatch.setattr(foundry_local_client, "_LegacyDeviceType", LegacyDeviceType)
    with patch(
        "agent_framework_foundry_local._foundry_local_client.FoundryLocalManager",
        return_value=mock_foundry_local_manager,
    ):
        FoundryLocalClient(model="test-model-id", device="cpu")

    mock_foundry_local_manager.get_model_info.assert_called_once_with(
        "test-model-id",
        device=LegacyDeviceType.CPU,
    )

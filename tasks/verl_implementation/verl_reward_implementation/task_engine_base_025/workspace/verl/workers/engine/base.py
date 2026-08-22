
"""
The abstract base class defining the interface for model training engines.
"""
import logging
import os
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Generator, Optional

import torch
from tensordict import TensorDict

from verl.utils.device import get_device_name, get_vendor
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class BaseEngine:
    """
    Abstract base class defining the interface for model training engines. Interface is subject to
    change before release.

    Engine implementations must subclass BaseEngine and provide concrete behavior for all methods.
    """

    def initialize(self):
        """Instantiate or load the model, optimizer, and learning rate scheduler.

        Should prepare all components necessary for training or evaluation."""
        # Default no-op; subclasses must override to build model, optimizer, lr_scheduler.
        self.module = None
        self.optimizer = None
        self.lr_scheduler = None

    @property
    @abstractmethod
    def is_param_offload_enabled(self) -> bool:
        """Whether parameter offloading is enabled."""
        raise NotImplementedError(
            "Implement BaseEngine.is_param_offload_enabled for verl_reward_implementation"
        )

    @property
    @abstractmethod
    def is_optimizer_offload_enabled(self) -> bool:
        """Whether optimizer offloading is enabled."""
        raise NotImplementedError(
            "Implement BaseEngine.is_optimizer_offload_enabled for verl_reward_implementation"
        )

    def train_mode(self, **kwargs):
        """Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode"""
        return BaseEngineCtx(self, mode="train", **kwargs)

    def eval_mode(self, **kwargs):
        """Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode"""
        return BaseEngineCtx(self, mode="eval", **kwargs)

    def optimizer_zero_grad(self):
        """Zero the gradients of the optimizer."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        """Perform an optimization step using the optimizer."""
        if self.optimizer is not None:
            self.optimizer.step()

    def lr_scheduler_step(self):
        """Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s)."""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            # Try to extract the current learning rate
            try:
                if hasattr(self.lr_scheduler, "get_last_lr"):
                    return self.lr_scheduler.get_last_lr()
                elif hasattr(self.lr_scheduler, "get_lr"):
                    return self.lr_scheduler.get_lr()
            except Exception:
                pass
        return None

    def forward_backward_batch(
        self, data: TensorDict, loss_function: Callable, forward_only=False
    ) -> Any:
        """Perform a forward pass and optionally a backward pass on a batch of data.

        Args:
            data: The input data for the forward pass, typically containing tensors and metadata.
            loss_function: The loss function to optimize. See `verl.workers.roles.utils.losses` for examples.
            forward_only: If True, perform only the forward pass. If False, perform forward and backward pass.

        Returns:
            Any: The output of the forward pass, which can be used for loss computation or other purposes."""
        raise NotImplementedError(
            "Implement BaseEngine.forward_backward_batch for verl_reward_implementation"
        )

    def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
        """Perform a training step on a batch of data.

        Args:
            data: The input data for training, typically containing tensors and metadata.
            loss_function: A function that computes the loss and metrics given a batch and predictions.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the aggregated training metrics for the batch."""
        # Default implementation: delegate to forward_backward_batch.
        return self.forward_backward_batch(data, loss_function, forward_only=False)

    def infer_batch(
        self, data: TensorDict, loss_function: Optional[Callable] = None
    ) -> Any:
        """Perform inference on a batch of data.

        Args:
            data: The input data for inference, typically containing tensors and metadata.

        Returns:
            Any: The output of the inference, which can be used for predictions or other purposes."""
        # Default implementation: delegate to forward_backward_batch with forward_only=True.
        return self.forward_backward_batch(data, loss_function, forward_only=True)

    def get_per_tensor_param(
        self,
    ) -> tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]:
        """Get a generator that yields per-tensor parameters and optional peft config.

        Returns:
            Generator[tuple[str, torch.Tensor]]: A generator that yields tuples of parameter names and tensors.
            Optional[dict]: Optional peft config."""
        # Default: iterate through model's named_parameters.
        if self.module is None:
            return iter([]), None

        def _gen():
            for name, param in self.module.named_parameters():
                yield name, param

        return _gen(), None

    def get_data_parallel_size(self):
        """Return the data-parallel world size (default 1)."""
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    def get_data_parallel_rank(self):
        """Return the data-parallel rank (default 0)."""
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def get_data_parallel_group(self):
        """Return the data-parallel process group (default WORLD)."""
        if torch.distributed.is_initialized():
            return torch.distributed.group.WORLD
        return None

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
            grad: If True, move the gradient buffer."""
        if model and self.module is not None:
            self.module.to(device)
        if optimizer and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
            **kwargs: Arbitrary keyword arguments."""
        os.makedirs(local_path, exist_ok=True)
        ckpt_path = os.path.join(local_path, f"checkpoint_{global_step}.pt")
        state = {"global_step": global_step}
        if self.module is not None:
            state["model_state_dict"] = self.module.state_dict()
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        torch.save(state, ckpt_path)

        # Optionally clean up old checkpoints
        if max_ckpt_to_keep is not None:
            ckpt_files = sorted(
                [f for f in os.listdir(local_path) if f.startswith("checkpoint_") and f.endswith(".pt")]
            )
            while len(ckpt_files) > max_ckpt_to_keep:
                oldest = ckpt_files.pop(0)
                os.remove(os.path.join(local_path, oldest))

        logger.info(f"Saved checkpoint to {ckpt_path}")

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        del_local_after_load: bool = True,
        **kwargs,
    ) -> None:
        """Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
            **kwargs: Arbitrary keyword arguments."""
        ckpt_path = local_path
        if not os.path.exists(ckpt_path):
            # Try globbing for a checkpoint file
            ckpt_files = sorted(
                [f for f in os.listdir(local_path) if f.startswith("checkpoint_") and f.endswith(".pt")]
            )
            if ckpt_files:
                ckpt_path = os.path.join(local_path, ckpt_files[-1])
            else:
                raise FileNotFoundError(f"No checkpoint found at {local_path}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if self.module is not None and "model_state_dict" in state:
            self.module.load_state_dict(state["model_state_dict"], strict=False)
        if self.optimizer is not None and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in state:
            self.lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {ckpt_path}")

    def is_mp_src_rank_with_outputs(self):
        """Whether the current rank is the first rank in model parallel group that contains model outputs"""
        # Default: all ranks have outputs (no model parallelism).
        return True

    def disable_adapter(self) -> ContextManager:
        """Disable all adapters temporarily under the context in the model for LoRA"""
        # Default: no-op context. Subclasses with LoRA should override.
        if self.module is not None and hasattr(self.module, "disable_adapter"):
            return self.module.disable_adapter()
        return nullcontext()


class BaseEngineCtx:
    """Base Engine context that handles load and offload of model/optimizer when
    entering/exiting train or eval mode."""

    def __init__(self, engine: BaseEngine, mode, **kwargs):
        """Base Engine context that handles load and offload

        Args:
            engine: The engine instance to manage.
            mode: Either "train" or "eval".
            **kwargs: Additional options (disable_auto_offload, zero_grad_on_exit, etc.)
        """
        self.engine = engine
        self.mode = mode
        self.disable_auto_offload = kwargs.get("disable_auto_offload", False)
        self.zero_grad_on_exit = kwargs.get("zero_grad_on_exit", True)
        self._prev_mode = None

    def _context_switch(self, device):
        """Switch the engine to the specified device.

        Args:
            device: Target device ("cpu" or accelerator device name).
        """
        if device == "cpu":
            # Offload: move model and optimizer to CPU
            self.engine.to(
                device="cpu",
                model=self.engine.is_param_offload_enabled,
                optimizer=self.engine.is_optimizer_offload_enabled,
                grad=self.engine.is_param_offload_enabled,
            )
        else:
            # Load: move model and optimizer to accelerator
            self.engine.to(
                device=device,
                model=self.engine.is_param_offload_enabled,
                optimizer=self.engine.is_optimizer_offload_enabled,
                grad=self.engine.is_param_offload_enabled,
            )

    def __enter__(self):
        # Store previous mode
        self._prev_mode = getattr(self.engine, "mode", None)
        self.engine.mode = self.mode

        # Load model/optimizer to GPU if offloading is enabled and not disabled
        if not self.disable_auto_offload:
            device_name = get_device_name()
            self._context_switch(device_name)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Offload model/optimizer back to CPU if offloading is enabled and not disabled
        if not self.disable_auto_offload:
            self._context_switch("cpu")

        # Restore previous mode
        self.engine.mode = self._prev_mode
        return False


class EngineRegistry:
    """
    A registry for managing and instantiating different types of training engines.

    This class uses a dictionary to store engine classes, mapping a string key to each class.
    It provides a decorator `register` to add new engines to the registry and a `new` method
    to create an instance of a registered engine.
    """

    _engines = {}

    @classmethod
    def register(
        cls,
        model_type: str,
        backend: list[str] | str,
        device: list[str] | str = "cuda",
        vendor: list[str] | str | None = None,
    ):
        """A class method decorator that registers an engine class with a given key.

        This allows for dynamic instantiation of engine classes by their registered key.

        Args:
            model_type (str): The type of the model
            backend (list[str] | str): The backend to use for the model type
            device (list[str] | str): The device type (e.g., "cuda", "npu", "cpu") this engine supports,
                default is "cuda"
            vendor (list[str] | str | None): The hardware vendor (e.g., "nvidia", "metax") this engine
                supports. If None, the engine is registered as the default for the device type.

        Returns:
            A decorator function that takes an engine class and registers it.
        """
        # Normalize single values to lists
        if isinstance(backend, str):
            backend = [backend]
        if isinstance(device, str):
            device = [device]
        if isinstance(vendor, str):
            vendor = [vendor]

        def decorator(engine_cls):
            for b in backend:
                for d in device:
                    key = (model_type, b, d)
                    if vendor is not None:
                        for v in vendor:
                            cls._engines[(model_type, b, d, v)] = engine_cls
                    cls._engines[key] = engine_cls
            return engine_cls

        return decorator

    @classmethod
    def get_engine_cls(cls, model_type: str, backend: str):
        """Look up the registered engine class for the given model_type and backend.

        Args:
            model_type: The model type (e.g. "language_model", "value_model").
            backend: The backend name (e.g. "fsdp", "fsdp2", "megatron", "torchtitan").

        Returns:
            The engine class, or raises ValueError if not found.
        """
        current_device = get_device_name()
        current_vendor = get_vendor()

        # Try exact match: (model_type, backend, device, vendor)
        key = (model_type, backend, current_device, current_vendor)
        if key in cls._engines:
            return cls._engines[key]

        # Try without vendor: (model_type, backend, device)
        key = (model_type, backend, current_device)
        if key in cls._engines:
            return cls._engines[key]

        # Fallback: try any device/vendor combination
        for (mt, be, dev), engine_cls in cls._engines.items():
            if mt == model_type and be == backend and dev == current_device:
                return engine_cls

        raise ValueError(
            f"No engine registered for model_type={model_type}, backend={backend}, "
            f"device={current_device}. Available: {list(cls._engines.keys())}"
        )

    @classmethod
    def new(cls, model_type, backend, *args, **kwargs):
        """Function to create a new training engine instance based on the provided config.

        Args:
            model_type: The model type (e.g. "language_model").
            backend: The backend name (e.g. "fsdp").
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            engine: An instance of the training engine corresponding to the config.

        Raises:
            ValueError: If no engine is registered for the given model_type and backend.
        """
        engine_cls = cls.get_engine_cls(model_type, backend)
        return engine_cls(*args, **kwargs)

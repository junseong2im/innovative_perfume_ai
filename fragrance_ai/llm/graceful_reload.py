# fragrance_ai/llm/graceful_reload.py
"""
Graceful Model Reload System
모델 교체 시 new worker warm-up → 트래픽 스위칭
"""

import time
import logging
import threading
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Reload States
# ============================================================================

class ReloadState(str, Enum):
    """Model reload states"""
    IDLE = "idle"
    WARMING_UP = "warming_up"
    READY = "ready"
    SWITCHING = "switching"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReloadStatus:
    """Status of model reload operation"""
    model_name: str
    state: ReloadState
    progress_percent: float
    message: str
    started_at: str
    completed_at: Optional[str]
    error: Optional[str]


# ============================================================================
# Graceful Reload Manager
# ============================================================================

class GracefulReloadManager:
    """Manages graceful model reloads with zero downtime"""

    def __init__(self):
        self.old_models: Dict[str, Any] = {}
        self.new_models: Dict[str, Any] = {}
        self.active_models: Dict[str, Any] = {}  # Currently serving traffic
        self.reload_status: Dict[str, ReloadStatus] = {}
        self._lock = threading.Lock()

    def start_reload(
        self,
        model_name: str,
        loader_func: Callable[[], Any],
        warmup_func: Optional[Callable[[Any], bool]] = None,
        warmup_iterations: int = 5
    ) -> ReloadStatus:
        """
        Start graceful reload of a model

        Args:
            model_name: Model identifier
            loader_func: Function to load new model
            warmup_func: Function to warm up model (returns True if successful)
            warmup_iterations: Number of warmup iterations

        Returns:
            ReloadStatus
        """
        with self._lock:
            # Check if reload already in progress
            if model_name in self.reload_status:
                status = self.reload_status[model_name]
                if status.state in [ReloadState.WARMING_UP, ReloadState.SWITCHING]:
                    logger.warning(f"Reload already in progress for {model_name}")
                    return status

            # Initialize reload status
            status = ReloadStatus(
                model_name=model_name,
                state=ReloadState.WARMING_UP,
                progress_percent=0.0,
                message="Starting model reload",
                started_at=datetime.utcnow().isoformat(),
                completed_at=None,
                error=None
            )
            self.reload_status[model_name] = status

            # Store old model for fallback
            if model_name in self.active_models:
                self.old_models[model_name] = self.active_models[model_name]

        # Start reload in background thread
        thread = threading.Thread(
            target=self._reload_worker,
            args=(model_name, loader_func, warmup_func, warmup_iterations),
            daemon=True
        )
        thread.start()

        logger.info(f"Started graceful reload for {model_name}")
        return status

    def _reload_worker(
        self,
        model_name: str,
        loader_func: Callable[[], Any],
        warmup_func: Optional[Callable[[Any], bool]],
        warmup_iterations: int
    ):
        """Background worker for model reload"""
        try:
            # Phase 1: Load new model
            logger.info(f"[{model_name}] Phase 1: Loading new model...")
            self._update_status(model_name, ReloadState.WARMING_UP, 10.0, "Loading new model")

            new_model = loader_func()

            if new_model is None:
                raise ValueError("Model loader returned None")

            with self._lock:
                self.new_models[model_name] = new_model

            # Phase 2: Warm up new model
            logger.info(f"[{model_name}] Phase 2: Warming up new model...")
            self._update_status(model_name, ReloadState.WARMING_UP, 30.0, "Warming up new model")

            if warmup_func:
                for i in range(warmup_iterations):
                    progress = 30.0 + (i + 1) * (50.0 / warmup_iterations)
                    self._update_status(
                        model_name,
                        ReloadState.WARMING_UP,
                        progress,
                        f"Warmup iteration {i+1}/{warmup_iterations}"
                    )

                    success = warmup_func(new_model)
                    if not success:
                        raise ValueError(f"Warmup failed at iteration {i+1}")

                    time.sleep(0.1)  # Small delay between warmups

            # Phase 3: Switch traffic to new model
            logger.info(f"[{model_name}] Phase 3: Switching traffic...")
            self._update_status(model_name, ReloadState.SWITCHING, 80.0, "Switching traffic")

            with self._lock:
                self.active_models[model_name] = self.new_models[model_name]

            # Small delay to let in-flight requests complete
            time.sleep(0.5)

            # Phase 4: Cleanup old model
            logger.info(f"[{model_name}] Phase 4: Cleanup...")
            self._update_status(model_name, ReloadState.SWITCHING, 90.0, "Cleaning up old model")

            with self._lock:
                if model_name in self.old_models:
                    old_model = self.old_models[model_name]
                    # Cleanup old model (release memory, etc.)
                    del self.old_models[model_name]
                    del old_model

                if model_name in self.new_models:
                    del self.new_models[model_name]

            # Completed
            self._update_status(
                model_name,
                ReloadState.COMPLETED,
                100.0,
                "Reload completed successfully",
                completed=True
            )

            logger.info(f"[{model_name}] Graceful reload completed successfully")

        except Exception as e:
            error_msg = f"Reload failed: {str(e)}"
            logger.error(f"[{model_name}] {error_msg}", exc_info=True)

            # Rollback to old model if available
            with self._lock:
                if model_name in self.old_models:
                    logger.warning(f"[{model_name}] Rolling back to old model")
                    self.active_models[model_name] = self.old_models[model_name]
                    del self.old_models[model_name]

                # Cleanup failed new model
                if model_name in self.new_models:
                    del self.new_models[model_name]

            self._update_status(
                model_name,
                ReloadState.FAILED,
                0.0,
                error_msg,
                completed=True,
                error=error_msg
            )

    def _update_status(
        self,
        model_name: str,
        state: ReloadState,
        progress: float,
        message: str,
        completed: bool = False,
        error: Optional[str] = None
    ):
        """Update reload status"""
        with self._lock:
            if model_name in self.reload_status:
                status = self.reload_status[model_name]
                status.state = state
                status.progress_percent = progress
                status.message = message
                status.error = error

                if completed:
                    status.completed_at = datetime.utcnow().isoformat()

    def get_reload_status(self, model_name: str) -> Optional[ReloadStatus]:
        """Get reload status for a model"""
        with self._lock:
            return self.reload_status.get(model_name)

    def get_active_model(self, model_name: str) -> Optional[Any]:
        """Get currently active model serving traffic"""
        with self._lock:
            return self.active_models.get(model_name)

    def set_active_model(self, model_name: str, model: Any):
        """Manually set active model"""
        with self._lock:
            self.active_models[model_name] = model
            logger.info(f"Set active model: {model_name}")

    def is_reload_in_progress(self, model_name: str) -> bool:
        """Check if reload is in progress"""
        with self._lock:
            if model_name not in self.reload_status:
                return False

            status = self.reload_status[model_name]
            return status.state in [ReloadState.WARMING_UP, ReloadState.SWITCHING]


# ============================================================================
# Global Reload Manager Instance
# ============================================================================

_reload_manager: Optional[GracefulReloadManager] = None


def get_reload_manager() -> GracefulReloadManager:
    """Get global reload manager instance"""
    global _reload_manager
    if _reload_manager is None:
        _reload_manager = GracefulReloadManager()
    return _reload_manager


# ============================================================================
# Convenience Functions
# ============================================================================

def reload_model(
    model_name: str,
    loader_func: Callable[[], Any],
    warmup_func: Optional[Callable[[Any], bool]] = None,
    warmup_iterations: int = 5
) -> ReloadStatus:
    """
    Convenience function to reload a model

    Args:
        model_name: Model identifier
        loader_func: Function to load new model
        warmup_func: Function to warm up model
        warmup_iterations: Number of warmup iterations

    Returns:
        ReloadStatus
    """
    manager = get_reload_manager()
    return manager.start_reload(
        model_name=model_name,
        loader_func=loader_func,
        warmup_func=warmup_func,
        warmup_iterations=warmup_iterations
    )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'ReloadState',
    'ReloadStatus',
    'GracefulReloadManager',
    'get_reload_manager',
    'reload_model'
]

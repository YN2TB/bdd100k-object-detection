"""Decoded-image caching that preserves fresh inputs for random augmentation."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable


class DecodedImageCache:
    """Preload unique decoded inputs, refusing the host RAM ceiling."""

    def __init__(self, keys: Iterable[Any], decode: Callable,
                 memory_percent: Callable[[], float] | None = None):
        if memory_percent is None:
            import psutil
            memory_percent = lambda: psutil.virtual_memory().percent
        self.values = {}
        self.peak_ram_percent = float(memory_percent())
        for key in dict.fromkeys(keys):
            if self.peak_ram_percent >= 75:
                raise RuntimeError('RAM cache reached the 75% host memory ceiling')
            self.values[key] = decode(key)
            self.peak_ram_percent = max(self.peak_ram_percent, float(memory_percent()))
        if self.peak_ram_percent >= 75:
            raise RuntimeError('RAM cache reached the 75% host memory ceiling')

    def __call__(self, key: Any) -> Any:
        return deepcopy(self.values[key])

    def __len__(self) -> int:
        return len(self.values)


def cached_rfdetr_datamodule(base):
    """Cache native decoded images without caching stochastic transforms."""
    class CachedDataModule(base):
        _bddcv_cached = False

        def setup(self, stage=None):
            super().setup(stage)
            if stage == 'fit' and not self._bddcv_cached:
                for dataset in (self._dataset_train, self._dataset_val):
                    dataset._decode_image = DecodedImageCache(dataset.ids, dataset._decode_image)
                self._bddcv_cached = True
    return CachedDataModule

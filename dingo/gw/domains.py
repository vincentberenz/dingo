from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, lru_cache, wraps
from typing import Optional

import numpy as np
import torch
from multipledispatch import dispatch
from typing_extensions import override


@dataclass
class DomainParameters:
    delta_f: float
    f_min: float
    f_max: float
    f_ref: float
    delta_t: float


class Domain(ABC):
    """Defines the physical domain on which the data of interest live.

    This includes a specification of the bins or points,
    and a few additional properties associated with the data.
    """

    @abstractmethod
    def get_parameters(self) -> DomainParameters:
        raise NotImplementedError(
            "Subclasses of Domain must implement the get_parameters method."
        )

    @abstractmethod
    def __len__(self):
        """Number of bins or points in the domain"""
        raise NotImplementedError(
            "Subclasses of Domain of must implement __len__ method."
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Array of bins in the domain"""
        raise NotImplementedError(
            "Subclasses of Domain must implement __call__ method."
        )

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError("Subclasses of Domain must implement update method.")

    @abstractmethod
    def time_translate_data(self, data, dt) -> np.ndarray:
        """Time translate strain data by dt seconds."""
        raise NotImplementedError(
            "Subclasses of Domain must implement time_translate_data method."
        )

    @property
    @abstractmethod
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution"""
        # FIXME: For this to make sense, it assumes knowledge about how the domain is used in conjunction
        #  with (waveform) data, whitening and adding noise. Is this the best place to define this?
        raise NotImplementedError(
            "Subclasses of Domain must implement noise_std property."
        )

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """The sampling rate of the data [Hz]."""
        raise NotImplementedError(
            "Subclasses of Domain must implement sampling_rate property."
        )

    @property
    @abstractmethod
    def f_max(self) -> float:
        """The maximum frequency [Hz] is set to half the sampling rate."""
        raise NotImplementedError("Subclasses of Domain must implement f_max property.")

    @property
    @abstractmethod
    def duration(self) -> float:
        """Waveform duration in seconds."""
        raise NotImplementedError(
            "Subclasses of Domain must implement duration property."
        )

    @property
    @abstractmethod
    def min_idx(self) -> int:
        raise NotImplementedError(
            "Subclasses of Domain must implement min_idx property."
        )

    @property
    @abstractmethod
    def max_idx(self) -> int:
        raise NotImplementedError(
            "Subclasses of Domain must implement max_idx property."
        )


class _SampleFrequencies:

    def __init__(self, f_min: float, f_max: float, delta_f: float) -> None:
        self._len = int(f_max / delta_f) + 1
        self._f_min = f_min
        self._f_max = f_max
        self._sample_frequencies = np.linspace(
            0.0, f_max, num=self._len, endpoint=True, dtype=np.float32
        )

    def get(self):
        return self._sample_frequencies

    def __len__(self) -> int:
        return self._len

    @cached_property
    def frequency_mask(self):
        return self._sample_frequencies > self._f_min

    @cached_property
    def _sample_frequency_torch(self):
        return torch.linspace(0.0, self.f_max, steps=len(self), dtype=torch.float32)

    @cached_property
    def _sample_frequency_torch_cuda(self):
        return self._sample_frequencies_torch.to("cuda")


def _reset_sf(func):
    @wraps(func)
    def wrapper(instance, value):
        func(instance, value)
        self._sample_frequences = _SampleFrequencies(instance._f_max, instance._delta_f)

    return wrapper


class FrequencyDomain(Domain):
    """Defines the physical domain on which the data of interest live.

    The frequency bins are assumed to be uniform between [0, f_max]
    with spacing delta_f.
    Given a finite length of time domain data, the Fourier domain data
    starts at a frequency f_min and is zero below this frequency.
    window_kwargs specify windowing used for FFT to obtain FD data from TD
    data in practice.
    """

    def __init__(
        self,
        f_min: float,
        f_max: float,
        delta_f: float,
        window_factor: Optional[float] = None,
    ):
        self._f_min = f_min
        self._f_max = f_max
        self._delta_f = delta_f
        self._window_factor = window_factor

        self._sample_frequences = _SampleFrequencies(f_min, f_max, delta_f)

    @override
    def get_parameters(self, f_start: Optional[float] = None) -> DomainParameters:
        return DomainParameters(
            delta_f=self.delta_f,
            f_max=self.f_max,
            f_min=f_start if f_start is not None else self.f_min,
            delta_t=0.5 / self.f_max,
        )

    @override
    def update(self, f_min: Optional[float], f_max: Optional[float]) -> None:
        """
        Update the domain range.
        Both values must be in the original domain interval.
        None values have no effect on the interval.

        Parameters
        ----------
        f_min
          new minimum value. Must be in the original domain interval.
        f_max
          new minimum value.

        Raises
        ------
        ValueError if f_min or f_max is not in the original interval

        """

        def _in_range(self, f: float) -> bool:
            return f >= self._f_min and f <= self._f_max

        def _get_value(self, f: Optional[float]) -> Optional[float]:
            if f is None:
                return None
            if not self._in_range(f):
                raise ValueError(
                    f"Cannot update FrequencyDomain range, {f} not in interval [{self._f_min, self._f_max}]"
                )
            return f

        f_min = _get_value(self, f_min)
        self._f_min = f_min if f_min is not None else self._f_min
        f_max = _get_value(self, f_max)
        self._f_max = f_max if f_max is not None else self._f_max

    @override
    def time_translate_data(self, data, dt):
        """
        Time translate frequency-domain data by dt. Time translation corresponds (in
        frequency domain) to multiplication by

        .. math::
            \exp(-2 \pi i \, f \, dt).

        This method allows for multiple batch dimensions. For torch.Tensor data,
        allow for either a complex or a (real, imag) representation.

        Parameters
        ----------
        data : array-like (numpy, torch)
            Shape (B, C, N), where

                - B corresponds to any dimension >= 0,
                - C is either absent (for complex data) or has dimension >= 2 (for data
                  represented as real and imaginary parts), and
                - N is either len(self) or len(self)-self.min_idx (for truncated data).

        dt : torch tensor, or scalar (if data is numpy)
            Shape (B)

        Returns
        -------
        Array-like of the same form as data.
        """
        f = self._get_sample_frequencies_astype(data)
        if isinstance(data, np.ndarray):
            # Assume numpy arrays un-batched, since they are only used at train time.
            phase_shift = 2 * np.pi * dt * f
        elif isinstance(data, torch.Tensor):
            # Allow for possible multiple "batch" dimensions (e.g., batch + detector,
            # which might have independent time shifts).
            phase_shift = 2 * np.pi * torch.einsum("...,i", dt, f)
        else:
            raise NotImplementedError(
                f"Time translation not implemented for data of " "type {data}."
            )
        return self.add_phase(data, phase_shift)

    @override
    def __len__(self):
        """Number of frequency bins in the domain [0, f_max]"""
        return int(self.f_max / self.delta_f) + 1

    @override
    def __call__(self) -> np.ndarray:
        """Array of uniform frequency bins in the domain [0, f_max]"""
        return self.sample_frequencies

    @property
    @override
    def min_idx(self):
        return round(self._f_min / self._delta_f)

    @property
    @override
    def max_idx(self):
        return round(self._f_max / self._delta_f)

    @property
    @override
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        TODO: This description makes some assumptions that need to be clarified.
        Windowing of TD data; tapering window has a slope -> reduces power only for noise,
        but not for the signal which is in the main part unaffected by the taper
        """
        if self._window_factor is None:
            raise ValueError("Window factor needs to be set for noise_std.")
        return np.sqrt(self._window_factor) / np.sqrt(4.0 * self._delta_f)

    @property
    @override
    def f_max(self) -> float:
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        return self._f_max

    @f_max.setter
    @_reset_sf
    def f_max(self, value):
        self._f_max = float(value)

    @property
    @override
    def f_min(self) -> float:
        """The minimum frequency [Hz]."""
        return self._f_min

    @f_min.setter
    @_reset_sf
    def f_min(self, value):
        self._f_min = float(value)

    @property
    @override
    def delta_f(self) -> float:
        """The frequency spacing of the uniform grid [Hz]."""
        return self._delta_f

    @delta_f.setter
    @_reset_sf
    def delta_f(self, value):
        self._delta_f = float(value)

    @property
    @override
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return 1.0 / self.delta_f

    @property
    @override
    def sampling_rate(self) -> float:
        return 2.0 * self.f_max

    @property
    @override
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        return {
            "type": "FrequencyDomain",
            "f_min": self.f_min,
            "f_max": self.f_max,
            "delta_f": self.delta_f,
            "window_factor": self.window_factor,
        }

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        if not any(
            [
                getattr(self, attr) != getattr(other, attr)
                for attr in ("_f_min", "_f_max", "_delta_f", "_window_factor")
            ]
        ):
            return False
        return True

    def update_data(self, data: np.ndarray, axis: int = -1, low_value: float = 0.0):
        """
        Adjusts data to be compatible with the domain:

        * Below f_min, it sets the data to low_value (typically 0.0 for a waveform,
          but for a PSD this might be a large value).
        * Above f_max, it truncates the data array.

        Parameters
        ----------
        data : np.ndarray
            Data array
        axis : int
            Which data axis to apply the adjustment along.
        low_value : float
            Below f_min, set the data to this value.

        Returns
        -------
        np.ndarray
            The new data array.
        """
        sl = [slice(None)] * data.ndim

        # First truncate beyond f_max.
        sl[axis] = slice(0, self.max_idx + 1)
        data = data[tuple(sl)]

        # Set data value below f_min to low_value.
        sl[axis] = slice(0, self.min_idx)
        data[tuple(sl)] = low_value

        return data

    @dispatch(np.ndarray, np.ndarray)
    def add_phase(data: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Add a (frequency-dependent) phase to a frequency series for numpy arrays.
        Assumes data is a complex frequency series.

        Convention: the phase phi(f) is defined via exp(- 1j * phi(f)).

        Parameters
        ----------
        data : np.ndarray
        phase : np.ndarray

        Returns
        -------
        New array of the same shape as data.
        """
        if not np.iscomplexobj(data):
            raise TypeError("Numpy data must be a complex array.")
        return data * np.exp(-1j * phase)

    @dispatch(torch.Tensor, torch.Tensor)  # type: ignore
    def add_phase(  # type: ignore
        data: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Add a (frequency-dependent) phase to a frequency series for torch tensors.
        Handles both complex tensors and real-imaginary part representation.

        Convention: the phase phi(f) is defined via exp(- 1j * phi(f)).

        Parameters
        ----------
        data : torch.Tensor
        phase : torch.Tensor

        Returns
        -------
        New tensor of the same shape as data.
        """
        if torch.is_complex(data):
            # Expand the trailing batch dimensions to allow for broadcasting.
            while phase.dim() < data.dim():
                phase = phase[..., None, :]
            return data * torch.exp(-1j * phase)
        else:
            # The first two components of the second last index should be the real
            # and imaginary parts of the data. Remaining components correspond to,
            # e.g., the ASD. The "-1" below accounts for this extra dimension when
            # broadcasting.
            while phase.dim() < data.dim() - 1:
                phase = phase[..., None, :]

            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)
            result = torch.empty_like(data)
            result[..., 0, :] = (
                data[..., 0, :] * cos_phase + data[..., 1, :] * sin_phase
            )
            result[..., 1, :] = (
                data[..., 1, :] * cos_phase - data[..., 0, :] * sin_phase
            )
            if data.shape[-2] > 2:
                result[..., 2:, :] = data[..., 2:, :]
            return result

    def __getitem__(self, idx):
        """Slice of uniform frequency grid."""
        sample_frequencies = self.__call__()
        return sample_frequencies[idx]

    @property
    def sample_frequencies(self):
        return self._sample_frequences.get()

    @property
    def _sample_frequencies_torch(self):
        if self._sample_frequencies_torch is None:
            num_bins = len(self)
            self._sample_frequencies_torch = torch.linspace(
                0.0, self.f_max, steps=num_bins, dtype=torch.float32
            )
        return self._sample_frequencies_torch

    @property
    def _sample_frequencies_torch_cuda(self):
        if self._sample_frequencies_torch_cuda is None:
            self._sample_frequencies_torch_cuda = self.sample_frequencies_torch.to(
                "cuda"
            )
        return self._sample_frequencies_torch_cuda

    def _get_sample_frequencies_astype(self, data):
        """
        Returns a 1D frequency array compatible with the last index of data array.

        Decides whether array is numpy or torch tensor (and cuda vs cpu), and whether it
        contains the leading zeros below f_min.

        Parameters
        ----------
        data : Union[np.array, torch.Tensor]
            Sample data

        Returns
        -------
        frequency array compatible with last index
        """
        # Type
        if isinstance(data, np.ndarray):
            f = self.sample_frequencies
        elif isinstance(data, torch.Tensor):
            if data.is_cuda:
                f = self._sample_frequencies_torch_cuda
            else:
                f = self._sample_frequencies_torch
        else:
            raise TypeError("Invalid data type. Should be np.array or torch.Tensor.")

        # Whether to include zeros below f_min
        if data.shape[-1] == len(self) - self.min_idx:
            f = f[self.min_idx :]
        elif data.shape[-1] != len(self):
            raise TypeError(
                f"Data with {data.shape[-1]} frequency bins is "
                f"incompatible with domain."
            )

        return f

    @property
    def frequency_mask(self) -> np.ndarray:
        """Mask which selects frequency bins greater than or equal to the
        starting frequency"""
        return self._sample_frequences.frequency_mask

    @property
    def frequency_mask_length(self) -> int:
        """Number of samples in the subdomain domain[frequency_mask]."""
        mask = self.frequency_mask
        return len(np.flatnonzero(np.asarray(mask)))

    # Vincent: This does not seem to be used (?)
    # @property
    # def window_factor(self):
    #    return self._window_factor
    #
    # @window_factor.setter
    # def window_factor(self, value):
    #    """Set self._window_factor and clear cache of self.noise_std."""
    #    self._window_factor = float(value)


class TimeDomain(Domain):
    """Defines the physical time domain on which the data of interest live.

    The time bins are assumed to be uniform between [0, duration]
    with spacing 1 / sampling_rate.
    window_factor is used to compute noise_std().
    """

    def __init__(self, time_duration: float, sampling_rate: float):
        self._time_duration = time_duration
        self._sampling_rate = sampling_rate

    @override
    def update(self) -> None:
        raise NotImplementedError("TimeDomain does not support update")

    @lru_cache()
    def __len__(self):
        """Number of time bins given duration and sampling rate"""
        return int(self._time_duration * self._sampling_rate)

    @lru_cache()
    def __call__(self) -> np.ndarray:
        """Array of uniform times at which data is sampled"""
        num_bins = self.__len__()
        return np.linspace(
            0.0,
            self._time_duration,
            num=num_bins,
            endpoint=False,
            dtype=np.float32,
        )

    @property
    def delta_t(self) -> float:
        """The size of the time bins"""
        return 1.0 / self._sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t: float):
        self._sampling_rate = 1.0 / delta_t

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        return 1.0 / np.sqrt(2.0 * self.delta_t)

    def time_translate_data(self, data, dt) -> np.ndarray:
        raise NotImplementedError

    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        return self._sampling_rate / 2.0

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return self._time_duration

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def min_idx(self) -> int:
        return 0

    @property
    def max_idx(self) -> int:
        return round(self._time_duration * self._sampling_rate)

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        return {
            "type": "TimeDomain",
            "time_duration": self._time_duration,
            "sampling_rate": self._sampling_rate,
        }


class PCADomain(Domain):
    """TODO"""

    # Not super important right now
    # FIXME: Should this be defined for FD or TD bases or both?
    # Nrb instead of Nf

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        # FIXME
        return np.sqrt(self.window_factor) / np.sqrt(4.0 * self.delta_f)


def build_domain(settings: dict) -> Domain:
    """
    Instantiate a domain class from settings.

    Parameters
    ----------
    settings : dict
        Dicionary with 'type' key denoting the type of domain, and keys corresponding
        to the kwargs needed to construct the Domain.

    Returns
    -------
    A Domain instance of the correct type.
    """
    if "type" not in settings:
        raise ValueError(
            f'Domain settings must include a "type" key. Settings included '
            f"the keys {settings.keys()}."
        )

    # The settings other than 'type' correspond to the kwargs of the Domain constructor.
    kwargs = {k: v for k, v in settings.items() if k != "type"}
    if settings["type"] in ["FrequencyDomain", "FD"]:
        return FrequencyDomain(**kwargs)
    elif settings["type"] == ["TimeDomain", "TD"]:
        return TimeDomain(**kwargs)
    else:
        raise NotImplementedError(f'Domain {settings["name"]} not implemented.')


def build_domain_from_model_metadata(model_metadata) -> Domain:
    """
    Instantiate a domain class from settings of model.

    Parameters
    ----------
    model_metadata: dict
        model metadata containing information to build the domain
        typically obtained from the model.metadata attribute

    Returns
    -------
    A Domain instance of the correct type.
    """
    domain = build_domain(model_metadata["dataset_settings"]["domain"])
    if "domain_update" in model_metadata["train_settings"]["data"]:
        domain.update(model_metadata["train_settings"]["data"]["domain_update"])
    domain.window_factor = get_window_factor(
        model_metadata["train_settings"]["data"]["window"]
    )
    return domain


if __name__ == "__main__":
    kwargs = {"f_min": 20, "f_max": 2048, "delta_f": 0.125}
    domain = FrequencyDomain(**kwargs)

    d1 = domain()
    d2 = domain()
    print("Clearing cache.", end=" ")
    domain.clear_cache_for_all_instances()
    print("Done.")
    d3 = domain()

    print("Changing domain range.", end=" ")
    domain.set_new_range(20, 100)
    print("Done.")

    d4 = domain()
    d5 = domain()

    print(len(d1), len(d4))

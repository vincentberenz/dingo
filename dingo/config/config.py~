from dataclasses import dataclass
from typing import Any, Optional, Generic, TypeVar, Union
from dingo.gw.domains import Domain, FrequencyDomain, TimeDomain
from dingo.core.utils.factory import class_factory


ConfigDict = dict[str, Union[Any, 'ConfigDict']]


class Configured:
    def __init__(self) -> None:
        # raw configuration dictionaries
        self.settings: Optional[Any] = None
        # factories generated from raw dictionaries
        self.factories: Optional[Any] = None


T = TypeVar('T', bound=Configured)


@dataclass
class Factories(Generic[T]):

    def __init__(self) -> None:
        self.settings: "Settings"

    def instantiate(self) -> T:
        raise NotImplementedError()


@dataclass
class Settings(Generic[T]):

    def build_factories(self) -> Factories[T]:
        raise NotImplementedError

    @classmethod
    def instantiate(cls, config_dict: ConfigDict) -> T:
        # config_dict: likely content of a yaml config file
        # reading the higher level keys
        settings = cls(**config_dict)
        # generating the factories ("build_factories")
        factories = settings.build_factories()
        # applying the factories
        t: T = factories.instantiate()
        # keeping "logs" of the construction process
        t.factories = factories
        t.settings = settings
        return t


@dataclass
class DomainFactory:
    type: str
    args: dict[str, Any]

    def instantiate(self) -> Domain:
        if self.type == "FrequencyDomain":
            return class_factory(FrequencyDomain, self.args)
        elif self.type == "TimeDomain":
            return class_factory(TimeDomain, self.args)
        return class_factory(self.type, self.args)


@dataclass
class WaveFormDataset(Configured):
    domain: Domain
    num_samples: int


@dataclass
class WaveformDatasetConfig(Factories[WaveFormDataset]):
    # see dingo.gw.dataset.generate_waveform_dataset
    domain_factory: DomainFactory
    num_samples: int

    # intrinsic_prior: Optional[Any] = None  # todo
    # waveform_generator: Optional[Any] = None  # todo
    # compression: Optional[CompressionConfig] = None  # todo

    def instantiate(self) -> WaveFormDataset:
        return WaveFormDataset(
            domain=self.domain_factory.instantiate(),
            num_samples=self.num_samples
        )


@dataclass
class WaveformDatasetSettings(Settings[WaveFormDataset]):
    # The keys expected from the yaml settings file
    domain: ConfigDict
    waveform_generator: ConfigDict
    intrinsic_prior: ConfigDict
    compression: Optional[ConfigDict]
    num_samples: int

    def preprocess(self) -> WaveformDatasetConfig:
        return WaveformDatasetConfig(
            domain_factory=DomainFactory(
                type=self.domain["type"],  # type: ignore
                args={k: v for k, v in self.domain.items() if k != "type"}
            ),
            num_samples=self.num_samples
        )

# usage:

# waveform_dataset: WaveFormDataset = WaveformDatasetSettings.instanciate(config_dict)

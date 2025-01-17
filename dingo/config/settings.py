from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Generic, TypeVar, Union


ConfigDict = dict[str, Union[Any, 'ConfigDict']]

T = TypeVar('T', bound="Configured")


class Configured(Generic[T]):
    def __init__(self) -> None:
        # raw configuration dictionaries
        self.settings: Optional["Settings[T]"] = None
        # factories generated from raw dictionaries
        self.factories: Optional["Factories[T]"] = None


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

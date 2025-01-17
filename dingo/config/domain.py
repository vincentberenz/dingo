from dataclasses import dataclass
from typing import Any
from dingo.gw.domains import Domain, FrequencyDomain, TimeDomain
from dingo.config.factory import class_factory


@dataclass
class DomainFactory:
    type: str
    args: dict[str, Any]

    def instantiate(self) -> Domain:
        if self.type in ("FrequencyDomain", "FD"):
            return class_factory(FrequencyDomain, self.args)
        elif self.type in ("TimeDomain", "TD"):
            return class_factory(TimeDomain, self.args)
        # type has to be an import path to a class
        return class_factory(self.type, self.args)

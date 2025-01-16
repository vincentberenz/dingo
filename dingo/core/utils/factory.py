"""
Module defining class_factory, a function for instantiating classes
based on configuration dicts.

E.g.

```
domain : FrequencyDomain = class_factory(
    "dingo.gw.domain.FrequencyDomain",
    {
         f_min: 20.0,
         f_max: 1024.0,
         delta_f: 0.25
    }
)
```

"""

from multipledispatch import dispatch
import importlib
import inspect
from typing import Any, Type, TypeVar
import logging
from rich.table import Table

_logger = logging.getLogger("factory")


def _instantiation_table(cls: Type, params: dict[str, Any]) -> Table:
    # used for logging instantiated class as a beautiful table, providing
    # users with the parameters values used for instantiation.
    table = Table(title=f"Class Instantiation: {cls.__name__}")
    table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Value", style="green")

    for param, value in params.items():
        param_type = type(value).__name__
        table.add_row(param, param_type, str(value))

    return Table


_Class = TypeVar("_Class")


def _create_instance(cls: Type[_Class], params: dict[str, Any]) -> _Class:
    # Instantiate cls based on the parameters.
    # Checks parameters are suitable for instantiating cls, an explicit
    # Value or TypeError being raised if not.
    # Log the class instiantiation (info level)

    init_signature = inspect.signature(cls.__init__)
    init_params = init_signature.parameters

    # Validate the dictionary against the __init__ parameters
    for name, param in init_params.items():
        if name == "self":
            continue

        if name not in params:
            if param.default is param.empty:
                raise ValueError(f"Missing required parameter: {name}")
            continue

        # Check type annotations
        if param.annotation is not param.empty:
            expected_type = param.annotation
            if not isinstance(params[name], expected_type):
                raise TypeError(
                    f"Parameter '{name}' is expected to be of type {expected_type}, "
                    f"but got {type(params[name])}"
                )

    if _logger.isEnabledFor(logging.INFO):
        table = _instantiation_table(cls, params)
        _logger.info(table)

    return cls(**params)


@dispatch(Type)
def class_factory(cls: Type[_Class], params: dict[str, Any]) -> _Class:
    """
    Instantiate the class based on the parameters.
    Log the class instantiation (info level)

    Parameters
    ----------
    cls
      the class to instanciate
    params
      dictionary to be "cast" as parameters to the class constructor

    Returns
    -------
    The generated instance

    Raises
    ------
    ValueError
      If params is missing a required argument
    TypeError
      If a value of param is of unexpected type
    """
    return _create_instance(cls, params)


@dispatch(str)
def class_factory(class_path: str, params: dict[str, Any]) -> Any:
    """
    Instantiate the class based on the parameters.
    Log the class instantiation (info level)

    Parameters
    ----------
    class_path
      the import path of the class to instanciate
    params
      dictionary to be "cast" as parameters to the class constructor

    Returns
    -------
    The generated instance

    Raises
    ------
    ValueError
      If params is missing a required argument
    TypeError
      If a value of param is of unexpected type
    ModuleNotFoundError
      If import of the class based on the class path failed
    """
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ModuleNotFoundError(
            f"Failed to import class '{class_name}' from '{module_name}'."
        ) from e
    return _create_instance(cls, params)

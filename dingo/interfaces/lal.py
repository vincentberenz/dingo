from dataclasses import asdict, dataclass
from numbers import Number
from typing import Any, List, Optional, Tuple, TypedDict, Union

import lal
import numpy as np
from bilby.gw.conversion import (
    bilby_to_lalsimulation_spins,
    convert_to_lal_binary_black_hole_parameters,
)


def _convert_to_scalar(x: Union[np.ndarray, float]) -> Union[Number | float]:
    """
    Convert a single element array to a number.

    Parameters
    ----------
    x:
        Array or number

    Returns
    -------
    A number
    """
    if isinstance(x, np.ndarray):
        if x.shape == () or x.shape == (1,):
            return x.item()
        else:
            raise ValueError(
                f"Expected an array of length one, but go shape = {x.shape}"
            )
    else:
        return x


@dataclass
class PolarizationDict:
    h_plus: np.ndarray
    h_cross: np.ndarray


@dataclass
class LaLSimulationSpins:
    iota: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float


@dataclass
class SimInspiralChooseFDModesParameters:
    mass_1: float
    mass_2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    longAscNodes: float
    eccentricity: float
    meanPerAno: float
    delta_f: float
    f_min: float
    f_max: float
    f_ref: float
    lal_params: List[Any]
    approximant: Any


@dataclass
class LalBinaryBlackHoleParameters:
    luminosity_distance: float
    redshift: float
    a_1: float
    a_2: float
    cos_tilt_1: float
    cos_tilt_2: float
    phi_jl: float
    phi_12: float
    phase: float
    tilt_1: float
    tilt_2: float
    theta_jn: float
    f_ref: float

    # Mass parameters
    mass_1: float
    mass_2: float
    total_mass: float
    chirp_mass: float
    mass_ratio: float
    symmetric_mass_ratio: float

    # Source frame mass parameters
    mass_1_source: float
    mass_2_source: float
    total_mass_source: float
    chirp_mass_source: float

    def get_lal_simulation_spins(
        self, spin_conversion_phase: Optional[float]
    ) -> LaLSimulationSpins:
        keys: Tuple[str, ...] = (
            "theta_jn",
            "phi_jl",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "a_1",
            "a_2",
            "mass_1",
            ",ass_2",
            "f_ref",
            "phase",
        )
        args: List[float] = [getattr(self, k) for k in keys]
        if spin_conversion_phase is not None:
            args[-1] = spin_conversion_phase
        iota_and_cart_spins: List[float] = [
            float(_convert_to_scalar(value))
            for value in bilby_to_lalsimulation_spins(args)
        ]
        return LaLSimulationSpins(*iota_and_cart_spins)


@dataclass
class WaveformParams:
    luminosity_distance: Optional[float] = None
    redshift: Optional[float] = None
    comoving_distance: Optional[float] = None
    chi_1: Optional[float] = None
    chi_2: Optional[float] = None
    chi_1_in_plane: Optional[float] = None
    chi_2_in_plane: Optional[float] = None
    a_1: Optional[float] = None
    a_2: Optional[float] = None
    phi_jl: Optional[float] = None
    phi_12: Optional[float] = None
    cos_tilt_1: Optional[float] = None
    cos_tilt_2: Optional[float] = None
    delta_phase: Optional[float] = None
    psi: Optional[float] = None
    theta_jn: Optional[float] = None
    f_ref: Optional[float] = None

    # Mass parameters
    mass_1: Optional[float] = None
    mass_2: Optional[float] = None
    total_mass: Optional[float] = None
    chirp_mass: Optional[float] = None
    mass_ratio: Optional[float] = None
    symmetric_mass_ratio: Optional[float] = None

    # Source frame mass parameters
    mass_1_source: Optional[float] = None
    mass_2_source: Optional[float] = None
    total_mass_source: Optional[float] = None
    chirp_mass_source: Optional[float] = None

    def to_binary_black_hole_parameters(
        self, convert_to_SI: bool
    ) -> LalBinaryBlackHoleParameters:
        params = asdict(self)
        converted_params, _ = convert_to_lal_binary_black_hole_parameters(params)
        if convert_to_SI:
            converted_params.mass_1 *= lal.MSUN_SI
            converted_params.mass_2 *= lal.MSUN_SI
            converted_params.luminosity_distance *= 1e6 * lal.PC_SI
        return LalBinaryBlackHoleParameters(**converted_params)


#######################


class LalSourceFrameParameters(TypedDict, total=False):
    theta_jn: float
    phi_jl: float
    tilt_1: float
    tilt_2: float
    phi_12: float
    a_1: float
    a_2: float
    mass_1: float
    mass_2: float
    f_ref: float
    phase: float


def to_lal_simulation_spins(
    lalBHP: LaLBlackHoleParameters,
    spin_conversion_phase: Optional[float],
) -> LaLSimulationSpins:

    keys = LalSourceFrameParameters.__annotations__.keys()
    params = [lalBHP[k] for k in keys]  # type: ignore
    if spin_conversion_phase is not None:
        params[-1] = spin_conversion_phase
    iota_and_cart_spins = bilby_to_lalsimulation_spins(*params)
    spin_values: list[float] = [
        # todo: solve conversion Number to float.
        float(_convert_to_scalar(x))
        for x in iota_and_cart_spins
    ]
    d: LaLSimulationSpins = {  # type: ignore
        k: v for k, v in zip(LaLSimulationSpins.__annotations__.keys(), spin_values)
    }
    return d


def check_consistency(parameters: BlackHoleParameters) -> Optional[str]:
    # Vincent: To double check ! Generated with GPT-4/Phind
    """
    Check the consistency of an instance of BlackHoleInputParameters,
    including before call to  the
    convert_to_lal_binary_black_hole_parameters function.

    Parameters
    ----------
    parameters : BlackHoleInputParameters
        Dictionary of parameter values to check for consistency.

    Returns
    -------
        None if parameters are consistant, an error message
        otherwise.
    """
    # Check for required mass parameters
    if "mass_1" not in parameters or "mass_2" not in parameters:
        return "Missing mass parameters"

    # Check for spin parameters consistency
    for idx in ["1", "2"]:
        chi_key = f"chi_{idx}"
        a_key = f"a_{idx}"
        if chi_key in parameters:
            if f"chi_{idx}_in_plane" in parameters:
                # Check if the calculated 'a' is consistent
                calculated_a = (
                    parameters[chi_key] ** 2  # type: ignore
                    + parameters[f"chi_{idx}_in_plane"] ** 2  # type: ignore
                ) ** 0.5
                if a_key in parameters and not np.isclose(
                    parameters[a_key], calculated_a  # type: ignore
                ):
                    return f"Error: Inconsistent spin magnitude for {a_key}"
            elif a_key in parameters:
                # Check if 'cos_tilt' can be calculated
                if parameters[a_key] == 0:  # type: ignore
                    return str(f"Division by zero in calculating cos_tilt for {a_key}")

    # Check for extrinsic parameters
    if "luminosity_distance" not in parameters:
        if "redshift" not in parameters and "comoving_distance" not in parameters:
            return "Error: Missing distance parameters"

    # Check for angle parameters
    for angle in ["tilt_1", "tilt_2", "theta_jn"]:
        cos_angle = f"cos_{angle}"
        if cos_angle in parameters:
            if not (-1 <= parameters[cos_angle] <= 1):  # type: ignore
                return f"Error: {cos_angle} must be between -1 and 1."

    # If all checks pass
    return None


class SimInspiralFDArgs(TypedDict, total=False):
    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
    # group___l_a_l_sim_inspiral__c.html#ga4e21f4a33fc0238ebd38793ae2bb8745
    m1: float
    m2: float
    S1x: float
    S1y: float
    S1z: float
    S2x: float
    S2y: float
    S2z: float
    distance: float
    inclination: float
    PhiRef: float
    longAsNodes: float
    eccentricity: float
    meanPerAno: float
    deltaF: float
    f_min: float
    f_max: float
    f_ref: float
    lal_params: Optional[Any]  # todo: not Any. LaLBlackHoleParameters ?
    approximant: int

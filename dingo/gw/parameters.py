import numpy as np
from typing import TypedDict, Optional, Union
from numbers import Number
from bilby.gw.conversion import bilby_to_lalsimulation_spins


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


class PolarizationDict(TypedDict):
    h_plus: np.ndarray
    h_cross: np.ndarray


class BlackHoleParameters(TypedDict, total=False):
    mass_1: float
    mass_2: float
    a_1: float
    a_2: float
    tilt_1: float
    tilt_2: float
    phi_12: float
    phi_jl: float
    luminosity_distance: float
    theta_jn: float
    phase: float
    ra: float
    dec: float
    geocent_time: float
    psi: float
    redshift: float
    comoving_distance: float
    chi_1: float
    chi_2: float
    chi_1_in_plane: float
    chi_2_in_plane: float
    delta_phase: float


class LaLBlackHoleParameters(TypedDict, total=False):
    mass_1: float
    mass_2: float
    a_1: float
    a_2: float
    tilt_1: float
    tilt_2: float
    phi_12: float
    phi_jl: float
    luminosity_distance: float
    theta_jn: float
    phase: float
    ra: float
    dec: float
    geocent_time: float
    psi: float
    redshift: float
    cos_tilt_1: float
    cos_tilt_2: float


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


class LaLSimulationSpins(TypedDict):
    iota: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float


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
        float(_convert_to_scalar(x)) for x in iota_and_cart_spins
    ]
    d: LaLSimulationSpins = {  # type: ignore
        k: v for k, v in zip(
            LaLSimulationSpins.__annotations__.keys(),
            spin_values
        )
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
    if 'mass_1' not in parameters or 'mass_2' not in parameters:
        return "Missing mass parameters"

    # Check for spin parameters consistency
    for idx in ['1', '2']:
        chi_key = f'chi_{idx}'
        a_key = f'a_{idx}'
        if chi_key in parameters:
            if f"chi_{idx}_in_plane" in parameters:
                # Check if the calculated 'a' is consistent
                calculated_a = (
                    parameters[chi_key] ** 2 +  # type: ignore
                    parameters[f"chi_{idx}_in_plane"] ** 2  # type: ignore
                ) ** 0.5
                if a_key in parameters and not np.isclose(
                        parameters[a_key], calculated_a  # type: ignore
                ):
                    return f"Error: Inconsistent spin magnitude for {a_key}"
            elif a_key in parameters:
                # Check if 'cos_tilt' can be calculated
                if parameters[a_key] == 0:  # type: ignore
                    return str(
                        f"Division by zero in calculating cos_tilt for {a_key}"
                    )

    # Check for extrinsic parameters
    if 'luminosity_distance' not in parameters:
        if 'redshift' not in parameters and 'comoving_distance' not in parameters:
            return "Error: Missing distance parameters"

    # Check for angle parameters
    for angle in ['tilt_1', 'tilt_2', 'theta_jn']:
        cos_angle = f'cos_{angle}'
        if cos_angle in parameters:
            if not (-1 <= parameters[cos_angle] <= 1):  # type: ignore
                return f"Error: {cos_angle} must be between -1 and 1."

    # If all checks pass
    return None


class SimInspiralFDArgs:
    phase: Union[float, int]
    mass_1: Union[float, int]
    mass_2: Union[float, int]
    spin_1x: Union[float, int]
    spin_1y: Union[float, int]
    spin_1z: Union[float, int]
    spin_2x: Union[float, int]
    spin_2y: Union[float, int]
    spin_2z: Union[float, int]
    reference_frequency: Union[float, int]
    luminosity_distance: Union[float, int]
    iota: Union[float, int]
    longitude_ascending_nodes: Union[float, int]
    eccentricity: Union[float, int]
    mean_per_ano: Union[float, int]
    delta_frequency: Union[float, int]
    minimum_frequency: Union[float, int]
    maximum_frequency: Union[float, int]
    waveform_dictionary: Optional[lal.Dict]
    approximant: Optional[int, str]

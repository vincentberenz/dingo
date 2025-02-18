import copy
from dataclasses import asdict, astuple, dataclass
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, TypedDict, Union

import lal
import lalsimulation as LS
import numpy as np
from bilby.gw.conversion import (
    bilby_to_lalsimulation_spins,
    convert_to_lal_binary_black_hole_parameters,
)
from nptyping import NDArray, Shape
from typing_extensions import override

import dingo.gw.waveform_generator.wfg_utils as wfg_utils
from dingo.gw.domains import Domain, DomainParameters

Approximant: TypeAlias = int

# numpy array of shape (n,) and dtype complex 128
FrequencySeries: TypeAlias = NDArray[Shape["n"], np.complex128]
Mode: TypeAlias = Tuple[int, int]


def get_approximant(approximant: str) -> Approximant:
    return Approximant(LS.GetApproximantFromString(approximant))


LalParams: TypeAlias = lal.Dict


def get_lal_params(mode_list: List[Mode]) -> LalParams:
    lal_params = lal.CreateDict()
    ma = LS.SimInspiralCreateModeArray()
    for ell, m in mode_list:
        LS.SimInspiralModeArrayActivateMode(ma, ell, m)
    LS.SimInspiralWaveformParamsInsertModeArray(lal_params, ma)
    return lal_params


def _convert_to_float(x: Union[np.ndarray, Number, float]) -> float:
    """
    Convert a single element array to a number.

    Parameters
    ----------
    x:
        Array or float

    Returns
    -------
    A number
    """
    if isinstance(x, np.ndarray):
        if x.shape == () or x.shape == (1,):
            return float(x.item())
        else:
            raise ValueError(
                f"Expected an array of length one, but go shape = {x.shape}"
            )
    else:
        return float(x)


@dataclass
class PolarizationDict:
    h_plus: np.ndarray
    h_cross: np.ndarray


def rotate_z(
    angle: float, vx: float, vy: float, vz: float
) -> Tuple[float, float, float]:
    vx_new = vx * np.cos(angle) - vy * np.sin(angle)
    vy_new = vx * np.sin(angle) + vy * np.cos(angle)
    return vx_new, vy_new, vz


def rotate_y(
    angle: float, vx: float, vy: float, vz: float
) -> Tuple[float, float, float]:
    vx_new = vx * np.cos(angle) + vz * np.sin(angle)
    vz_new = -vx * np.sin(angle) + vz * np.cos(angle)
    return vx_new, vy, vz_new


@dataclass
class LaLSimulationSpins:
    iota: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float

    def get_JL0_euler_angles(
        self, m1: float, m2: float, converted_to_SI: bool, f_ref: float, phase: float
    ) -> Tuple[float, float, float]:

        if converted_to_SI:
            m1 /= lal.MSUN_SI
            m2 /= lal.MSUN_SI

        m = m1 + m2
        eta = m1 * m2 / m**2
        v0 = (m * lal.MTSUN_SI * np.pi * f_ref) ** (1 / 3)

        m1sq = m1 * m1
        m2sq = m2 * m2

        s1x = m1sq * self.s1x
        s1y = m1sq * self.s1y
        s1z = m1sq * self.s1z
        s2x = m2sq * self.s2x
        s2y = m2sq * self.s2y
        s2z = m2sq * self.s2z

        delta = np.sqrt(1 - 4 * eta)
        m1_prime = (1 + delta) / 2
        m2_prime = (1 - delta) / 2
        Sl = m1_prime**2 * self.s1z + m2_prime**2 * self.s2z
        Sigmal = self.s2z * m2_prime - self.s1z * m1_prime

        # This calculation of the orbital angular momentum is taken from Appendix G.2 of PRD 103, 104056 (2021).
        # It may not align exactly with the various XPHM PrecVersions, but the error should not be too big.
        Lmag = (m * m * eta / v0) * (
            1
            + v0 * v0 * (1.5 + eta / 6)
            + (27 / 8 - 19 * eta / 8 + eta**2 / 24) * v0**4
            + (
                7 * eta**3 / 1296
                + 31 * eta**2 / 24
                + (41 * np.pi**2 / 24 - 6889 / 144) * eta
                + 135 / 16
            )
            * v0**6
            + (
                -55 * eta**4 / 31104
                - 215 * eta**3 / 1728
                + (356035 / 3456 - 2255 * np.pi**2 / 576) * eta**2
                + eta
                * (
                    -64 * np.log(16 * v0**2) / 3
                    - 6455 * np.pi**2 / 1536
                    - 128 * lal.GAMMA / 3
                    + 98869 / 5760
                )
                + 2835 / 128
            )
            * v0**8
            + (-35 * Sl / 6 - 5 * delta * Sigmal / 2) * v0**3
            + (
                (-77 / 8 + 427 * eta / 72) * Sl
                + delta * (-21 / 8 + 35 * eta / 12) * Sigmal
            )
            * v0**5
        )

        Jx = s1x + s2x
        Jy = s1y + s2y
        Jz = Lmag + s1z + s2z

        Jnorm = np.sqrt(Jx * Jx + Jy * Jy + Jz * Jz)
        Jhatx = Jx / Jnorm
        Jhaty = Jy / Jnorm
        Jhatz = Jz / Jnorm

        # The calculation of the Euler angles is described in Appendix C of PRD 103, 104056 (2021).
        theta_JL0 = np.arccos(Jhatz)
        phi_JL0 = np.arctan2(Jhaty, Jhatx)

        Nx = np.sin(self.iota) * np.cos(np.pi / 2 - phase)
        Ny = np.sin(self.iota) * np.sin(np.pi / 2 - phase)
        Nz = np.cos(self.iota)

        # Rotate N into J' frame.
        Nx_Jp, Ny_Jp, Nz_Jp = rotate_y(-theta_JL0, *rotate_z(-phi_JL0, Nx, Ny, Nz))

        kappa = np.arctan2(Ny_Jp, Nx_Jp)

        alpha_0 = np.pi - kappa
        beta_0 = theta_JL0
        gamma_0 = np.pi - phi_JL0

        return alpha_0, beta_0, gamma_0

    def convert_J_to_L0_frame(
        self,
        hlm_J: Dict[Mode, FrequencySeries],
        m1: float,
        m2: float,
        converted_to_SI: bool,
        f_ref: float,
        phase: float,
    ) -> Dict[Mode, FrequencySeries]:

        alpha_0, beta_0, gamma_0 = self.get_JL0_euler_angles(
            m1, m2, converted_to_SI, f_ref, phase
        )

        hlm_L0 = {}
        for (l, m), hlm in hlm_J.items():
            for mp in range(-l, l + 1):
                wigner_D = (
                    np.exp(1j * m * alpha_0)
                    * np.exp(1j * mp * gamma_0)
                    * lal.WignerdMatrix(l, m, mp, beta_0)
                )
                if (l, mp) not in hlm_L0:
                    hlm_L0[(l, mp)] = wigner_D * hlm
                else:
                    hlm_L0[(l, mp)] += wigner_D * hlm

        return hlm_L0


@dataclass
class SimInspiralChooseFDModesParameters(LaLSimulationSpins):
    mass_1: float
    mass_2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    delta_f: float
    f_min: float
    f_max: float
    f_ref: float
    phase: float
    r: float
    iota: float
    lal_params: Optional[LalParams]
    approximant: Approximant

    @override
    def get_JL0_euler_angles(
        self, hlm_J: Dict[Mode, FrequencySeries], spin_conversion_phase: Optional[float]
    ) -> Dict[Mode, FrequencySeries]:
        phase = self.phase
        if spin_conversion_phase is not None:
            phase = 0.0
        converted_to_SI = True
        return LaLSimulationSpins.get_JL0_euler_to_angles(
            hlm_J, self.m1, self.m2, converted_to_SI, self.f_ref, phase
        )


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
            float(_convert_to_float(value))
            for value in bilby_to_lalsimulation_spins(args)
        ]
        return LaLSimulationSpins(*iota_and_cart_spins)

    def to_SimInspiralChooseFDModes_parameters(
        self,
        domain: Domain,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[LalParams],
        approximant: Optional[Approximant],
    ) -> SimInspiralChooseFDModesParameters:
        spins: LaLSimulationSpins = self.get_lal_simulation_spins(spin_conversion_phase)
        domain_params = asdict(domain.get_parameters())
        # adding iota, s1x, ..., s2x, ...
        parameters = asdict(spins)
        # direct mapping from this instance
        for k in ("mass_1", "mass_2", "phase"):
            parameters[k] = getattr(self, k)
        # adding domain related params
        for k in ("delta_t", "f_min", "f_max", "f_ref"):
            parameters[k] = domain_params[k]
        # other params
        parameters["r"] = self.luminosity_distance
        parameters["lal_params"] = lal_params
        parameters["approximant"] = approximant
        return SimInspiralChooseFDModesParameters(**parameters)


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

    # are those the same ?
    delta_phase: Optional[float] = None
    phase: Optional[float] = None

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


def sim_inspiral_choose_FD_modes(
    params: WaveformParams,
    f_ref: float,
    convert_to_SI: bool,
    domain: Domain,
    approximant: Approximant,
    mode_list: List[Mode],
    spin_conversion_phase: Optional[float],
    lal_params: Optional[LalParams],
):
    supported_approximants: Tuple[Optional[Approximant], ...] = (Approximant(101),)
    if approximant not in supported_approximants:
        raise ValueError(
            "the 'LS.SimInspiralChooseFDModes' supports only the approximents: "
            f"{','.join([str(ap) for ap in supported_approximants])} ({approximant} not supported)"
        )

    # generating the frequencies series
    params_ = copy.deepcopy(params)
    params_.f_ref = f_ref
    bbh_parameters = params_.to_binary_black_hole_parameters(convert_to_SI)
    lal_params = get_lal_params(mode_list)
    si_choose_fd_modes_params = bbh_parameters.to_SimInspiralChooseFDModes_parameters(
        domain, spin_conversion_phase, lal_params, approximant
    )
    hlm_fd__: LS.SphHarmFrequencySeries = LS.SimInspiralChooseFDModes(
        list(astuple(si_choose_fd_modes_params))
    )
    hlm_fd_: Dict[Mode, lal.COMPLEX16FrequencySeries] = (
        wfg_utils.linked_list_modes_to_dict_modes(hlm_fd__)
    )
    hlm_fd: Dict[Mode, FrequencySeries] = {k: v.data.data for k, v in hlm_fd_.items()}

    return si_choose_fd_modes_params.get_JL0_euler_angles(hlm_fd, spin_conversion_phase)


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

import json
from typing import Optional
from copy import copy
from copy import deepcopy
from typing import Protocol
from typing import NewType
from dataclasses import dataclass, asdict
from typing import TypedDict, Literal
import numpy as np

import lal
import lalsimulation as LS


Approximant = Literal[
    "TaylorT1", "TaylorT2", "TaylorT3", "TaylorF1",
    "EccentricFD", "TaylorF2", "TaylorF2Ecc",
    "TaylorF2NLTides", "TaylorR2F4", "TaylorF2RedSpin",
    "TaylorF2RedSpinTidal", "PadeT1", "PadeF1", "EOB",
    "BCV", "BCVSpin", "SpinTaylorT1",
    "SpinTaylorT2", "SpinTaylorT3",
    "SpinTaylorT4", "SpinTaylorT5", "SpinTaylorF2",
    "SpinTaylorFrameless", "SpinTaylor",
    "PhenSpinTaylor", "PhenSpinTaylorRD", "SpinQuadTaylor",
    "FindChirpSP", "FindChirpPTF", "GeneratePPN",
    "BCVC", "FrameFile", "AmpCorPPN", "NumRel",
    "NumRelNinja2", "Eccentricity",
    "EOBNR", "EOBNRv2", "EOBNRv2HM",
    "EOBNRv2_ROM", "EOBNRv2HM_ROM",
    "TEOBResum_ROM", "SEOBNRv1", "SEOBNRv2", "SEOBNRv2_opt",
    "SEOBNRv3", "SEOBNRv3_pert", "SEOBNRv3_opt",
    "SEOBNRv3_opt_rk4", "SEOBNRv4",
    "SEOBNRv4_opt", "SEOBNRv4P", "SEOBNRv4PHM",
    "SEOBNRv2T", "SEOBNRv4T",
    "SEOBNRv1_ROM_EffectiveSpin", "SEOBNRv1_ROM_DoubleSpin",
    "SEOBNRv2_ROM_EffectiveSpin",
    "SEOBNRv2_ROM_DoubleSpin", "SEOBNRv2_ROM_DoubleSpin_HI",
    "Lackey_Tidal_2013_SEOBNRv2_ROM", "SEOBNRv4_ROM", "SEOBNRv4HM_ROM",
    "SEOBNRv4_ROM_NRTidal", "SEOBNRv4_ROM_NRTidalv2",
    "SEOBNRv4_ROM_NRTidalv2_NSBH",
    "SEOBNRv4T_surrogate", "HGimri",
    "IMRPhenomA", "IMRPhenomB", "IMRPhenomC", "IMRPhenomD",
    "IMRPhenomD_NRTidal", "IMRPhenomD_NRTidalv2",
    "IMRPhenomNSBH", "IMRPhenomHM", "IMRPhenomP",
    "IMRPhenomPv2", "IMRPhenomPv2_NRTidal", "IMRPhenomPv2_NRTidalv2",
    "IMRPhenomFC", "TaylorEt", "TaylorT4",
    "EccentricTD", "TaylorN", "SpinTaylorT4Fourier",
    "SpinTaylorT5Fourier", "SpinDominatedWf", "NR_hdf5", "NRSur4d2s",
    "NRSur7dq2", "NRSur7dq4", "SEOBNRv4HM", "NRHybSur3dq8", "IMRPhenomXAS",
    "IMRPhenomXHM", "IMRPhenomPv3", "IMRPhenomPv3HM",
    "IMRPhenomXP", "IMRPhenomXPHM",
    "TEOBResumS", "IMRPhenomT", "IMRPhenomTHM", "IMRPhenomTP", "IMRPhenomTPHM",
    "SEOBNRv5_ROM", "SEOBNRv4HM_PA", "pSEOBNRv4HM_PA", "IMRPhenomXAS_NRTidalv2",
    "IMRPhenomXP_NRTidalv2", "IMRPhenomXO4a", "ExternalPython", "SEOBNRv5HM_ROM",
    "IMRPhenomXAS_NRTidalv3", "IMRPhenomXP_NRTidalv3", "SEOBNRv5_ROM_NRTidalv3"
]


def _dict_to_list(d: dict) -> list:
    r: list = []
    for value in d.values():
        if type(value) is dict:
            r.extend(_dict_to_list(value))
        else:
            r.append(value)
    return r


def _dataclass_to_list(dc) -> list:
    return _dict_to_list(asdict(dc))


Seconds = NewType["Seconds", int]
Nanoseconds = NewType["Nanoseconds", int]


@dataclass
class LALUnit:
    # ligo time gps://lscsoft.docs.ligo.org/lalsuite/lal/struct_l_a_l_unit.html
    power_of_ten: int
    unit_numerator: int
    unit_denominator_minus_one: int

    @classmethod
    def from_lal_LALUnit(cls, lalunit) -> "LALUnit":
        return cls(
            power_of_ten=lalunit.powerOfTen,
            unit_numerator=lalunit.unit_numerator,
            unit_denominator_minus_one=lalunit.unitDenominatorMinusOne
        )


@dataclass
class FrequencySeries:
    # ligo_time_gps:
    # https://lscsoft.docs.ligo.org/lalsuite
    # /lal/struct_l_i_g_o_time_g_p_s.html
    name: str
    f0: float
    delta_f: float
    epoch: tuple[Seconds, Nanoseconds]  #
    data: np.ndarray
    sample_units: LALUnit

    @classmethod
    def from_lal_FrequencySeries(cls, frequencySeries) -> "FrequencySeries":
        return cls(
            name=frequencySeries.name,
            f0=frequencySeries.f0,
            delta_f=frequencySeries.deltaF,
            epoch=(frequencySeries.epoch.gpsSeconds,
                   frequencySeries.epoch.gpsNanoSeconds),
            data=frequencySeries.data.data,
            sample_units=LALUnit.from_lal_LALUnit(frequencySeries.sampleUnits)
        )

    def max(self) -> int:
        return np.max(np.abs(self.data))


@dataclass
class Polarization:
    hplus: FrequencySeries
    hcross: FrequencySeries

    def max(self) -> int:
        return max(self.hcross.max(), self.hplus.max())


@dataclass
class CartesianSpins:
    x: float
    y: float
    z: float


@dataclass
class ECC:
    long_as_nodes: float
    eccentricity: float
    mean_per_ano: float


@dataclass
class DomainParams:
    delta_f: float
    f_min: float
    f_max: float
    delta_t: float


@dataclass
class LalParams:
    ...

    @classmethod
    def create(cls) -> "LalParams":
        ...

    def to_lal_dict(self) -> Any:
        ...

    @classmethod
    def from_lal_dict(cls, LalDict) -> "LalParams":
        ...


@dataclass
class SimInspiral:
    m1: float
    m2: float
    s1: CartesianSpins
    s2: CartesianSpins
    distance: float
    inclination: float
    PhiRef: float
    ecc: ECC
    domain_params: DomainParams
    lal_params: Optional[LalParams]
    approximant: Approximant


class SimInspiralFunction(Protocol):
    def __call__(self, sim_inspiral_params: SimInspiral) -> Polarization:
        ...


class NumericalInstability(Exception):
    ...


def _sim_inspiral(sim_inspiral_params: SimInspiral, fn: SimInspiralFunction) -> Polarization:
    params = _dataclass_to_list(sim_inspiral_params)
    params[-1] = LS.GetApproximantFromString(params[-1])
    hp, hc = fn(params)
    return Polarization(
        hplus=FrequencySeries.from_lal_FrequencySeries(hp),
        hcross=FrequencySeries.from_lal_FrequencySeries(hc)
    )


def sim_inspiral_TD(sim_inspiral_params: SimInspiral) -> Polarization:
    return _sim_inspiral(sim_inspiral_params, LS.SimInspiralTD)


def sim_inspiral_FD(
        sim_inspiral_params: SimInspiral,
        turn_off_multibanding: bool,
        threshold: float = 1e-20,
        repetition: int = 2
) -> Polarization:
    polarization: Polarization = _sim_inspiral(
        sim_inspiral_params, LS.SimInspiralFD
    )

    if not turn_off_multibanding:
        return polarization

    if polarization.max() > threshold:
        if sim_inspiral_params.lal_params is None:
            sim_inspiral_params.lal_params = LalParams.create()
        d = sim_inspiral_params.lal_params.to_lal_dict()
        for _ in range(repetition):
            LS.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(
                d, 0
            )
        sim_inspiral_params.lal_params = LalParams.from_lal_dict(d)
        polarization = _sim_inspiral(
            sim_inspiral_params, LS.SimInspiralFD
        )

        if polarization.max() > threshold:
            raise NumericalInstability(
                "sim_inspiral_FD: detected numerical instability "
                f"for parameters {json.dumps(asdict(sim_inspiral_params))}. "
                "Attempted to turn off multibanding, but this might not have "
                "fixed it"
            )

        return polarization

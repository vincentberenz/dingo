from dataclasses import dataclass, asdict
from typing import TypedDict, Literal
import numpy as np

import lal
import lalsimulation as LS


Approximant = Literal[
    "TaylorT1", "TaylorT2", "TaylorT3", "TaylorF1",
    "EccentricFD", "TaylorF2", "TaylorF2Ecc", "TaylorF2NLTides", "TaylorR2F4", "TaylorF2RedSpin",
    "TaylorF2RedSpinTidal", "PadeT1", "PadeF1", "EOB", "BCV", "BCVSpin", "SpinTaylorT1",
    "SpinTaylorT2", "SpinTaylorT3", "SpinTaylorT4", "SpinTaylorT5", "SpinTaylorF2",
    "SpinTaylorFrameless", "SpinTaylor", "PhenSpinTaylor", "PhenSpinTaylorRD", "SpinQuadTaylor",
    "FindChirpSP", "FindChirpPTF", "GeneratePPN", "BCVC", "FrameFile", "AmpCorPPN", "NumRel",
    "NumRelNinja2", "Eccentricity", "EOBNR", "EOBNRv2", "EOBNRv2HM",
    "EOBNRv2_ROM", "EOBNRv2HM_ROM", "TEOBResum_ROM", "SEOBNRv1", "SEOBNRv2", "SEOBNRv2_opt",
    "SEOBNRv3", "SEOBNRv3_pert", "SEOBNRv3_opt", "SEOBNRv3_opt_rk4", "SEOBNRv4",
    "SEOBNRv4_opt", "SEOBNRv4P", "SEOBNRv4PHM", "SEOBNRv2T", "SEOBNRv4T",
    "SEOBNRv1_ROM_EffectiveSpin", "SEOBNRv1_ROM_DoubleSpin",
    "SEOBNRv2_ROM_EffectiveSpin", "SEOBNRv2_ROM_DoubleSpin", "SEOBNRv2_ROM_DoubleSpin_HI",
    "Lackey_Tidal_2013_SEOBNRv2_ROM", "SEOBNRv4_ROM", "SEOBNRv4HM_ROM",
    "SEOBNRv4_ROM_NRTidal", "SEOBNRv4_ROM_NRTidalv2", "SEOBNRv4_ROM_NRTidalv2_NSBH",
    "SEOBNRv4T_surrogate", "HGimri", "IMRPhenomA", "IMRPhenomB", "IMRPhenomC", "IMRPhenomD",
    "IMRPhenomD_NRTidal", "IMRPhenomD_NRTidalv2", "IMRPhenomNSBH", "IMRPhenomHM", "IMRPhenomP",
    "IMRPhenomPv2", "IMRPhenomPv2_NRTidal", "IMRPhenomPv2_NRTidalv2",
    "IMRPhenomFC", "TaylorEt", "TaylorT4", "EccentricTD", "TaylorN", "SpinTaylorT4Fourier",
    "SpinTaylorT5Fourier", "SpinDominatedWf", "NR_hdf5", "NRSur4d2s",
    "NRSur7dq2", "NRSur7dq4", "SEOBNRv4HM", "NRHybSur3dq8", "IMRPhenomXAS",
    "IMRPhenomXHM", "IMRPhenomPv3", "IMRPhenomPv3HM", "IMRPhenomXP", "IMRPhenomXPHM",
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


@dataclass
class Polarization:
    hplus: np.ndarray
    hcross: np.ndarray


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
    lal_params: LalParams
    approximant: Approximant


def sim_inspiral_TD(sim_inspiral_params: SimInspiral) -> Polarization:
    params = _dataclass_to_list(sim_inspiral_params)
    params[-1] = LS.GetApproximantFromString(params[-1])
    hp, hc = LS.SimInspiralTD(*params)
    return Polarization(hplus=hp.data.data, hcross=hc.data.data)

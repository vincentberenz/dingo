# Accepted lal functions:

#            "SimInspiralFD",
#            "SimInspiralTD",
#            "SimInspiralChooseTDModes",
#            "SimInspiralChooseFDModes",
#            "SimIMRPhenomXPCalculateModelParametersFromSourceFrame",


# frequency domain and lal_target_function is SimInspiralFD (default)
def _generate_hplus_hcross(
    domain: FrequencyDomain, parameters: Dict[str, float], catch_waveform_errors=True
) -> Dict[str, np.ndarray]: ...


# frequency domain and lal_target_function is SimInspiralTD
# TODO: does this exist ?
def _generate_hplus_hcross(
    domain: FrequencyDomain, parameters: Dict[str, float], catch_waveform_errors=True
) -> Dict[str, np.ndarray]: ...


# frequency domain and lal_target_function is SimInspiralChooseTDModes
# TODO: does this exist ?
def _generate_hplus_hcross(
    domain: FrequencyDomain, parameters: Dict[str, float], catch_waveform_errors=True
) -> Dict[str, np.ndarray]: ...


# frequency domain and lal_target_function is SimInspiralChooseFDModes
def _generate_hplus_hcross(
    domain: FrequencyDomain, parameters: Dict[str, float], catch_waveform_errors=True
) -> Dict[str, np.ndarray]: ...


# frequency domain and lal_target_function is SimIMRPhenomXPCalculateModelParametersFromSourceFrame
def _generate_hplus_hcross(
    domain: FrequencyDomain, parameters: Dict[str, float], catch_waveform_errors=True
) -> Dict[str, np.ndarray]: ...


# frequency domain and lal_target_function is *NOT* SimInspiralFD
# TODO: then which functions are allowed ?
def _generate_hplus_hcross(
    domain: FrequencyDomain, parameters: Dict[str, float], catch_waveform_errors=True
) -> Dict[str, np.ndarray]: ...


# time domain and lal_target_function is SimInspiralTD (default)
#
# !!!! time domain not implemented yet !!!!
# https://github.com/dingo-gw/dingo/blob/main/dingo/gw/waveform_generator/waveform_generator.py#L329
#
def _generate_hplus_hcross(
    domain: TimeDomain, parameters: Dict[str, float], catch_waveform_errors=True
) -> Dict[str, np.ndarray]: ...


class WaveformGenerator:
    """Generate polarizations using LALSimulation routines in the specified domain for a
    single GW coalescence given a set of waveform parameters.
    """

    def __init__(
        self,
        approximant: str,
        domain: Domain,
        f_ref: float,
        f_start: float = None,
        mode_list: List[Tuple] = None,
        transform=None,
        spin_conversion_phase=None,
        **kwargs,
    ): ...

    def generate_hplus_hcross(
        self, parameters: Dict[str, float], catch_waveform_errors=True
    ) -> Dict[str, np.ndarray]:

        wf_dict = self._selected_generate_hplus_hcross(
            self.domain, parameters, catch_waveform_erros
        )

        if self.transform is not None:
            return self.transform(wf_dict)
        return wf_dict

import numpy as np
from scipy.signal import tukey
from scipy.interpolate import interp1d
from bilby.gw.detector import PowerSpectralDensity

from dingo.gw.prior import default_extrinsic_dict
from dingo.gw.prior import BBHExtrinsicPriorDict



def find_axis(array, dim):
    """Looks for axis with dimension dim in array, and returns its index."""
    indices = np.where(np.array(array.shape) == dim)[0]
    if len(indices) > 1:
        raise ValueError(
            f"Automatic axis detection ambiguous. Array of shape "
            f"{array.shape} has {len(indices)} axes of dimension "
            f"{dim}."
        )
    if len(indices) < 1:
        raise ValueError(
            f"Automatic axis detection failed. Array of shape "
            f"{array.shape} has no axis of dimension {dim}."
        )
    else:
        return indices[0]


def truncate_array(array, axis, lower, upper):
    """Truncate array to [lower:upper] along selected axis."""
    sl = [slice(None)] * array.ndim
    sl[axis] = slice(lower, upper)
    return array[tuple(sl)]


def get_window(window_kwargs):
    """Compute window from window_kwargs."""
    type = window_kwargs["type"]
    if type == "tukey":
        roll_off, T, f_s = (
            window_kwargs["roll_off"],
            window_kwargs["T"],
            window_kwargs["f_s"],
        )
        alpha = 2 * roll_off / T
        w = tukey(int(T * f_s), alpha)
        return w
    else:
        raise NotImplementedError(f"Unknown window type {type}.")


def get_window_factor(window):
    """
    TODO
    Parameters
    ----------
    window

    Returns
    -------

    """
    """Compute window factor."""
    if type(window) == dict:
        window = get_window(window)
    return np.sum(window ** 2) / len(window)


def get_extrinsic_prior_dict(extrinsic_prior):
    """Build dict for extrinsic prior by starting with
    default_extrinsic_dict, and overwriting every element for which
    extrinsic_prior is not default."""
    extrinsic_prior_dict = default_extrinsic_dict.copy()
    for k, v in extrinsic_prior.items():
        if v.lower() != "default":
            extrinsic_prior_dict[k] = v
    return extrinsic_prior_dict


def get_mismatch(a, b, domain, asd_file="aLIGO_ZERO_DET_high_P_asd.txt"):
    """
    Mistmatch is 1 - overlap, where overlap is defined by
    inner(a, b) / sqrt(inner(a, a) * inner(b, b)).
    See e.g. Eq. (44) in https://arxiv.org/pdf/1106.1021.pdf.

    Parameters
    ----------
    a
    b
    domain
    asd_file

    Returns
    -------

    """
    psd = PowerSpectralDensity(asd_file=asd_file)
    asd_interp = interp1d(
        psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
    )
    asd_array = asd_interp(domain.sample_frequencies)
    # whiten a and b.
    min_idx = domain.min_idx
    delta_f = domain.delta_f
    inner_ab = 4 * delta_f * np.sum((a.conj() * b / asd_array**2)[min_idx:], axis=0).real
    inner_aa = 4 * delta_f * np.sum((a.conj() * a / asd_array**2)[min_idx:], axis=0).real
    inner_bb = 4 * delta_f * np.sum((b.conj() * b / asd_array**2)[min_idx:], axis=0).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap

def get_standardization_dict(
    extrinsic_prior_dict, wfd, selected_parameters, transform=None
):
    """
    Calculates the mean and standard deviation of parameters. This is needed for
    standardizing neural-network input and output.

    Parameters
    ----------
    extrinsic_prior_dict : dict
    wfd : WaveformDataset
    selected_parameters : list[str]
        List of parameters for which to estimate standardization factors.
    transform : Transform
        Operator that will generate samples for parameters contained in
        selected_parameters that are not contained in the intrinsic or extrinsic prior.
        (E.g., H1_time, L1_time_proxy)

    Returns
    -------

    """
    # The intrinsic standardization is estimated based on the entire dataset.
    mean_intrinsic, std_intrinsic = wfd.parameter_mean_std()

    # Some of the extrinsic prior parameters have analytic means and standard
    # deviations. If possible, this will either get these, or else it will estimate
    # them numerically.
    ext_prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)
    mean_extrinsic, std_extrinsic = ext_prior.mean_std(ext_prior.keys())

    # Check that overlap between intrinsic and extrinsic parameters is only
    # due to fiducial values (-> std 0)
    for k in std_intrinsic.keys() & std_extrinsic.keys():
        assert std_intrinsic[k] == 0

    # Merge dicts, overwriting fiducial values for parameters (e.g.,
    # luminosity_distance) in intrinsic parameters by the extrinsic ones
    mean = {**mean_intrinsic, **mean_extrinsic}
    std = {**std_intrinsic, **std_extrinsic}

    # For all remaining parameters that require standardization, we use the transform
    # to sample these and estimate the mean and standard deviation numerically.
    additional_parameters = [p for p in selected_parameters if p not in mean]
    if additional_parameters:
        num_samples = min(100_000, len(wfd.parameters))
        samples = {p: np.empty(num_samples) for p in additional_parameters}
        for n in range(num_samples):
            sample = {'parameters': wfd.parameters.iloc[n].to_dict()}
            sample = transform(sample)
            for p in additional_parameters:
                # This assumes all of the additional parameters are contained within
                # extrinsic_parameters. We have set it up so this is the case for the
                # GNPE proxies and the detector coalescence times.
                samples[p][n] = sample["extrinsic_parameters"][p]
        mean_additional = {p: np.mean(samples[p]).item() for p in additional_parameters}
        std_additional = {p: np.std(samples[p]).item() for p in additional_parameters}

        mean.update(mean_additional)
        std.update(std_additional)

    standardization_dict = {
        "mean": {k: mean[k] for k in selected_parameters},
        "std": {k: std[k] for k in selected_parameters},
    }
    return standardization_dict

from fitburst.backend import general
import fitburst.routines as rt
from fitburst.analysis.model import SpectrumModeler

import sys
import numpy as np
import scipy.special as ss


class FitBurstModel(SpectrumModeler):
    def __init__(
        self,
        freqs,
        times,
        dm_incoherent=0,
        factor_freq_upsample=1,
        factor_time_upsample=1,
        num_components=1,
        is_dedispersed=False,
        is_folded=False,
        scintillation=False,
        verbose=False,
    ):
        super().__init__(
            freqs,
            times,
            dm_incoherent,
            factor_freq_upsample,
            factor_time_upsample,
            num_components,
            is_dedispersed,
            is_folded,
            scintillation,
            verbose,
        )

    @staticmethod
    def identify_pbf_regime(arg, threshold=-20.0):
        flag = arg > threshold  # if arg.ndim == 1 else np.all(arg > threshold)
        valid = np.flatnonzero(flag)
        invalid = np.flatnonzero(~flag) if valid.size < flag.size else None

        if valid.size == 0:
            valid = None
        elif (valid[-1] + 1 - valid[0]) == valid.size:
            valid = slice(valid[0], valid[-1] + 1)

        if invalid is not None and (invalid[-1] + 1 - invalid[0]) == invalid.size:
            invalid = slice(invalid[0], invalid[-1] + 1)

        return valid, invalid

    @staticmethod
    def downsample_tile(orig, factors):
        if len(orig.shape) != len(factors):
            raise ValueError(
                f"Mismatch between number of factors and array shape: {len(factors)} vs {len(orig.shape)}"
            )
        shape_new = []
        pos = []
        for i, (f, s) in enumerate(zip(factors, orig.shape)):
            shape_new.append(s // f)
            shape_new.append(f)
            pos.append(i * 2 + 1)

        return np.mean(orig.reshape(shape_new), axis=tuple(pos))

    @staticmethod
    def compute_profile_pbf(
        time: float,
        toa: float,
        width: float,
        freq: float,
        ref_freq: float,
        sc_time_ref: float,
        sc_index: float = -4.0,
        normalize: bool = False,
    ):
        sc_time = sc_time_ref * (freq / ref_freq) ** sc_index

        z = (
            (time - toa) / width
        )  # Normally a 1D array with all the times for one frequency and one component. Here a (num_freq, num_time, num_comp) array

        ratio = (
            np.tile(np.array(width).reshape(1, len(width)), (len(sc_time), 1))
            / sc_time[:, None]
        )  # Normally a constant for one frequency per component. Here a (num_freq, num_comp) array

        ratio = np.tile(ratio[:, np.newaxis, :], (1, time.shape[1], 1))

        arg1 = (
            (ratio - z) / np.sqrt(2)
        )  # Normally a 1D array with all the times for one frequency and one component. Here a (num_freq, num_time, num_comp) array

        amp_term = (
            np.sqrt(np.pi / 2) * ratio
            if normalize
            else np.tile(
                (sc_time_ref / sc_time).reshape(len(sc_time), 1, 1),
                (1, time.shape[1], time.shape[2]),
            )
        )

        profile = np.empty(time.shape, dtype=float)

        r1, r2 = FitBurstModel.identify_pbf_regime(arg1)

        if r1 is not None:
            if type(r1) != slice:
                r1 = np.unravel_index(r1, arg1.shape)
            profile[r1] = amp_term[r1] * np.exp(-0.5 * z[r1] ** 2) * ss.erfcx(arg1[r1])

        if r2 is not None:
            if type(r2) != slice:
                r2 = np.unravel_index(r2, arg1.shape)
            arg2 = (ratio / 2 - z) * ratio
            profile[r2] = amp_term[r2] * np.exp(arg2[r2]) * ss.erfc(arg1[r2])

        return profile

    @staticmethod
    def compute_profile_gaussian(
        values: float, mean: float, width: float, normalize: bool = False
    ):
        norm_factor = 1.0

        if normalize:
            norm_factor /= np.sqrt(2 * np.pi) / width

        profile = norm_factor * np.exp(-0.5 * ((values - mean) / width) ** 2)
        return profile

    @staticmethod
    def compute_spectrum_rpl(
        freqs: np.ndarray, freq_ref: float, sp_idx: float, sp_run: float
    ):
        sp_idx = np.array(sp_idx)
        sp_run = np.array(sp_run)
        log_freq = np.tile(
            np.log(freqs / freq_ref).reshape(len(freqs), 1), (1, len(sp_idx))
        )
        exponent = log_freq * sp_idx[None, :] + log_freq**2 * sp_run[None, :]
        spectrum = np.exp(exponent)
        return spectrum

    @staticmethod
    def compute_amplitude_per_channel(data: float, model: float) -> float:
        data_prof_vector = np.einsum("ft,ftc->fc", data, model)
        amplitudes_matrix = np.einsum("fti,ftj->fij", model, model)
        if np.__version__ > "1":
            res = np.zeros(data_prof_vector.shape)
            for i in range(data_prof_vector.shape[0]):
                res[i] = np.linalg.solve(amplitudes_matrix[i], data_prof_vector[i])
            return res
        else:
            return np.linalg.solve(amplitudes_matrix, data_prof_vector)

    @staticmethod
    def compute_profile(
        times: float,
        arrival_time: float,
        sc_time_ref: float,
        sc_index: float,
        width: float,
        freqs: float,
        ref_freq: float,
        is_folded: bool = False,
    ):
        times_copy = times.copy()
        if is_folded:
            res_time = np.diff(times_copy, axis=1)[:, 0]
            start = times[:, -1] + res_time
            stop = times[:, -1] + (res_time * times.shape[1])
            times_extended = np.linspace(
                start=start, stop=stop, num=times.shape[1], axis=1
            )
            times_copy = np.append(times, times_extended, axis=1)

        profile = np.zeros(times_copy.shape, dtype=float)
        sc_time = sc_time_ref * (freqs / ref_freq) ** sc_index
        normalize = general.get("flags", False)
        if type(normalize) == dict:
            normalize = normalize.get("normalize_pbf", False)

        if np.any(sc_time > 0.0):
            threshold = (np.array(width) * -5).reshape(1, 1, len(width))
            mask = times_copy < threshold
            replacement_value = np.tile(
                (-5 * np.array(width)).reshape(1, 1, (len(width))),
                (times_copy.shape[0], times_copy.shape[1], 1),
            )

            times_copy[mask] = replacement_value[mask]

            profile = FitBurstModel.compute_profile_pbf(
                times_copy,
                arrival_time,
                width,
                freqs,
                ref_freq,
                sc_time_ref,
                sc_index=sc_index,
                normalize=normalize,
            )
        else:
            profile = FitBurstModel.compute_profile_gaussian(
                times_copy, arrival_time, width
            )

        if is_folded:
            profile = profile.reshape(
                times.shape[0], 2, times.shape[1], times.shape[2]
            ).mean(axis=1)

        return profile

    def compute_model(self, data=None):
        if self.scintillation and data is None:
            sys.exit("ERROR: scintillation modelling is desired by data are missing!")

        if self.verbose:
            for current_component in range(self.num_components):
                current_amplitude = self.amplitude[current_component]
                current_arrival_time = self.arrival_time[current_component]
                current_dm = self.dm[0]
                current_dm_index = self.dm_index[0]
                current_ref_freq = self.ref_freq[current_component]
                current_sc_idx = self.scattering_index[0]
                current_sc_time = self.scattering_timescale[0]
                current_sp_idx = self.spectral_index[current_component]
                current_sp_run = self.spectral_running[current_component]
                current_width = self.burst_width[current_component]

                if self.scintillation:
                    print(
                        f"{current_dm:.5f} {current_arrival_time:.5f} ",
                        f"{current_sc_idx:.5f}  {current_sc_time:.5f}  {current_width:.5f}",
                    )
                else:
                    print(
                        f"{current_dm:.5f}  {current_amplitude:.5f}  {current_arrival_time:.5f}  ",
                        f"{current_sc_idx:.5f}  {current_sc_time:.5f}  {current_width:.5f} {current_sp_idx:.5f}  {current_sp_run:.5f}",
                    )

        # create an upsampled version of the current frequency label.
        # even if no upsampling is desired, this will return an array
        # of length 1.
        upsampled_frequencies = rt.manipulate.upsample_1d(
            self.freqs, self.res_freq, self.factor_freq_upsample
        )

        # first, compute "full" delays for all upsampled frequency labels.
        dm_delay = rt.ism.compute_time_dm_delay(
            self.dm_incoherent + self.dm[0],
            general["constants"]["dispersion"],
            self.dm_index[0],
            upsampled_frequencies,
            self.ref_freq[0],
        )

        # then compute "relative" delays with respect to central frequency.
        dm_delay -= np.repeat(
            rt.ism.compute_time_dm_delay(
                self.dm_incoherent,
                general["constants"]["dispersion"],
                self.dm_index[0],
                self.freqs,
                self.ref_freq[0],
            ),
            self.factor_freq_upsample,
        )

        # create a 2D array for each component containing the times and remove the toa.
        time_dt = self.times.copy()
        new_time_dt = np.array([time_dt] * self.num_components)
        toas = np.tile(
            np.array(self.arrival_time)[:, np.newaxis], (1, time_dt.shape[0])
        )

        # now compute current-times array corrected for relative delay.
        upsampled_times = rt.manipulate.upsample_1d(
            (new_time_dt - toas).T, self.res_time, self.factor_time_upsample
        )

        # # the meshgrid stuff -> removing the dm delay
        upsampled_times = upsampled_times[np.newaxis, :, :]
        # Shape of tile: (num_upsampled_freqs, num_upsampled_times, num_components)
        tile = np.tile(upsampled_times, (upsampled_frequencies.shape[0], 1, 1))
        tile -= dm_delay[:, None, None]

        self.timediff_per_component = self.downsample_tile(
            tile, [self.factor_freq_upsample, self.factor_time_upsample, 1]
        )

        # next, compute and store raw temporal profile.
        profile = self.compute_profile(
            tile,
            0,
            self.scattering_timescale[0],
            self.scattering_index[0],
            self.burst_width,
            upsampled_frequencies,
            self.ref_freq[0],
            self.is_folded,
        )

        self.timeprof_per_component = self.downsample_tile(
            profile, [self.factor_freq_upsample, self.factor_time_upsample, 1]
        )

        profile *= self.compute_spectrum_rpl(
            upsampled_frequencies,
            self.ref_freq[0],
            self.spectral_index,
            self.spectral_running,
        )[:, None, :]

        profile = self.downsample_tile(
            profile, [self.factor_freq_upsample, self.factor_time_upsample, 1]
        )

        self.amplitude_per_component = np.tile(
            (
                self.compute_spectrum_rpl(
                    self.freqs,
                    self.ref_freq[0],
                    self.spectral_index,
                    self.spectral_running,
                )
                * (10 ** np.array(self.amplitude))
            )[:, np.newaxis, :],
            (1, len(self.times), 1),
        )

        self.spectrum_per_component = (10 ** np.array(self.amplitude)) * profile

        # if desired, then compute per-channel amplitudes in cases where scintillation is significant.
        if self.scintillation:
            current_amplitudes = np.tile(
                self.compute_amplitude_per_channel(data, self.timeprof_per_component)[
                    :, np.newaxis, :
                ],
                (1, len(self.times), 1),
            )

            self.amplitude_per_component = current_amplitudes
            self.spectrum_per_component = (
                current_amplitudes * self.timeprof_per_component
            )

        return np.sum(self.spectrum_per_component, axis=2)

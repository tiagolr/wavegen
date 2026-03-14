#!/usr/bin/env python3
"""
Copyright 2019 Harvey Ormston

This file is part of osc_gen.

    osc_gen is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    osc_gen is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with osc_gen.  If not, see <https://www.gnu.org/licenses/>.
"""

import math
from copy import deepcopy

import numpy as np


def normalize(inp):
    """Normalize a signal to the range +/- 1

    @param inp seq : A sequence of samples
    """

    dc_bias = (np.amax(inp) + np.amin(inp)) / 2
    inp -= dc_bias
    amp = np.amax(np.absolute(inp))

    if amp > 0:
        inp /= amp

    return inp


def mix(inp_a, inp_b, amount=0.5):
    """Mix two signals together.

    @param inp_a np.ndarray : first input
    @param inp_b np.ndarray : seconds input
    @param amount float : mix amount, 0 outputs only inp_a, 1 outputs only
        inp_b, values between 0 and 1 output a propotional mix of the two.
    """

    amount = np.clip(amount, 0, 1)

    return normalize((inp_a * (1 - amount) + inp_b * amount))


def fundamental(inp, fs):
    """Find the fundamental frequency in Hz of a given input"""

    window = np.hamming(inp.size)
    sig = np.fft.fft(inp * window)
    freqs = np.fft.fftfreq(sig.size)
    i = np.argmax(np.abs(sig))

    return np.abs(freqs[i] * fs)


def harmonic_series(inp):
    """Find the harmonic series of a periodic input"""

    fft_mult = min(64, inp.size // 501)
    fft_mult = max(fft_mult, 1)
    fft_len = 501 * fft_mult

    # if the input has insufficient data, loop it a number of times
    while inp.size < fft_len:
        inp = np.tile(inp, 2)

    # produce symmetrical, windowed fft
    idx1 = int(np.floor((fft_len + 1) / 2))
    idx2 = int(np.floor(fft_len / 2))
    windowed = inp[:fft_len] * np.hamming(fft_len)
    fft_half = 1024 * fft_mult
    buf = np.zeros(fft_half)
    buf[:idx1] = windowed[idx2:]
    buf[fft_half - idx2 :] = windowed[:idx2]
    fft = np.fft.fft(buf)[: fft_half // 2]

    # peak amplitude assumed to be fundamental frequency
    i_fund = np.argmax(np.abs(fft))
    i_fund = max(i_fund, 4)

    # get fft components from only the harmonics, harmonics are picked by
    # taking the value with the highest amplitude around each harmonic
    # frequency
    start = i_fund // 4
    harmonics = np.array(
        [
            fft[i - start : i + start][np.abs(fft[i - start : i + start]).argmax()]
            for i in range(i_fund, fft_half // 2, i_fund)
        ]
    )

    # normalize magnitude and phase
    hs_amp = np.abs(harmonics)
    hs_ang = np.angle(harmonics)
    harmonics = hs_amp * np.exp(1j * (hs_ang - hs_ang[0])) / hs_amp[0]

    return harmonics


def slice_cycles(inp, n, fs):
    """Extact n single-cycle slices from a signal"""

    def nearest(arr, val):
        """find the nearest value in an array to a given value"""
        return arr[np.argmin(np.abs(arr - val))]

    zero_crossings = np.where(np.diff(np.sign(inp)) > 0)[0] + 1

    if not zero_crossings.size:
        raise ValueError("No zero crossings found.")

    freq = fundamental(inp, fs)
    samples_per_cycle = fs / freq
    end = len(inp) - samples_per_cycle

    slots = np.linspace(0, end, n)
    slots = np.around(slots).astype(int)
    slots = np.unique([nearest(zero_crossings, slot) for slot in slots])

    return [inp[x : x + int(samples_per_cycle)] for x in slots]


def resynthesize(inp, sig_gen):
    """
    Resynthesize a signal from its harmonic series

    @param sig_gen SigGen : SigGen to use for regenerating the signal.
    """

    sine_gen = deepcopy(sig_gen)
    max_harmonic = sig_gen.num_points // 2
    harmonics = harmonic_series(inp)
    outp = np.zeros(sine_gen.num_points)

    for i, harmonic in enumerate(harmonics):

        sine_gen.harmonic = i
        sine_gen.amp = np.abs(harmonic)
        sine_gen.phase = np.angle(harmonic)

        outp += sine_gen.sin()

        if i >= max_harmonic:
            break

    return normalize(outp)
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

import matplotlib.pyplot as plt
import numpy as np

DARKGREY = "#222222"
LIGHTGREY = "#555555"

plt.style.use("dark_background")
plt.rcParams["axes.facecolor"] = DARKGREY
plt.rcParams["axes.edgecolor"] = LIGHTGREY
plt.rcParams["axes.labelcolor"] = LIGHTGREY
plt.rcParams["patch.edgecolor"] = LIGHTGREY
plt.rcParams["savefig.facecolor"] = DARKGREY
plt.rcParams["figure.facecolor"] = DARKGREY

CMAP = plt.get_cmap("cool")


def plot_wave(wave, title="", save=False):
    """Plot a single wave"""

    plt.plot(wave, color=CMAP(0))

    if not save:
        plt.title(title, color=LIGHTGREY)

    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.spines["bottom"].set_color(LIGHTGREY)
    frame.spines["top"].set_color(LIGHTGREY)
    frame.xaxis.label.set_color(LIGHTGREY)
    frame.tick_params(axis="x", colors=DARKGREY)
    frame.tick_params(axis="y", colors=DARKGREY)

    plt.grid(True, color=LIGHTGREY)
    plt.tight_layout(pad=0.0)

    if save:
        plt.savefig(save)
    else:
        plt.show()

    plt.gcf().clear()


def plot_wavetable(wavetable, title="", spacing=None, alpha=0.75):
    """Plot all waves in a wavetable in 3D, overlapping along Y like a real wavetable."""
    wavetable = np.array(wavetable)
    numtables = len(wavetable)

    # Colormap
    cmap = plt.get_cmap('cool')
    colors = [cmap(i) for i in np.linspace(0, 1, numtables)]

    # Prepare 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("white")

    # Plot each wave with slight Y offset
    for i, wave in enumerate(wavetable):
        x = np.arange(len(wave))          # sample index
        y = np.full(len(wave))  # offset along Y
        z = wave                           # amplitude along Z
        ax.plot(x, y, z, color=colors[i], alpha=alpha)

    # Labels and title
    ax.set_xlabel("Sample Index", color=DARKGREY)
    ax.set_ylabel("Wave Slot", color=DARKGREY)
    ax.set_zlabel("Amplitude", color=DARKGREY)
    ax.set_title(title, color=LIGHTGREY)

    plt.show()
import os
from pathlib import Path

import matplotlib.pyplot as plt
import xhermes
from matplotlib.animation import PillowWriter

this_dir = Path(__file__).parent


class DashBoard:
    def __init__(self, ds, ncols=1):
        self.ds = ds
        self.panels = {}
        self.panel_kws = {}
        self.ncols = ncols

    def add_panel(self, panel_name, **kws):
        self.panels[panel_name] = []
        self.panel_kws[panel_name] = kws

    def add_plot(self, panel_name, var_name, label, **kws):
        if panel_name not in self.panels:
            raise ValueError(
                f"Panel {panel_name} does not exist. Call add_panel first."
            )
        self.panels[panel_name].append((var_name, label, kws))

    def plot(self):
        n_panels = len(self.panels)
        if self.ncols > 1:
            n_rows = (n_panels + self.ncols - 1) // self.ncols
        else:
            n_rows = n_panels

        fig, axes = plt.subplots(
            n_rows, self.ncols, figsize=(8 * self.ncols, 4 * n_rows)
        )
        # Make it work with single panel
        if n_panels == 1:
            axes = [axes]

        axes_flat = axes.flatten()[:n_panels]

        for ax, (panel_name, plots) in zip(axes_flat, self.panels.items()):
            for var_name, label, kws in plots:
                self.ds[var_name].plot(ax=ax, label=label, **kws)
            panel_kws = self.panel_kws[panel_name].copy()
            plot_legend = panel_kws.pop("plot_legend", True)
            if plot_legend:
                ax.legend()
            panel_title = panel_kws.pop("title", panel_name)
            ax.set(**panel_kws)
            ax.set_title(panel_title)

        plt.tight_layout()
        return fig

    def save(self, outpath, **kws):
        fig = self.plot()
        fig.savefig(outpath, **kws)
        print(f"Saved plot to {outpath}")


def last_output_dashboard(ds, outpath):
    dashboard = DashBoard(ds, ncols=2)

    dashboard.add_panel("Densities", ylabel="Density [m^-3]", yscale="log")
    dashboard.add_plot("Densities", "Ne", "Electrons")
    dashboard.add_plot("Densities", "Nd+", "Ions")
    dashboard.add_plot("Densities", "Nd", "Neutrals")
    dashboard.add_plot("Densities", "Nd2", "Molecules")

    dashboard.add_panel("Temperatures", ylabel="Temperature [eV]", yscale="log")
    dashboard.add_plot("Temperatures", "Te", "Electrons")
    dashboard.add_plot("Temperatures", "Td+", "Ions")
    dashboard.add_plot("Temperatures", "Td", "Neutrals")
    dashboard.add_plot("Temperatures", "Td2", "Molecules")

    dashboard.add_panel("Pressures", ylabel="Pressure", yscale="log")
    dashboard.add_plot("Pressures", "Pe", "Electrons")
    dashboard.add_plot("Pressures", "Pd+", "Ions")
    dashboard.add_plot("Pressures", "Pd", "Neutrals")
    dashboard.add_plot("Pressures", "Pd2", "Molecules")

    dashboard.save(outpath, dpi=300)


def all_outputs_dashboard(ds, outpath):
    dashboard = DashBoard(ds, ncols=2)

    dashboard.add_panel("Ne", ylabel="Density [m^-3]", yscale="log", plot_legend=False)
    dashboard.add_plot("Ne", "Ne", "Electrons")

    dashboard.add_panel("Nd", ylabel="Density [m^-3]", yscale="log", plot_legend=False)
    dashboard.add_plot("Nd", "Nd", "Neutrals")

    dashboard.add_panel("Nd+", ylabel="Density [m^-3]", yscale="log", plot_legend=False)
    dashboard.add_plot("Nd+", "Nd+", "Ions")

    dashboard.add_panel("Nd2", ylabel="Density [m^-3]", yscale="log", plot_legend=False)
    dashboard.add_plot("Nd2", "Nd2", "Molecules")

    dashboard.save(outpath, dpi=300)


def animation(ds, flds, outpath=None, **kws):
    for fld in flds:
        if type(fld) is list:
            assert all(isinstance(f, str) for f in fld), (
                "All elements in a sublist of flds must be strings"
            )
        elif type(fld) is str:
            pass
        else:
            raise ValueError(
                "Each element of flds must be either a string or a list of strings"
            )
    ds = ds.squeeze()
    if outpath is None:
        outpath = this_dir / "kotov_animation"
    figsize = kws.pop("figsize", (18, 8))
    mask_nonpositive = kws.pop("mask_nonpositive", False)
    if mask_nonpositive:
        ds = ds.copy()
        field_names = [
            field_name
            for field_group in flds
            for field_name in (
                field_group if isinstance(field_group, list) else [field_group]
            )
        ]
        for field_name in field_names:
            ds[field_name] = ds[field_name].where(ds[field_name] > 0)

    with plt.rc_context({"figure.figsize": figsize}):
        animation = ds.bout.animate_list(flds, show=False, save_as=None, **kws)

        plot_axes = {block.ax for block in animation.blocks}
        for axis in list(animation.blocks[0].ax.figure.axes):
            if axis not in plot_axes:
                axis.remove()
        for axis in plot_axes:
            axis.legend(loc="lower left")
        plot_axes_in_order = list(dict.fromkeys(block.ax for block in animation.blocks))
        for axis, ylabel in zip(
            plot_axes_in_order,
            [
                "number density [normalised]",
                "Temperature [normalised] ",
                "Pressure [normalised] ",
            ],
        ):
            axis.set_ylabel(ylabel)

        animation.blocks[0].ax.figure.subplots_adjust(hspace=0.25)
        animation.save(outpath.__str__() + ".gif", writer=PillowWriter(fps=8))


if __name__ == "__main__":
    suffix = "_orig_long_lowndt1e4"
    debug = False
    lbl = f"low_n_srcs{suffix}"
    if debug:
        lbl += "_debug"
    data_dir = f"/home/oparry/code/arch-H3/kotov_testing/{lbl}"
    print(f"Plotting data from {data_dir}")
    output_dir = this_dir / "kotov_testing" / lbl
    os.makedirs(output_dir, exist_ok=True)
    try:
        ds = xhermes.open(data_dir, unnormalise=False)
    except OSError:
        raise OSError(f"Error when trying to read data from {data_dir}.")

    last_output_dashboard(ds.isel(t=-1), output_dir / "kotov_last_output.png")
    # all_outputs_dashboard(ds.isel(pos=slice(2, -2)), output_dir / "kotov_evo.png")

    anim_outpath = output_dir / "kotov_animation"
    animation(
        ds,
        [
            ["Ne", "Nd", "Nd+", "Nd2", "Nd2+"],
            ["Te", "Td", "Td+", "Td2", "Td2+"],
            ["Pe", "Pd", "Pd+", "Pd2", "Pd2+"],
        ],
        titles=["Densities", "Temperatures", "Pressures"],
        outpath=anim_outpath,
        logscale=True,
        mask_nonpositive=True,
        fps=8,
        tight_layout=False,
        controls=None,
    )
    print(f"Saved animation to {anim_outpath}.gif")

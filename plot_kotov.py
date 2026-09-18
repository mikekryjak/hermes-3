import os
from pathlib import Path

import matplotlib.pyplot as plt
import xhermes

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
            plot_legend = self.panel_kws[panel_name].pop("plot_legend", True)
            if plot_legend:
                ax.legend()
            panel_title = self.panel_kws[panel_name].get("title", "")
            ax.set(title=panel_title, **self.panel_kws[panel_name])

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
    with plt.rc_context({"figure.figsize": figsize}):
        ds.bout.animate_list(flds, show=False, save_as=str(outpath), **kws)


if __name__ == "__main__":
    lbl = "NPD_all"
    data_dir = f"/home/oparry/code/arch-H3/kotov_testing/{lbl}"
    output_dir = this_dir / lbl
    os.makedirs(output_dir, exist_ok=True)
    try:
        ds = xhermes.open(data_dir, unnormalise=True)
    except OSError:
        raise OSError(f"Error when trying to read data from {data_dir}.")

    last_output_dashboard(ds.isel(t=-1), output_dir / "kotov_last_output.png")
    # all_outputs_dashboard(ds.isel(pos=slice(2, -2)), output_dir / "kotov_evo.png")

    anim_outpath = output_dir / "kotov_animation"
    animation(
        ds,
        [
            ["Ne", "Nd", "Nd+", "Nd2"],
            ["Te", "Td", "Td+", "Td2"],
            ["Pe", "Pd", "Pd+", "Pd2"],
        ],
        outpath=anim_outpath,
        logscale=True,
        fps=8,
        tight_layout=False,
        controls=None,
    )
    print(f"Saved animation to {anim_outpath}.gif")

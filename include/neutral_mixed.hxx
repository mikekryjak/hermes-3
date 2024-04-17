
#pragma once
#ifndef NEUTRAL_MIXED_H
#define NEUTRAL_MIXED_H

#include <memory>
#include <string>

#include <bout/invert_laplace.hxx>

#include "component.hxx"

/// Evolve density, parallel momentum and pressure
/// for a neutral gas species with cross-field diffusion
struct NeutralMixed : public Component {
  ///
  /// @param name     The name of the species e.g. "h"
  /// @param options  Top-level options. Settings will be taken from options[name]
  /// @param solver   Time-integration solver to be used
  NeutralMixed(const std::string& name, Options& options, Solver *solver);
  
  /// Modify the given simulation state
  void transform(Options &state) override;
  
  /// Use the final simulation state to update internal state
  /// (e.g. time derivatives)
  void finally(const Options &state) override;

  /// Add extra fields for output, or set attributes e.g docstrings
  void outputVars(Options &state) override;

  /// Preconditioner
  void precon(const Options &state, BoutReal gamma) override;
private:
  std::string name;  ///< Species name
  
  Field3D Nn, Pn, NVn; // Density, pressure and parallel momentum
  Field3D Vn; ///< Neutral parallel velocity
  Field3D Vth; ///< Thermal velocity of Maxwellian in one direction
  Field3D Tn; ///< Neutral temperature
  Field3D Nnlim, Pnlim, logPnlim, Vnlim, Tnlim; // Limited in regions of low density

  BoutReal AA; ///< Atomic mass (proton = 1)

  Field3D Dnn; ///< Diffusion coefficient
  Field3D DnnNn, DnnPn, DnnTn, DnnNVn; ///< Used for operators
  Field3D eta_n; ///< Viscosity
  Field3D kappa_n; ///< Thermal conductivity

  bool sheath_ydown, sheath_yup;

  BoutReal nn_floor; ///< Minimum Nn used when dividing NVn by Nn to get Vn.
  BoutReal pn_floor; ///< Minimum Pn used when dividing Pn by Nn to get Tn.

  bool flux_limit; ///< Impose flux limiter?
  bool particle_flux_limiter, heat_flux_limiter, momentum_flux_limiter; ///< Which limiters to impose
  BoutReal maximum_mfp; ///< Maximum mean free path for diffusion. 0.1 by default, -1 is off.
  BoutReal flux_limit_alpha, heat_flux_limit_alpha, mom_flux_limit_alpha;
  BoutReal flux_limit_gamma;
  Field3D particle_flux_factor; ///< Particle flux scaling factor
  Field3D momentum_flux_factor;
  Field3D heat_flux_factor;

  Field3D SPd_par_adv, SPd_par_compr, SPd_perp_adv, SPd_perp_compr, SPd_perp_cond, SPd_par_cond, SPd_src, SPd_ext_src, SPd_visc_heat; ///< Neutral pressure terms

  bool neutral_viscosity; ///< include viscosity?
  bool evolve_momentum; ///< Evolve parallel momentum?

  bool precondition {true}; ///< Enable preconditioner?
  std::unique_ptr<Laplacian> inv; ///< Laplacian inversion used for preconditioning

  Field3D density_source, pressure_source; ///< External input source
  Field3D Sn, Sp, Snv; ///< Particle, pressure and momentum source

  bool output_ddt; ///< Save time derivatives?
  bool diagnose, diagnose_eqns; ///< Save additional diagnostics?
  BoutReal perp_pressure_form, perp_cond_form, kappa_form, eta_form; ///< Form of the perpendicular neutral pressure terms
  bool upwind_perp_diffusion; ///< Use a more dissipative perpendicular diffusion operator?
};

namespace {
RegisterComponent<NeutralMixed> registersolverneutralmixed("neutral_mixed");
}

#endif // NEUTRAL_MIXED_H

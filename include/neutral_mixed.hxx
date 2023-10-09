
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
  Field3D Tn; ///< Neutral temperature
  Field3D Nnlim, Pnlim, logPnlim, Vnlim, Tnlim; // Limited in regions of low density

  BoutReal AA; ///< Atomic mass (proton = 1)

  Field3D Dnn; ///< Diffusion coefficient
  Field3D DnnNn, DnnPn, DnnTn, DnnNVn; ///< Used for operators
  Field3D eta_n; ///< Viscosity
  Field3D kappa_n; ///< Thermal conductivity

  bool sheath_ydown, sheath_yup;

  BoutReal nn_floor; ///< Minimum Nn used when dividing NVn by Nn to get Vn.

  bool flux_limit; ///< Impose flux limiter?
  bool separate_heat_limiters;
  bool particle_flux_limiter, heat_flux_limiter, conv_heat_flux_limiter, cond_heat_flux_limiter, momentum_flux_limiter; ///< Which limiters to impose
  BoutReal maximum_mfp; ///< Maximum mean free path for diffusion. 0.1 by default, -1 is off.
  BoutReal flux_limit_alpha, cond_flux_limit_alpha, conv_flux_limit_alpha, mom_flux_limit_alpha;
  BoutReal flux_limit_gamma;
  Field3D particle_flux_factor; ///< Particle flux scaling factor
  Field3D momentum_flux_factor;
  Field3D heat_flux_factor, cond_heat_flux_factor, conv_heat_flux_factor;

  BoutReal flux_factor_timescale; ///< Timescale over which flux factors vary
  BoutReal flux_factor_time {-1.0}; ///< Last time the flux factors were updated
  Field3D particle_flux_avg; ///< Moving average flux factor
  Field3D momentum_flux_avg; ///< Moving average flux factor
  Field3D heat_flux_avg, cond_heat_flux_avg, conv_heat_flux_avg; ///< Moving average flux factor

  bool neutral_viscosity; ///< include viscosity?

  bool precondition {true}; ///< Enable preconditioner?
  std::unique_ptr<Laplacian> inv; ///< Laplacian inversion used for preconditioning

  Field3D density_source, pressure_source; ///< External input source
  Field3D Sn, Sp, Snv; ///< Particle, pressure and momentum source

  bool output_ddt; ///< Save time derivatives?
  bool diagnose; ///< Save additional diagnostics?
};

namespace {
RegisterComponent<NeutralMixed> registersolverneutralmixed("neutral_mixed");
}

#endif // NEUTRAL_MIXED_H

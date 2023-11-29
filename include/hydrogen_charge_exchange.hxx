#pragma once
#ifndef HYDROGEN_CHARGE_EXCHANGE_H
#define HYDROGEN_CHARGE_EXCHANGE_H

#include <bout/constants.hxx>

#include "component.hxx"

/// Hydrogen charge exchange total rate coefficient
///
///   p + H(1s) -> H(1s) + p
///
/// Reaction 3.1.8 from Amjuel (p43)
///
/// Scaled to different isotope masses and finite neutral particle
/// temperatures by using the effective temperature (Amjuel p43)
///
/// T_eff = (M/M_1)T_1 + (M/M_2)T_2
///
///
/// Important: If this is included then ion_neutral collisions
///            should probably be disabled in the `collisions` component,
///            to avoid double-counting.
///
struct HydrogenChargeExchange : public Component {
  ///
  /// @param alloptions Settings, which should include:
  ///        - units
  ///          - eV
  ///          - inv_meters_cubed
  ///          - seconds
  HydrogenChargeExchange(std::string name, Options& alloptions, Solver*) {
    // Get the units
    const auto& units = alloptions["units"];
    Tnorm = get<BoutReal>(units["eV"]);
    Nnorm = get<BoutReal>(units["inv_meters_cubed"]);
    FreqNorm = 1. / get<BoutReal>(units["seconds"]);
  }

protected:
  BoutReal Tnorm, Nnorm, FreqNorm; ///< Normalisations

  /// Calculate the charge exchange cross-section
  ///
  /// atom1 + ion1 -> atom2 + ion2
  ///
  /// and transfer of mass, momentum and energy from:
  ///
  /// atom1 -> ion2, ion1 -> atom2
  ///
  /// Assumes that both atom1 and ion1 have:
  ///   - AA
  ///   - density
  ///   - velocity
  ///   - temperature
  ///
  /// Sets in all species:
  ///   - density_source     [If atom1 != atom2 or ion1 != ion2]
  ///   - momentum_source
  ///   - energy_source
  ///
  /// Modifies collision_frequency for atom1 and ion1
  ///
  /// Diagnostic output
  ///  R            Reaction rate, transfer of particles in case of different isotopes
  ///  atom_mom     Momentum removed from atom1, added to ion2
  ///  ion_mom      Momentum removed from ion1, added to atom2
  ///  atom_energy  Energy removed from atom1, added to ion2
  ///  ion_energy   Energy removed from ion1, added to atom2
  ///
  void calculate_rates(Options& atom1, Options& ion1, Options& atom2, Options& ion2,
                       Field3D& R, Field3D& atom_mom, Field3D& ion_mom,
                       Field3D& atom_energy, Field3D& ion_energy);
};

/// Hydrogen charge exchange
/// Templated on a char to allow 'h', 'd' and 't' species to be treated with the same code
///
/// @tparam Isotope1   The isotope ('h', 'd' or 't') of the initial atom
/// @tparam Isotope2   The isotope ('h', 'd' or 't') of the initial ion
///
///   atom   +   ion     ->   ion      +    atom
/// Isotope1 + Isotope2+ -> Isotope1+  +  Isotope2
///
/// Diagnostics
/// -----------
///
/// If diagnose = true is set in the options, then the following diagnostics are saved:
///   - F<Isotope1><Isotope2>+_cx  (e.g. Fhd+_cx) the momentum added to Isotope1 atoms due
///                                due to charge exchange with Isotope2 ions.
///                                There is a corresponding loss of momentum for the
///                                Isotope1 ions d/dt(NVh)  = ... + Fhd+_cx   // Atom
///                                momentum source d/dt(NVh+) = ... - Fhd+_cx   // Ion
///                                momentum sink
///   - E<Isotope1><Isotope2>+_cx  Energy added to Isotope1 atoms due to charge exchange
///   with
///                                Isotope2 ions. This contributes to two pressure
///                                equations d/dt(3/2 Ph)  = ... + Ehd+_cx d/dt(3/2 Ph+) =
///                                ... - Ehd+_cx
///
/// If Isotope1 != Isotope2 then there is also the source of energy for Isotope2 atoms
/// and a source of particles:
///   - F<Isotope2>+<Isotope1>_cx  Source of momentum for Isotope2 ions, sink for Isotope2
///   atoms
///   - E<Isotope2>+<Isotope1>_cx  Source of energy for Isotope2 ions, sink for Isotope2
///   atoms
///   - S<Isotope1><Isotope2>+_cx  Source of Isotope1 atoms due to charge exchange with
///   Isotope2 ions
///                                Note: S<Isotope2><Isotope1>+_cx =
///                                -S<Isotope1><Isotope2>+_cx For example Shd+_cx
///                                contributes to four density equations: d/dt(Nh)  = ...
///                                + Shd+_cx d/dt(Nh+) = ... - Shd+_cx d/dt(Nd)  = ... -
///                                Shd+_cx d/dt(Nd+) = ... + Shd+_cx
///
template <char Isotope1, char Isotope2, char Kind>
struct HydrogenChargeExchangeIsotope : public HydrogenChargeExchange {
  HydrogenChargeExchangeIsotope(std::string name, Options& alloptions, Solver* solver)
      : HydrogenChargeExchange(name, alloptions, solver) {

    diagnose = alloptions[name]["diagnose"]
                   .doc("Output additional diagnostics?")
                   .withDefault<bool>(false);
  }

  void transform(Options& state) override {
    Field3D R, atom_mom, ion_mom, atom_energy, ion_energy;

    // Prepare species names
    std::string atom1{Isotope1};
    std::string ion1{Isotope1, '+'};
    std::string atom2{Isotope2};
    std::string ion2{Isotope2, '+'};

    // CX for hot neutral species: add * identifier
    if (Kind == '*') {
      atom1 += '*';
      atom2 += '*';

    // CX for cold -> hot neutrals: add * on product neutral only
    } else if (Kind == 'x') {
      atom2 += '*';
    }

    calculate_rates(state["species"][atom1],             // e.g. "h"
                    state["species"][ion1],              // e.g. "d+"
                    state["species"][atom2],             // e.g. "d"
                    state["species"][ion2],              // e.g. "h+"
                    R, atom_mom, ion_mom, atom_energy, ion_energy); // Transfer channels

    if (diagnose) {
      // Calculate diagnostics to be written to dump file
      if (Isotope1 == Isotope2) and (atom1 == atom2) {
        // Simpler case of same isotopes and atoms
        //  - No net particle source/sink
        //  - atoms lose atom_mom, gain ion_mom
        //
        F = ion_mom - atom_mom;       // Momentum transferred to atoms due to CX with ions
        E = ion_energy - atom_energy; // Energy transferred to atoms

      } else if (Isotope1 == Isotope2) and (atom1 != atom2) {
        // Same isotopes, but hot and cold neutral species

        // Properties are transferred from LHS to RHS of reaction:
        // atom1 + ion1 -> atom2 + ion2
        // Energy and momentum flow:
        // - From atom1 (cold) to ion2
        // - From ion1 to atom2 (hot)

        S = R;               // Source of hot atoms (CX always source of hot atoms)
        F = -atom_mom;       // Cold neutrals lose momentum
        Fhot = -ion_mom;     // Hot neutrals gain LHS ion momentum
        Fi = -atom_mom;      // RHS ions get cold neutral momentum

        E = -atom_energy;    // Cold neutrals lose energy
        Ehot = -ion_energy;  // Hot neutrals gain LHS ion energy
        Ei = -atom_energy;   // RHS ions get cold neutral energy

      } else if (Isotope1 != Isotope2) and (atom1 == atom2) {
        // Different isotopes
        S = -R;           // Source of Isotope1 atoms
        F = -atom_mom;    // Source of momentum for Isotope1 atoms
        F2 = -ion_mom;    // Source of momentum for Isotope2 ions
        E = -atom_energy; // Source of energy for Isotope1 atoms
        E2 = -ion_energy; // Source of energy for Isotope2 ions

      } else if (Isotope1 != Isotope2) and (atom1 != atom2) {
        throw BoutException("Multigroup neutral model not yet implemented for more than one plasma isotope");
      }
    }
  }

  void outputVars(Options& state) override {
    AUTO_TRACE();
    // Normalisations
    auto Nnorm = get<BoutReal>(state["Nnorm"]);
    auto Tnorm = get<BoutReal>(state["Tnorm"]);
    BoutReal Pnorm = SI::qe * Tnorm * Nnorm; // Pressure normalisation
    auto Omega_ci = get<BoutReal>(state["Omega_ci"]);
    auto Cs0 = get<BoutReal>(state["Cs0"]);

    if (diagnose) {
      // Save particle, momentum and energy channels

      // Prepare species names
      std::string atom1{Isotope1};
      std::string ion1{Isotope1, '+'};
      std::string atom2{Isotope2};
      std::string ion2{Isotope2, '+'};

      // CX for hot neutral species: add * identifier
      if (Kind == '*') {
        atom1 += '*';
        atom2 += '*';

      // CX for cold -> hot neutrals: add * on product neutral only
      } else if (Kind == 'x') {
        atom2 += '*';
      }

      
      set_with_attrs(state[{'F', Isotope1, Isotope2, '+', '_', 'c', 'x'}], // e.g Fhd+_cx
                     F,
                     {{"time_dimension", "t"},
                      {"units", "kg m^-2 s^-2"},
                      {"conversion", SI::Mp * Nnorm * Cs0 * Omega_ci},
                      {"standard_name", "momentum transfer"},
                      {"long_name", (std::string("Momentum transfer to ") + atom1
                                     + " from " + ion1 + " due to CX with " + ion2)},
                      {"source", "hydrogen_charge_exchange"}});

      set_with_attrs(state[{'E', Isotope1, Isotope2, '+', '_', 'c', 'x'}], // e.g Edt+_cx
                     E,
                     {{"time_dimension", "t"},
                      {"units", "W / m^3"},
                      {"conversion", Pnorm * Omega_ci},
                      {"standard_name", "energy transfer"},
                      {"long_name", (std::string("Energy transfer to ") + atom1 + " from "
                                     + ion1 + " due to CX with " + ion2)},
                      {"source", "hydrogen_charge_exchange"}});

      if (Isotope1 != Isotope2) {
        // Different isotope => particle source, second momentum & energy channel
        set_with_attrs(
            state[{'F', Isotope2, '+', Isotope1, '_', 'c', 'x'}], // e.g Fd+h_cx
            F2,
            {{"time_dimension", "t"},
             {"units", "kg m^-2 s^-2"},
             {"conversion", SI::Mp * Nnorm * Cs0 * Omega_ci},
             {"standard_name", "momentum transfer"},
             {"long_name", (std::string("Momentum transfer to ") + ion2 + " from " + atom2
                            + " due to CX with " + atom1)},
             {"source", "hydrogen_charge_exchange"}});

        set_with_attrs(
            state[{'E', Isotope2, '+', Isotope1, '_', 'c', 'x'}], // e.g Et+d_cx
            E2,
            {{"time_dimension", "t"},
             {"units", "W / m^3"},
             {"conversion", Pnorm * Omega_ci},
             {"standard_name", "energy transfer"},
             {"long_name", (std::string("Energy transfer to ") + ion2 + " from " + atom2
                            + " due to CX with " + atom1)},
             {"source", "hydrogen_charge_exchange"}});

        // Source of isotope1 atoms
        set_with_attrs(
            state[{'S', Isotope1, Isotope2, '+', '_', 'c', 'x'}], // e.g Shd+_cx
            S,
            {{"time_dimension", "t"},
             {"units", "m^-3 s^-1"},
             {"conversion", Nnorm * Omega_ci},
             {"standard_name", "particle transfer"},
             {"long_name", (std::string("Particle transfer to ") + atom1 + " from " + ion1
                            + " due to charge exchange with " + ion2)},
             {"source", "hydrogen_charge_exchange"}});
      }
    }
  }

private:
  bool diagnose; ///< Outputting diagnostics?
  Field3D S;     ///< Particle exchange, used if Isotope1 != Isotope2
  Field3D F, F2; ///< Momentum exchange
  Field3D E, E2; ///< Energy exchange
};

namespace {
/// Register three components, one for each hydrogen isotope
/// so no isotope dependence included.

/// Regular neutrals
RegisterComponent<HydrogenChargeExchangeIsotope<'h', 'h', '.'>>
    register_cx_hh("h + h+ -> h+ + h");
RegisterComponent<HydrogenChargeExchangeIsotope<'d', 'd', '.'>>
    register_cx_dd("d + d+ -> d+ + d");
RegisterComponent<HydrogenChargeExchangeIsotope<'t', 't', '.'>>
    register_cx_tt("t + t+ -> t+ + t");

/// Hot neutrals
RegisterComponent<HydrogenChargeExchangeIsotope<'h', 'h', '*'>>
    register_cx_hh("h* + h+ -> h+ + h*");
RegisterComponent<HydrogenChargeExchangeIsotope<'d', 'd', '*'>>
    register_cx_dd("d* + d+ -> d+ + d*");
RegisterComponent<HydrogenChargeExchangeIsotope<'t', 't', '*'>>
    register_cx_tt("t* + t+ -> t+ + t*");

/// Cold -> hot neutrals
RegisterComponent<HydrogenChargeExchangeIsotope<'h', 'h', 'x'>>
    register_cx_hh("h + h+ -> h+ + h*");
RegisterComponent<HydrogenChargeExchangeIsotope<'d', 'd', 'x'>>
    register_cx_dd("d + d+ -> d+ + d*");
RegisterComponent<HydrogenChargeExchangeIsotope<'t', 't', 'x'>>
    register_cx_tt("t + t+ -> t+ + t*");


// TODO: Implement hot neutrals for these
// Charge exchange between different isotopes
// RegisterComponent<HydrogenChargeExchangeIsotope<'h', 'd'>>
//     register_cx_hd("h + d+ -> h+ + d");
// RegisterComponent<HydrogenChargeExchangeIsotope<'d', 'h'>>
//     register_cx_dh("d + h+ -> d+ + h");

// RegisterComponent<HydrogenChargeExchangeIsotope<'h', 't'>>
//     register_cx_ht("h + t+ -> h+ + t");
// RegisterComponent<HydrogenChargeExchangeIsotope<'t', 'h'>>
//     register_cx_th("t + h+ -> t+ + h");

// RegisterComponent<HydrogenChargeExchangeIsotope<'d', 't'>>
//     register_cx_dt("d + t+ -> d+ + t");
// RegisterComponent<HydrogenChargeExchangeIsotope<'t', 'd'>>
//     register_cx_td("t + d+ -> t+ + d");
} // namespace

#endif // HYDROGEN_CHARGE_EXCHANGE_H

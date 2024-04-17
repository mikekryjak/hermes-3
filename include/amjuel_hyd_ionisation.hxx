#pragma once
#ifndef AMJUEL_HYD_IONISATION_H
#define AMJUEL_HYD_IONISATION_H

#include <bout/constants.hxx>

#include "amjuel_reaction.hxx"

/// Hydrogen ionisation, Amjuel rates
struct AmjuelHydIonisation : public AmjuelReaction {
  AmjuelHydIonisation(std::string name, Options& alloptions, Solver* solver)
      : AmjuelReaction(name, alloptions, solver) {}

  void calculate_rates(Options& electron, Options& atom, Options& ion,
                       Field3D& reaction_rate, Field3D& momentum_exchange,
                       Field3D& energy_exchange, Field3D& energy_loss, BoutReal& rate_multiplier, BoutReal& radiation_multiplier);
};

/// Hydrogen ionisation
/// Templated on a char to allow 'h', 'd' and 't' species to be treated with the same code
template <char Isotope, char Kind>
struct AmjuelHydIonisationIsotope : public AmjuelHydIonisation {
  AmjuelHydIonisationIsotope(std::string name, Options& alloptions, Solver* solver)
      : AmjuelHydIonisation(name, alloptions, solver) {

    diagnose = alloptions[name]["diagnose"]
                   .doc("Output additional diagnostics?")
                   .withDefault<bool>(false);

    rate_multiplier = alloptions[{Isotope}]["K_iz_multiplier"]
                           .doc("Scale the ionisation rate by this factor")
                           .withDefault<BoutReal>(1.0);

    radiation_multiplier = alloptions[{Isotope}]["R_ex_multiplier"]
                           .doc("Scale the ionisation excitation/de-excitation radiation rate by this factor")
                           .withDefault<BoutReal>(1.0);
  }

  void transform(Options& state) override {
    Options& electron = state["species"]["e"];

    std::string atomname{Isotope};
    if (Kind == '*') {
      atomname += "*";
    }

    Options& atom = state["species"][atomname];     // Cold neutral, e.g. "h"
    
    Options& ion = state["species"][{Isotope, '+'}]; // e.g. "h+"
    Field3D reaction_rate, momentum_exchange, energy_exchange, energy_loss;

    calculate_rates(electron, atom, ion, reaction_rate, momentum_exchange,
                    energy_exchange, energy_loss, rate_multiplier, radiation_multiplier);

    if (diagnose) {
      S = reaction_rate;
      F = momentum_exchange;
      E = energy_exchange;
      R = -energy_loss;
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

      std::string atom{Isotope};
      std::string ion{Isotope, '+'};

      // Hot neutrals have a * appended to the name
      if (Kind == '*') {
        atom = atom + '*';
      }
      set_with_attrs(state[std::string("S") + atom + ion + std::string("_iz")], S,
                     {{"time_dimension", "t"},
                      {"units", "m^-3 s^-1"},
                      {"conversion", Nnorm * Omega_ci},
                      {"standard_name", "particle source"},
                      {"long_name", std::string("Ionisation of ") + atom + " to " + ion},
                      {"source", "amjuel_hyd_ionisation"}});

      set_with_attrs(
          state[std::string("F") + atom + ion + std::string("_iz")], F,
          {{"time_dimension", "t"},
           {"units", "kg m^-2 s^-2"},
           {"conversion", SI::Mp * Nnorm * Cs0 * Omega_ci},
           {"standard_name", "momentum transfer"},
           {"long_name", (std::string("Momentum transfer due to ionisation of ") + atom
                          + " to " + ion)},
           {"source", "amjuel_hyd_ionisation"}});

      set_with_attrs(state[std::string("E") + atom + ion + std::string("_iz")], E,
                     {{"time_dimension", "t"},
                      {"units", "W / m^3"},
                      {"conversion", Pnorm * Omega_ci},
                      {"standard_name", "energy transfer"},
                      {"long_name", (std::string("Energy transfer due to ionisation of ")
                                     + atom + " to " + ion)},
                      {"source", "amjuel_hyd_ionisation"}});

      set_with_attrs(state[std::string("R") + atom + ion + std::string("_ex")], R,
                     {{"time_dimension", "t"},
                      {"units", "W / m^3"},
                      {"conversion", Pnorm * Omega_ci},
                      {"standard_name", "radiation loss"},
                      {"long_name", (std::string("Radiation loss due to ionisation of ")
                                     + atom + " to " + ion)},
                      {"source", "amjuel_hyd_ionisation"}});
    }
  }

private:
  bool diagnose; ///< Outputting diagnostics?
  BoutReal rate_multiplier, radiation_multiplier; ///< Scaling factor on reaction rate
  Field3D S;     ///< Particle exchange
  Field3D F;     ///< Momentum exchange
  Field3D E;     ///< Energy exchange
  Field3D R;     ///< Radiation loss
};

namespace {
/// Register three components, one for each hydrogen isotope
/// so no isotope dependence included.
RegisterComponent<AmjuelHydIonisationIsotope<'h', '.'>>
    registerionisation_h("h + e -> h+ + 2e");
RegisterComponent<AmjuelHydIonisationIsotope<'d', '.'>>
    registerionisation_d("d + e -> d+ + 2e");
RegisterComponent<AmjuelHydIonisationIsotope<'t', '.'>>
    registerionisation_t("t + e -> t+ + 2e");

    
RegisterComponent<AmjuelHydIonisationIsotope<'h', '*'>>
    registerionisation_h_hot("h* + e -> h+ + 2e");
RegisterComponent<AmjuelHydIonisationIsotope<'d', '*'>>
    registerionisation_d_hot("d* + e -> d+ + 2e");
RegisterComponent<AmjuelHydIonisationIsotope<'t', '*'>>
    registerionisation_t_hot("t* + e -> t+ + 2e");
} // namespace

#endif // AMJUEL_HYD_IONISATION_H

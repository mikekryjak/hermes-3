#include <iterator>

#include <bout/constants.hxx>

#include "../include/collisions.hxx"

namespace {
BoutReal floor(BoutReal value, BoutReal min) {
  if (value < min)
    return min;
  return value;
}
} // namespace

Collisions::Collisions(std::string name, Options& alloptions, Solver*) {
  AUTO_TRACE();
  const Options& units = alloptions["units"];

  // Normalisations
  Tnorm = units["eV"];
  Nnorm = units["inv_meters_cubed"];
  rho_s0 = units["meters"];
  Omega_ci = 1. / units["seconds"].as<BoutReal>();

  Options& options = alloptions[name];

  electron_electron = options["electron_electron"]
                          .doc("Include electron-electron collisions?")
                          .withDefault<bool>(true);
  electron_ion = options["electron_ion"]
                     .doc("Include electron-ion collisions?")
                     .withDefault<bool>(true);
  electron_neutral = options["electron_neutral"]
                         .doc("Include electron-neutral elastic collisions?")
                         .withDefault<bool>(true);
  ion_ion = options["ion_ion"]
                .doc("Include ion-ion elastic collisions?")
                .withDefault<bool>(true);
  ion_neutral = options["ion_neutral"]
                    .doc("Include ion-neutral elastic collisions?")
                    .withDefault<bool>(true);
  neutral_neutral = options["neutral_neutral"]
                        .doc("Include neutral-neutral elastic collisions?")
                        .withDefault<bool>(true);
}

/// Calculate transfer of momentum and energy between species1 and species2
/// nu_12    normalised frequency
///
/// Modifies
///   species1 and species2
///     - collision_frequency
///     - momentum_source   if species1 or species2 velocity is set
///     - energy_source     if species1 or species2 temperature is set
///
/// Note: A* variables are used for atomic mass numbers;
///       mass* variables are species masses in kg
void Collisions::collide(Options& species1, Options& species2, const Field3D& nu_12) {
  AUTO_TRACE();

  add(species1["collision_frequency"], nu_12);

  if (&species1 != &species2) {
    // For collisions between different species
    // m_a n_a \nu_{ab} = m_b n_b \nu_{ba}

    const BoutReal A1 = get<BoutReal>(species1["AA"]);
    const BoutReal A2 = get<BoutReal>(species2["AA"]);

    const Field3D density1 = get<Field3D>(species1["density"]);
    const Field3D density2 = get<Field3D>(species2["density"]);

    add(species2["collision_frequency"], nu_12 * (A1 / A2) * density1 / density2);

    // Momentum exchange
    if (species1.isSet("velocity") or species2.isSet("velocity")) {

      const Field3D velocity1 =
          species1.isSet("velocity") ? get<Field3D>(species1["velocity"]) : 0.0;
      const Field3D velocity2 =
          species2.isSet("velocity") ? get<Field3D>(species2["velocity"]) : 0.0;

      // F12 is the force on species 1 due to species 2 (normalised)
      const Field3D F12 = nu_12 * A1 * density1 * (velocity2 - velocity1);

      add(species1["momentum_source"], F12);
      subtract(species2["momentum_source"], F12);
    }

    // Energy exchange
    if (species1.isSet("temperature") or species2.isSet("temperature")) {
      // Q12 is heat transferred to species2 (normalised)

      const Field3D temperature1 = get<Field3D>(species1["temperature"]);
      const Field3D temperature2 = get<Field3D>(species2["temperature"]);

      const Field3D Q12 =
          nu_12 * 3. * density1 * (A1 / (A1 + A2)) * (temperature2 - temperature1);

      add(species1["energy_source"], Q12);
      subtract(species2["energy_source"], Q12);
    }
  }
}

void Collisions::transform(Options& state) {
  AUTO_TRACE();

  Options& allspecies = state["species"];

  // Treat electron collisions specially
  // electron-ion and electron-neutral collisions

  if (allspecies.isSection("e")) {
    Options& electrons = allspecies["e"];
    const Field3D Te = get<Field3D>(electrons["temperature"]) * Tnorm; // eV
    const Field3D Ne = get<Field3D>(electrons["density"]) * Nnorm;     // In m^-3
    
    for (auto& kv : allspecies.getChildren()) {
      if (kv.first == "e") {
        ////////////////////////////////////
        // electron-electron collisions

        if (!electron_electron)
          continue;

        const Field3D nu_ee = filledFrom(Ne, [&](auto& i) {
          const BoutReal Telim = floor(Te[i], 0.1);
          const BoutReal Nelim = floor(Ne[i], 1e10);
          const BoutReal logTe = log(Telim);
          // From NRL formulary 2019, page 34
          // Coefficient 30.4 from converting cm^-3 to m^-3
          // Note that this breaks when coulomb_log falls below 1
          const BoutReal coulomb_log = 30.4 - 0.5 * log(Nelim) + (5. / 4) * logTe
                                       - sqrt(1e-5 + SQ(logTe - 2) / 16.);

          const BoutReal v1sq = 2 * Telim * SI::qe / SI::Me;

          // Collision frequency
          const BoutReal nu = SQ(SQ(SI::qe)) * floor(Ne[i], 0.0) * floor(coulomb_log, 1.0)
                              * 2 / (3 * pow(PI * 2 * v1sq, 1.5) * SQ(SI::e0 * SI::Me));

          ASSERT2(std::isfinite(nu));
          return nu;
        });

        collide(electrons, electrons, nu_ee / Omega_ci);
        continue;
      }

      Options& species = allspecies[kv.first]; // Note: Need non-const

      if (species.isSet("charge") and (get<BoutReal>(species["charge"]) != 0.0)) {
        ////////////////////////////////////
        // electron-charged ion collisions

        if (!electron_ion)
          continue;

        const Field3D Ti = get<Field3D>(species["temperature"]) * Tnorm; // eV
        const Field3D Ni = get<Field3D>(species["density"]) * Nnorm;     // In m^-3

        const BoutReal Zi = get<BoutReal>(species["charge"]);
        const BoutReal Ai = get<BoutReal>(species["AA"]);
        const BoutReal me_mi = SI::Me / (SI::Mp * Ai); // m_e / m_i

        const Field3D nu_ei = filledFrom(Ne, [&](auto& i) {
          // NRL formulary 2019, page 34
          const BoutReal coulomb_log =
              ((Te[i] < 0.1) || (Ni[i] < 1e10) || (Ne[i] < 1e10)) ? 10
              : (Te[i] < Ti[i] * me_mi)
                  ? 23 - 0.5 * log(Ni[i]) + 1.5 * log(Ti[i]) - log(SQ(Zi) * Ai)
              : (Te[i] < 10 * SQ(Zi))
                  // Ti m_e/m_i < Te < 10 Z^2
                  ? 31.0 - 0.5 * log(Ne[i]) + log(Te[i])
                  // Ti m_e/m_i < 10 Z^2 < Te
                  : 30.0 - 0.5 * log(Ne[i]) - log(Zi) + 1.5 * log(Te[i]);

          // Calculate v_a^2, v_b^2
          const BoutReal vesq = 2 * floor(Te[i], 0.1) * SI::qe / SI::Me;
          const BoutReal visq = 2 * floor(Ti[i], 0.1) * SI::qe / (SI::Mp * Ai);

          // Collision frequency
          const BoutReal nu = SQ(SQ(SI::qe) * Zi) * floor(Ni[i], 0.0)
                              * floor(coulomb_log, 1.0) * (1. + me_mi)
                              / (3 * pow(PI * (vesq + visq), 1.5) * SQ(SI::e0 * SI::Me));

          ASSERT2(std::isfinite(nu));
          return nu;
        });

        collide(electrons, species, nu_ei / Omega_ci);

      } else {
        ////////////////////////////////////
        // electron-neutral collisions

        if (!electron_neutral)
          continue;

        // Neutral density
        Field3D Nn = get<Field3D>(species["density"]);

        BoutReal a0 = 5e-19; // Cross-section [m^2]

        const Field3D nu_en = filledFrom(Ne, [&](auto& i) {
          // Electron thermal speed (normalised)
          const BoutReal vth_e = sqrt((SI::Mp / SI::Me) * Te[i]);

          // Electron-neutral collision rate
          return vth_e * Nnorm * Nn[i] * a0 * rho_s0;
        });

        collide(electrons, species, nu_en);
      }
    }
  }

  // Iterate through other species
  // To avoid double counting, this needs to iterate over pairs
  // i.e. the diagonal and above
  //
  // Iterators kv1 and kv2 over the species map
  //
  //               kv2 ->
  //             species1  species2  species3
  // kv1   species1     X         X         X
  //  ||   species2               X         X
  //  \/   species3                         X
  //
  const std::map<std::string, Options>& children = allspecies.getChildren();
  for (auto kv1 = std::begin(children); kv1 != std::end(children); ++kv1) {
    if (kv1->first == "e")
      continue; // Skip electrons

    Options& species1 = allspecies[kv1->first];

    // If temperature isn't set, assume zero. in eV
    const Field3D temperature1 = species1.isSet("temperature")
                                     ? get<Field3D>(species1["temperature"]) * Tnorm
                                     : 0.0;

    const Field3D density1 = get<Field3D>(species1["density"]) * Nnorm;

    const BoutReal AA1 = get<BoutReal>(species1["AA"]);
    const BoutReal mass1 = AA1 * SI::Mp; // in Kg

    if (species1.isSet("charge") and (get<BoutReal>(species1["charge"]) != 0.0)) {
      // Charged species
      const BoutReal Z1 = get<BoutReal>(species1["charge"]);
      const BoutReal charge1 = Z1 * SI::qe; // in Coulombs

      // Copy the iterator, so we don't iterate over the
      // lower half of the matrix, but start at the diagonal
      for (std::map<std::string, Options>::const_iterator kv2 = kv1;
           kv2 != std::end(children); ++kv2) {
        if (kv2->first == "e")
          continue; // Skip electrons

        Options& species2 = allspecies[kv2->first];

        // Note: Here species1 could be equal to species2

        // If temperature isn't set, assume zero. in eV
        const Field3D temperature2 = species2.isSet("temperature")
                                         ? get<Field3D>(species2["temperature"]) * Tnorm
                                         : 0.0;

        const Field3D density2 = get<Field3D>(species2["density"]) * Nnorm;

        const BoutReal AA2 = get<BoutReal>(species2["AA"]);
        const BoutReal mass2 = AA2 * SI::Mp; // in Kg

        if (species2.isSet("charge") and (get<BoutReal>(species2["charge"]) != 0.0)) {
          //////////////////////////////
          // Both charged species

          if (!ion_ion)
            continue;

          const BoutReal Z2 = get<BoutReal>(species2["charge"]);
          const BoutReal charge2 = Z2 * SI::qe; // in Coulombs

          // Ion-ion collisions
          Field3D nu_12 = filledFrom(density1, [&](auto& i) {
            const BoutReal Tlim1 = floor(temperature1[i], 0.1);
            const BoutReal Tlim2 = floor(temperature2[i], 0.1);

            const BoutReal Nlim1 = floor(density1[i], 1e10);
            const BoutReal Nlim2 = floor(density2[i], 1e10);

            // Coulomb logarithm
            BoutReal coulomb_log =
                29.91
                - log((Z1 * Z2 * (AA1 + AA2)) / (AA1 * Tlim2 + AA2 * Tlim1)
                      * sqrt(Nlim1 * SQ(Z1) / Tlim1 + Nlim2 * SQ(Z2) / Tlim2));

            // Calculate v_a^2, v_b^2
            const BoutReal v1sq = 2 * Tlim1 * SI::qe / mass1;
            const BoutReal v2sq = 2 * Tlim2 * SI::qe / mass2;

            // Collision frequency
            const BoutReal nu = SQ(charge1 * charge2) * Nlim2 * floor(coulomb_log, 1.0)
                                * (1. + mass1 / mass2)
                                / (3 * pow(PI * (v1sq + v2sq), 1.5) * SQ(SI::e0 * mass1));
            ASSERT2(std::isfinite(nu));
            return nu;
          });

          // Update the species collision rates, momentum & energy exchange
          collide(species1, species2, nu_12 / Omega_ci);

        } else {
          // species1 charged, species2 neutral

          // Scattering of charged species 1
          // Neutral density
          Field3D Nn = get<Field3D>(species2["density"]);

          BoutReal a0 = 5e-19; // Cross-section [m^2]

          const Field3D nu_12 = filledFrom(density1, [&](auto& i) {
            // Relative velocity is sqrt( v1^2 + v2^2 )
            const BoutReal vrel =
                sqrt(temperature1[i] / (Tnorm * AA1) + temperature2[i] / (Tnorm * AA2));

            // Ion-neutral collision rate
            // Units: density [m^-3], a0 [m^2], rho_s0 [m]
            return vrel * density2[i] * a0 * rho_s0;
          });

          collide(species1, species2, nu_12);
        }
      }
    } else {
      // species1 neutral

      // Copy the iterator, so we don't iterate over the
      // lower half of the matrix, but start at the diagonal
      for (std::map<std::string, Options>::const_iterator kv2 = kv1;
           kv2 != std::end(children); ++kv2) {
        if (kv2->first == "e")
          continue; // Skip electrons

        Options& species2 = allspecies[kv2->first];

        // Note: Here species1 could be equal to species2

        // If temperature isn't set, assume zero
        const Field3D temperature2 =
            species2.isSet("temperature") ? get<Field3D>(species2["temperature"]) : 0.0;
        const BoutReal AA2 = get<BoutReal>(species2["AA"]);
        const Field3D density2 = get<Field3D>(species2["density"]) * Nnorm;

        if (species2.isSet("charge")) {
          // species1 neutral, species2 charged

          if (!ion_neutral)
            continue;

          // Scattering of charged species 2

          // This is from NRL. The cross-section can vary significantly
          BoutReal a0 = 5e-19; // Cross-section [m^2]

          const Field3D nu_12 = filledFrom(density1, [&](auto& i) {
            // Relative velocity is sqrt( v1^2 + v2^2 )
            const BoutReal vrel =
                sqrt(temperature1[i] / (Tnorm * AA1) + temperature2[i] / (Tnorm * AA2));

            // Ion-neutral collision rate
            // Units: density [m^-3], a0 [m^2], rho_s0 [m]
            return vrel * density2[i] * a0 * rho_s0;
          });

          collide(species1, species2, nu_12);

        } else {
          // Both species neutral

          if (!neutral_neutral)
            continue;

          // The cross section is given by π ( (d1 + d2)/2 )^2
          // where d is the kinetic diameter
          //
          // Typical values [m]
          //  H2  2.89e-10
          //  He  2.60e-10
          //  Ne  2.75e-10
          //
          BoutReal a0 = PI * SQ(2.8e-10); // Cross-section [m^2]

          const Field3D nu_12 = filledFrom(density1, [&](auto& i) {
            // Relative velocity is sqrt( v1^2 + v2^2 )
            const BoutReal vrel =
                sqrt(temperature1[i] / (Tnorm * AA1) + temperature2[i] / (Tnorm * AA2));

            // Ion-neutral collision rate
            // Units: density [m^-3], a0 [m^2], rho_s0 [m]
            return vrel * density2[i] * a0 * rho_s0;
          });

          collide(species1, species2, nu_12);
        }
      }
    }
  }
}

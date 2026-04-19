import numpy as np
from scipy import integrate


def get_source_hardn_ratios(source_row):
    if source_row["SpectrumType"].split()[0] == "PowerLaw":
        hardn_ratios = Hardn_ratio_calc.hardn_ratios(
            "PowerLaw",
            (
                source_row["PL_Flux_Density"],
                source_row["Pivot_Energy"],
                source_row["PL_Index"],
            ),
        )

    elif source_row["SpectrumType"].split()[0] == "LogParabola":
        hardn_ratios = Hardn_ratio_calc.hardn_ratios(
            "LogParabola",
            (
                source_row["LP_Flux_Density"],
                source_row["Pivot_Energy"],
                source_row["LP_Index"],
                source_row["LP_beta"],
            ),
        )

    elif source_row["SpectrumType"].split()[0] == "PLSuperExpCutoff":
        hardn_ratios = Hardn_ratio_calc.hardn_ratios(
            "PLSuperExpCutoff",
            (
                source_row["PLEC_Flux_Density"],
                source_row["Pivot_Energy"],
                source_row["PLEC_Index"],
                source_row["PLEC_Expfactor"],
                source_row["PLEC_Exp_Index"],
            ),
        )

    return hardn_ratios


class Hardn_ratio_calc:  # bands: 1: 50-100 MeV; 2: 100-300 MeV; 3:300 MeV-1 GeV;4: 1-3 GeV; 5: 3-10 GeV; 6: 10-30 GeV; 7: 30-300 Gev
    # parameters must be a tuple object containing the function parameters

    def power_law(E, K, E_0, gamma):  # PL_Flux_Density ,Pivot_Energy,PL_Index
        return E * (K * ((E / E_0) ** -gamma))

    def log_parabola(
        E, K, E_0, alfa, beta
    ):  # LP_Flux_Density ,Pivot_Energy,LP_Index,LP_beta
        return E * (K * ((E / E_0) ** (-alfa - beta * np.log(E / E_0))))

    def pl_super_exp_cutoff(
        E, K, E_0, gamma, a, b
    ):  # PLEC_Flux_Density,Pivot_Energy,PLEC_Index,PLEC_Expfactor,PLEC_Exp_Index,
        return E * (K * ((E / E_0) ** -gamma) * np.exp(a * (E_0**b - E**b)))

    def hardn_ratios(model, parameters):
        Energies = []
        hardn_ratios = []

        if model == "PowerLaw":
            Energies.append(
                integrate.quad(Hardn_ratio_calc.power_law, 100, 300, args=parameters)[0]
            )
            Energies.append(
                integrate.quad(Hardn_ratio_calc.power_law, 300, 1000, args=parameters)[
                    0
                ]
            )
            Energies.append(
                integrate.quad(Hardn_ratio_calc.power_law, 1000, 3000, args=parameters)[
                    0
                ]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.power_law, 3000, 10000, args=parameters
                )[0]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.power_law, 10000, 100000, args=parameters
                )[0]
            )

        elif model == "LogParabola":
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.log_parabola, 100, 300, args=parameters
                )[0]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.log_parabola, 300, 1000, args=parameters
                )[0]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.log_parabola, 1000, 3000, args=parameters
                )[0]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.log_parabola, 3000, 10000, args=parameters
                )[0]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.log_parabola, 10000, 100000, args=parameters
                )[0]
            )

        elif model == "PLSuperExpCutoff":
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.pl_super_exp_cutoff, 100, 300, args=parameters
                )[0]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.pl_super_exp_cutoff, 300, 1000, args=parameters
                )[0]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.pl_super_exp_cutoff, 1000, 3000, args=parameters
                )[0]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.pl_super_exp_cutoff, 3000, 10000, args=parameters
                )[0]
            )
            Energies.append(
                integrate.quad(
                    Hardn_ratio_calc.pl_super_exp_cutoff, 10000, 100000, args=parameters
                )[0]
            )

        # Now I calculate hardness ratios
        for i in range(0, len(Energies) - 1):
            hardn_ratios.append(
                (Energies[i + 1] - Energies[i]) / (Energies[i + 1] + Energies[i])
            )

        return hardn_ratios

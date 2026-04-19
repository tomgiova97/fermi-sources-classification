SOURCES_CATEGORIES = ["PULSAR", "AGN"]

# Labels used in the 4FGL file to indicate pulsar sources
PULS_LABELS = ["PSR", "psr"]

# Labels used in the 4FGL file to indicate Active-Galactic-Nuclei sources
AGN_LABELS = [
    "FSRQ",
    "fsrq",
    "BLL",
    "bll",
    "BCU",
    "bcu",
    "RDG",
    "rdg",
    "NLSY1",
    "nlsy1",
    "AGN",
    "agn",
    "ssrq",
    "sey",
]

# Source attributes taken in consideration for the analysis (excluding the hrdness ratios)
SOURCES_ORIGINAL_ATTRIBUTES = [
    "PL_Index",
    "LP_Index",
    "PLEC_Index",
    "Variability_Index",
    "PL_Flux_Density",
    "LP_Flux_Density",
    "PLEC_Flux_Density",
    "Unc_Energy_Flux100",
    "LP_SigCurv",
    "PLEC_SigCurv",
]

HARDN_RATIO_ATTRIBUTES = [
    "Hardn_Ratio_1",
    "Hardn_Ratio_2",
    "Hardn_Ratio_3",
    "Hardn_Ratio_4",
]

DR2_FILE_CORRUPTED_ROWS = [
    905,
    1117,
    1808,
    2204,
    2537,
    2581,
    2862,
    3625,
    4275,
    4571,
    4596,
    5173,
    5236,
    5674,
]

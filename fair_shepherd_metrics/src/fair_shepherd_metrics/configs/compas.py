"""COMPAS default experiment specification."""

CONFIG = {
    "name": "COMPAS",
    "source": "ProPublica COMPAS two-year recidivism cohort",
    "target": "two_year_recid",
    "positive_outcome": "re-arrest within two years",
    "sensitive_attributes": ("sex", "race_grouped"),
    "split": {"type": "random", "train": 0.5, "validation": 0.2, "test": 0.3},
    "derived_sensitive_attributes": {
        "race_grouped": "race with Asian and Native American mapped to Other"
    },
    "inclusion_filters": (
        "days_b_screening_arrest >= -30",
        "days_b_screening_arrest <= 30",
        "is_recid != -1",
        "c_charge_degree != 'O'",
        "score_text != 'N/A'",
    ),
    "numeric_features": (
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "days_b_screening_arrest",
        "c_days_from_compas",
        "screening_month",
        "screening_dayofweek",
    ),
    "categorical_features": ("age_cat", "c_charge_degree"),
    "derived_features": {
        "screening_month": "month(compas_screening_date)",
        "screening_dayofweek": "dayofweek(compas_screening_date)",
    },
    "excluded_features": (
        "c_jail_in",
        "c_jail_out",
        "compas_screening_date",
        "c_offense_date",
        "is_recid",
        "score_text",
    ),
    "normative_features": (
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "c_charge_degree",
    ),
    "proxy_features": (
        ("age", 1),
        ("age_cat", 1),
        ("screening_month", 1),
        ("screening_dayofweek", 1),
    ),
    "vertical_levels": (
        ("V1: L1_LowRisk", "priors_count == 0 and c_charge_degree == 'M'"),
        ("V2: L2_MidRisk", "priors_count > 0 and c_charge_degree == 'M'"),
        ("V3: L3_HighRisk", "c_charge_degree == 'F'"),
    ),
    "audit_groups": {
        "race": {
            "AA": "race_grouped == 'African-American'",
            "Cauc": "race_grouped == 'Caucasian'",
            "Hisp": "race_grouped == 'Hispanic'",
        },
        "sex": {
            "Male": "sex == 'Male'",
            "Female": "sex == 'Female'",
        },
        "race_sex": {
            "AA-M": "race_grouped == 'African-American' and sex == 'Male'",
            "Cauc-M": "race_grouped == 'Caucasian' and sex == 'Male'",
        },
    },
    "audit_pairs": (
        ("Race_AA_vs_Cauc", "race", "AA", "Cauc", 1),
        ("Race_AA_vs_Hisp", "race", "AA", "Hisp", 1),
        ("Sex_Female_vs_Male", "sex", "Female", "Male", 1),
    ),
    "training_proxy_groups": {
        "age_proxy": {
            "A": "age < 25",
            "B": "age > 45",
        },
        "month_proxy": {
            "A": "screening_month <= 6",
            "B": "screening_month > 6",
        },
    },
    "training_proxy_pairs": (
        ("P_Age", "age_proxy", "A", "B"),
        ("P_Month", "month_proxy", "A", "B"),
    ),
}

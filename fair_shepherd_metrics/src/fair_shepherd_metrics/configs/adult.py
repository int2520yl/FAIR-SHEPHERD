"""Adult default experiment specification."""

CONFIG = {
    "name": "Adult",
    "source": "UCI Adult Census Income",
    "target": "income-per-year",
    "positive_outcome": ">50K",
    "sensitive_attributes": ("sex", "race"),
    "split": {"type": "random", "train": 0.6, "validation": 0.2, "test": 0.2},
    "numeric_features": (
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ),
    "categorical_features": (
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "native-country",
    ),
    "dropped_features": ("fnlwgt", "relationship", "age"),
    "normative_features": (
        "education-num",
        "hours-per-week",
        "capital-gain",
        "capital-loss",
    ),
    "proxy_features": (
        ("marital-status", -1),
        ("occupation", -1),
        ("native-country", -1),
        ("workclass", -1),
    ),
    "vertical_levels": (
        ("V1: Ed-Low", "`education-num` <= 9"),
        ("V2: Ed-Mid", "`education-num` >= 10 and `education-num` <= 12"),
        ("V3: Ed-High", "`education-num` >= 13"),
    ),
    "audit_groups": {
        "race": {
            "White": "race == 'white'",
            "NonWhite": "race != 'white'",
        },
        "sex": {
            "Male": "sex == 'male'",
            "Female": "sex == 'female'",
        },
        "race_sex": {
            "WM": "race == 'white' and sex == 'male'",
            "WF": "race == 'white' and sex == 'female'",
            "NWM": "race != 'white' and sex == 'male'",
            "NWF": "race != 'white' and sex == 'female'",
        },
    },
    "audit_pairs": (
        ("Race_NonWhite_vs_White", "race", "NonWhite", "White", -1),
        ("Sex_Female_vs_Male", "sex", "Female", "Male", -1),
        ("Inter_NWF_vs_WM", "race_sex", "NWF", "WM", -1),
    ),
    "training_proxy_groups": {
        "marital_proxy": {
            "A": "`marital-status` == 'married-civ-spouse'",
            "B": "`marital-status` != 'married-civ-spouse'",
        },
        "occupation_proxy": {
            "A": "occupation in ['priv-house-serv', 'other-service']",
            "B": "occupation in ['exec-managerial', 'prof-specialty']",
        },
        "native_country_proxy": {
            "A": "`native-country` != 'united-states'",
            "B": "`native-country` == 'united-states'",
        },
    },
    "training_proxy_pairs": (
        ("P_Sex_Marital", "marital_proxy", "A", "B"),
        ("P_Race_Occ", "occupation_proxy", "A", "B"),
        ("P_Race_Native", "native_country_proxy", "A", "B"),
    ),
}

"""Credit Default experiment specification."""

CONFIG = {
    "name": "Credit Default",
    "source": "UCI Default of Credit Card Clients",
    "target": "default_payment_next_month",
    "positive_outcome": "default",
    "sensitive_attributes": ("sex_female",),
    "split": {"type": "random", "train": 0.7, "validation": 0.15, "test": 0.15},
    "derived_sensitive_attributes": {"sex_female": "1 if sex == 2 else 0"},
    "numeric_features": (
        "limit_bal",
        "age",
        "bill_amt1",
        "bill_amt2",
        "bill_amt3",
        "bill_amt4",
        "bill_amt5",
        "bill_amt6",
        "pay_amt1",
        "pay_amt2",
        "pay_amt3",
        "pay_amt4",
        "pay_amt5",
        "pay_amt6",
    ),
    "categorical_features": (
        "education",
        "marriage",
        "pay_0",
        "pay_2",
        "pay_3",
        "pay_4",
        "pay_5",
        "pay_6",
    ),
    "dropped_features": (
        "ID",
        "sex",
        "default.payment.next.month",
        "default payment next month",
        "default_payment_next_month",
    ),
    "normative_features": (
        "pay_0",
        "pay_2",
        "pay_3",
        "pay_4",
        "pay_5",
        "pay_6",
        "limit_bal",
        "bill_amt1",
        "pay_amt1",
        "pay_amt2",
        "pay_amt3",
    ),
    "proxy_features": (("age", 1), ("education", 1), ("marriage", 1)),
    "vertical_levels": (
        ("V1: On-Time", "pay_0 <= 0"),
        ("V2: Delayed", "pay_0 == 1 or pay_0 == 2"),
        ("V3: Defaulting", "pay_0 >= 3"),
    ),
    "audit_groups": {
        "sex": {
            "Male": "sex_female == 0",
            "Female": "sex_female == 1",
        }
    },
    "audit_pairs": (
        ("Sex_Female_vs_Male", "sex", "Female", "Male", 1),
    ),
    "training_proxy_groups": {
        "age_proxy": {
            "A": "age < 30",
            "B": "age > 50",
        },
        "marriage_proxy": {
            "A": "marriage == 1",
            "B": "marriage == 2",
        },
        "education_proxy": {
            "A": "education >= 3",
            "B": "education <= 2",
        },
    },
    "training_proxy_pairs": (
        ("P_Age", "age_proxy", "A", "B"),
        ("P_Marriage", "marriage_proxy", "A", "B"),
        ("P_Education", "education_proxy", "A", "B"),
    ),
}

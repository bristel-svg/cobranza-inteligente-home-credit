from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main(n_clients: int = 1800, random_state: int = 42) -> None:
    rng = np.random.default_rng(random_state)

    sk_curr = np.arange(100000, 100000 + n_clients)
    application = pd.DataFrame(
        {
            "SK_ID_CURR": sk_curr,
            "TARGET": rng.binomial(1, 0.08, size=n_clients),
            "CODE_GENDER": rng.choice(["M", "F"], size=n_clients),
            "AMT_INCOME_TOTAL": rng.lognormal(mean=11.2, sigma=0.5, size=n_clients).round(0),
            "AMT_CREDIT": rng.lognormal(mean=12.4, sigma=0.5, size=n_clients).round(0),
            "AMT_ANNUITY": rng.lognormal(mean=9.8, sigma=0.4, size=n_clients).round(0),
            "AMT_GOODS_PRICE": rng.lognormal(mean=12.2, sigma=0.5, size=n_clients).round(0),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner"], size=n_clients),
            "NAME_EDUCATION_TYPE": rng.choice(["Secondary", "Higher education", "Incomplete higher"], size=n_clients),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single", "Civil marriage"], size=n_clients),
            "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents", "Rented apartment"], size=n_clients),
            "REGION_RATING_CLIENT": rng.integers(1, 4, size=n_clients),
            "REGION_RATING_CLIENT_W_CITY": rng.integers(1, 4, size=n_clients),
            "DAYS_BIRTH": -rng.integers(20 * 365, 70 * 365, size=n_clients),
            "DAYS_EMPLOYED": -rng.integers(0, 30 * 365, size=n_clients),
            "DAYS_REGISTRATION": -rng.integers(0, 20 * 365, size=n_clients),
            "DAYS_ID_PUBLISH": -rng.integers(0, 15 * 365, size=n_clients),
            "OWN_CAR_AGE": rng.choice([np.nan, 1, 3, 5, 8, 12], size=n_clients),
            "EXT_SOURCE_1": rng.uniform(0, 1, size=n_clients),
            "EXT_SOURCE_2": rng.uniform(0, 1, size=n_clients),
            "EXT_SOURCE_3": rng.uniform(0, 1, size=n_clients),
            "CNT_CHILDREN": rng.integers(0, 4, size=n_clients),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, size=n_clients),
            "ORGANIZATION_TYPE": rng.choice(["Business Entity", "Self-employed", "School", "Government"], size=n_clients),
        }
    )

    prev_rows = []
    inst_rows = []
    prev_id = 200000
    for curr in sk_curr:
        risk = rng.beta(2, 8)
        n_loans = rng.integers(1, 4)
        for _ in range(n_loans):
            prev_id += 1
            cnt_payment = int(rng.integers(6, 24))
            credit_amt = float(rng.lognormal(mean=11.5, sigma=0.6))
            annuity = credit_amt / max(cnt_payment, 1) * rng.uniform(0.9, 1.2)
            prev_rows.append(
                {
                    "SK_ID_PREV": prev_id,
                    "SK_ID_CURR": curr,
                    "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Consumer loans"], p=[0.45, 0.55]),
                    "AMT_ANNUITY": round(annuity, 2),
                    "AMT_APPLICATION": round(credit_amt * rng.uniform(0.8, 1.2), 2),
                    "AMT_CREDIT": round(credit_amt, 2),
                    "AMT_DOWN_PAYMENT": round(max(0, credit_amt * rng.uniform(0, 0.2)), 2),
                    "AMT_GOODS_PRICE": round(credit_amt * rng.uniform(0.85, 1.05), 2),
                    "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused"], p=[0.9, 0.1]),
                    "DAYS_DECISION": -int(rng.integers(200, 1600)),
                    "CNT_PAYMENT": cnt_payment,
                    "NAME_PORTFOLIO": rng.choice(["Cash", "POS", "Cards"], p=[0.5, 0.4, 0.1]),
                    "NAME_PRODUCT_TYPE": rng.choice(["x-sell", "walk-in"], p=[0.4, 0.6]),
                    "CHANNEL_TYPE": rng.choice(["Credit and cash offices", "Country-wide", "Stone"], p=[0.5, 0.3, 0.2]),
                    "SELLERPLACE_AREA": int(rng.integers(-1, 5000)),
                    "NAME_YIELD_GROUP": rng.choice(["low_normal", "middle", "high"], p=[0.4, 0.4, 0.2]),
                    "PRODUCT_COMBINATION": rng.choice(["Cash", "POS household", "POS mobile"], p=[0.5, 0.3, 0.2]),
                }
            )
            start_day = -int(rng.integers(800, 1200))
            for k in range(1, cnt_payment + 1):
                due = start_day + 30 * k
                # Riesgo individual aumenta probabilidad y severidad del atraso.
                late_prob = min(0.08 + 0.65 * risk + 0.01 * k, 0.85)
                is_late = rng.random() < late_prob
                delay = int(rng.gamma(shape=2.0, scale=10.0)) if is_late else -int(rng.integers(0, 8))
                pay_day = due + delay
                partial_prob = min(0.04 + 0.45 * risk + 0.005 * k, 0.7)
                is_partial = rng.random() < partial_prob
                instalment = max(1000, annuity * rng.uniform(0.8, 1.2))
                payment_ratio = rng.uniform(0.15, 0.9) if is_partial else rng.uniform(0.98, 1.08)
                payment = instalment * payment_ratio
                inst_rows.append(
                    {
                        "SK_ID_PREV": prev_id,
                        "SK_ID_CURR": curr,
                        "NUM_INSTALMENT_VERSION": 1,
                        "NUM_INSTALMENT_NUMBER": k,
                        "DAYS_INSTALMENT": due,
                        "DAYS_ENTRY_PAYMENT": pay_day,
                        "AMT_INSTALMENT": round(instalment, 2),
                        "AMT_PAYMENT": round(payment, 2),
                    }
                )

    previous = pd.DataFrame(prev_rows)
    installments = pd.DataFrame(inst_rows)

    application.to_csv(RAW_DIR / "application_train.csv", index=False)
    previous.to_csv(RAW_DIR / "previous_application.csv", index=False)
    installments.to_csv(RAW_DIR / "installments_payments.csv", index=False)

    print("Datos sintéticos creados en data/raw/")
    print(f"application_train.csv: {len(application):,} filas")
    print(f"previous_application.csv: {len(previous):,} filas")
    print(f"installments_payments.csv: {len(installments):,} filas")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
    main(n_clients=n)

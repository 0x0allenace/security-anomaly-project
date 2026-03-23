# This script simulates various attack scenarios in an enterprise environment by injecting synthetic attack events into a dataset of normal user activity logs. The simulated attacks include credential stuffing, privilege misuse, abnormal session durations, and lateral movement. Each attack type is designed to mimic realistic patterns of malicious behavior while maintaining the overall structure and distribution of the original log data. The resulting dataset can be used for training and evaluating anomaly detection models in cybersecurity contexts.

from __future__ import annotations

from datetime import timedelta
import random
from typing import List

import pandas as pd

# Note: This script assumes that the raw logs have already been generated and saved as a CSV file using the generate_logs.py script. The attack simulation functions will read this raw log data, inject synthetic attack events, and then save the augmented dataset for use in training and testing anomaly detection models.

RAW_LOG_PATH = "data/raw/raw_logs.csv"
ATTACK_LOG_PATH = "data/raw/logs_with_attacks.csv"


def load_logs(path: str = RAW_LOG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def save_logs(df: pd.DataFrame, path: str = ATTACK_LOG_PATH) -> None:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(path, index=False)


def mark_attack_rows(
    attack_rows: List[dict],
    attack_type: str,
) -> List[dict]:
    for row in attack_rows:
        row["is_attack"] = 1
        row["attack_type"] = attack_type
    return attack_rows


# simulate credential stuffing attacks by generating multiple failed login attempts from random IP addresses targeting various user accounts, with a focus on off-hours activity to mimic real-world attack patterns

def simulate_credential_stuffing(
    df: pd.DataFrame,
    num_events: int = 120,
) -> pd.DataFrame:
    attack_rows: List[dict] = []

    users = df["user_id"].dropna().unique().tolist()
    ip_pool = [f"185.220.101.{i}" for i in range(10, 80)]

    for _ in range(num_events):
        base_row = df.sample(1).iloc[0].to_dict()
        target_user = random.choice(users)

        base_time = pd.Timestamp(base_row["timestamp"])
        attack_time = base_time.replace(
            hour=random.choice([0, 1, 2, 3, 4, 23]),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )

        new_row = base_row.copy()
        new_row["timestamp"] = attack_time
        new_row["user_id"] = target_user
        new_row["username"] = base_row["username"]
        new_row["action"] = "login"
        new_row["login_success"] = False
        new_row["failed_attempts"] = random.randint(5, 15)
        new_row["session_duration_min"] = random.randint(0, 2)
        new_row["source_ip"] = random.choice(ip_pool)

        attack_rows.append(new_row)

    attack_rows = mark_attack_rows(attack_rows, "credential_stuffing")
    return pd.concat([df, pd.DataFrame(attack_rows)], ignore_index=True)

# simulate privilege misuse by generating events where low-privilege users access sensitive resources during unusual hours, with longer session durations to mimic real-world attack patterns of insiders or compromised accounts attempting to access data they shouldn't have access to.


def simulate_privilege_misuse(
    df: pd.DataFrame,
    num_events: int = 80,
) -> pd.DataFrame:
    attack_rows: List[dict] = []

    low_priv_users = df[df["privilege_level"] == "low"]
    if low_priv_users.empty:
        return df

    sensitive_resources = ["admin_console", "security_dashboard", "finance_db"]

    for _ in range(num_events):
        base_row = low_priv_users.sample(1).iloc[0].to_dict()
        base_time = pd.Timestamp(base_row["timestamp"])

        new_row = base_row.copy()
        new_row["timestamp"] = base_time + \
            timedelta(minutes=random.randint(1, 30))
        new_row["action"] = "access"
        new_row["resource_accessed"] = random.choice(sensitive_resources)
        new_row["login_success"] = True
        new_row["failed_attempts"] = 0
        new_row["session_duration_min"] = random.randint(20, 180)

        attack_rows.append(new_row)

    attack_rows = mark_attack_rows(attack_rows, "privilege_misuse")

    return pd.concat([df, pd.DataFrame(attack_rows)], ignore_index=True)


# simulate abnormal session durations by creating events where users have unusually long or short session durations compared to their normal behavior, which can indicate potential account compromise or misuse. This is done by sampling successful login events and modifying their session duration to be either significantly longer or shorter than typical values.

def simulate_abnormal_session_duration(
    df: pd.DataFrame,
    num_events: int = 70,
) -> pd.DataFrame:
    attack_rows: List[dict] = []

    success_rows = df[df["login_success"] == True]
    if success_rows.empty:
        return df

    for _ in range(num_events):
        base_row = success_rows.sample(1).iloc[0].to_dict()
        base_time = pd.Timestamp(base_row["timestamp"])

        new_row = base_row.copy()
        new_row["timestamp"] = base_time + \
            timedelta(minutes=random.randint(1, 120))
        new_row["action"] = random.choice(["access", "logout"])
        new_row["session_duration_min"] = random.choice(
            [random.randint(300, 720), random.randint(1, 3)]
        )
        new_row["login_success"] = True
        new_row["failed_attempts"] = 0

        attack_rows.append(new_row)

    attack_rows = mark_attack_rows(attack_rows, "abnormal_session")
    return pd.concat([df, pd.DataFrame(attack_rows)], ignore_index=True)

# simulate lateral movement by generating sequences of access events from one user to another, mimicking the behavior of an attacker moving through the network after compromising an initial account. This involves creating a series of events where the attacker accesses multiple resources across different user accounts, with timestamps that suggest a progression of activity over time. The source IP addresses are also modified to reflect internal network movement.


def simulate_lateral_movement(
    df: pd.DataFrame,
    num_sequences: int = 25,
    hops_per_sequence: int = 4,
) -> pd.DataFrame:
    attack_rows: List[dict] = []

    users = df["user_id"].dropna().unique().tolist()
    resources = ["admin_console", "security_dashboard",
                 "finance_db", "file_share", "git_server"]

    for _ in range(num_sequences):
        base_row = df.sample(1).iloc[0].to_dict()
        moving_user = random.choice(users)
        start_time = pd.Timestamp(base_row["timestamp"]).replace(
            hour=random.choice([22, 23, 0, 1, 2, 3]),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )

        unique_resources = random.sample(
            resources, k=min(hops_per_sequence, len(resources)))

        for i, resource in enumerate(unique_resources):
            new_row = base_row.copy()
            new_row["timestamp"] = start_time + \
                timedelta(minutes=i * random.randint(1, 3))
            new_row["user_id"] = moving_user
            new_row["action"] = "access"
            new_row["resource_accessed"] = resource
            new_row["source_ip"] = f"172.16.{random.randint(1, 10)}.{random.randint(1, 254)}"
            new_row["login_success"] = True
            new_row["failed_attempts"] = 0
            new_row["session_duration_min"] = random.randint(5, 25)

            attack_rows.append(new_row)

    attack_rows = mark_attack_rows(attack_rows, "lateral_movement")
    return pd.concat([df, pd.DataFrame(attack_rows)], ignore_index=True)

# utility function to print a summary of the attack types and counts in the dataset, as well as the total number of rows, attack rows, and normal rows. This helps to quickly understand the distribution of attack events in the augmented dataset.


def print_attack_summary(df: pd.DataFrame) -> None:
    print("\nAttack Summary:")
    print(df["attack_type"].value_counts(dropna=False))
    print("\nTotal rows:", len(df))
    print("Attack rows:", int(df["is_attack"].sum()))
    print("Normal rows:", int((df["is_attack"] == 0).sum()))


def main() -> None:
    random.seed(42)

    df = load_logs()

    df = simulate_credential_stuffing(df, num_events=120)
    df = simulate_privilege_misuse(df, num_events=80)
    df = simulate_abnormal_session_duration(df, num_events=70)
    df = simulate_lateral_movement(df, num_sequences=25, hops_per_sequence=4)

    save_logs(df)
    print(f"Attack-injected logs saved to: {ATTACK_LOG_PATH}")
    print_attack_summary(df)
    print("\nSample attack rows:")
    print(df[df["is_attack"] == 1].head(10))


if __name__ == "__main__":
    main()

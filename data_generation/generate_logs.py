# generate synthetic normal login and access logs for an enterprise environment, simulating realistic user behavior based on profiles and working hours

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from typing import List, Dict

import pandas as pd
from faker import Faker


# generate user profiles with realistic attributes and working hours to create a foundation for simulating normal login behavior in an enterprise environment

fake = Faker()


@dataclass
class UserProfile:
    user_id: str
    username: str
    department: str
    role: str
    privilege_level: str
    normal_start_hour: int
    normal_end_hour: int
    home_ip_prefix: str
    location: str
    device_type: str

# realistic departments, roles, privilege levels, device types, and locations for user profiles


DEPARTMENTS = ["IT", "HR", "Finance", "Operations", "Security", "Engineering"]
ROLES = ["Analyst", "Engineer", "Manager", "Administrator", "Staff"]
PRIVILEGE_LEVELS = ["low", "medium", "high"]
DEVICE_TYPES = ["Windows-Laptop", "MacBook", "Linux-Workstation", "Mobile"]
LOCATIONS = ["New York", "Lagos", "London", "Toronto", "Berlin"]
RESOURCES = [
    "email_portal",
    "vpn_gateway",
    "finance_db",
    "hr_portal",
    "git_server",
    "file_share",
    "admin_console",
    "security_dashboard",
]

# generate random IP addresses based on user home IP prefix to simulate realistic login sources


def random_ip(prefix: str) -> str:
    return f"{prefix}.{random.randint(1, 254)}"

# generate synthetic logs based on user profiles, simulating normal login and access patterns with realistic timestamps, success rates, session durations, and resource access to create a dataset for anomaly detection model training and evaluation


def build_user_profiles(num_users: int = 50) -> List[UserProfile]:
    profiles: List[UserProfile] = []

    for i in range(1, num_users + 1):
        role = random.choice(ROLES)
        privilege = random.choices(
            PRIVILEGE_LEVELS,
            weights=[0.6, 0.3, 0.1],
            k=1,
        )[0]

        if role == "Administrator":
            privilege = "high"

        normal_start = random.randint(7, 10)
        normal_end = normal_start + random.randint(7, 10)

        profiles.append(
            UserProfile(
                user_id=f"U{i:03d}",
                username=fake.user_name(),
                department=random.choice(DEPARTMENTS),
                role=role,
                privilege_level=privilege,
                normal_start_hour=normal_start,
                normal_end_hour=min(normal_end, 23),
                home_ip_prefix=f"10.{random.randint(1, 20)}.{random.randint(1, 50)}",
                location=random.choice(LOCATIONS),
                device_type=random.choice(DEVICE_TYPES),
            )
        )

    return profiles

# choose resources based on user profiles to simulate realistic access patterns (e.g., finance users access finance_db more often, HR users access hr_portal more often, high privilege users access admin_console and security_dashboard more often)


def choose_resource(profile: UserProfile) -> str:
    if profile.department == "Finance":
        return random.choices(
            RESOURCES,
            weights=[2, 2, 6, 1, 2, 2, 1, 1],
            k=1,
        )[0]
    if profile.department == "HR":
        return random.choices(
            RESOURCES,
            weights=[2, 2, 1, 6, 2, 2, 1, 1],
            k=1,
        )[0]
    if profile.privilege_level == "high":
        return random.choices(
            RESOURCES,
            weights=[2, 2, 2, 2, 2, 2, 4, 3],
            k=1,
        )[0]
    return random.choice(RESOURCES[:-1])

# simulate enterprise login behavior based on user working hours


def generate_normal_logs(
    profiles: List[UserProfile],
    start_date: datetime,
    num_days: int = 14,
    min_events_per_user_per_day: int = 3,
    max_events_per_user_per_day: int = 8,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for day in range(num_days):
        current_day = start_date + timedelta(days=day)

        for profile in profiles:
            event_count = random.randint(
                min_events_per_user_per_day,
                max_events_per_user_per_day,
            )

            for _ in range(event_count):
                hour = random.randint(
                    profile.normal_start_hour,
                    profile.normal_end_hour,
                )
                minute = random.randint(0, 59)
                second = random.randint(0, 59)

                timestamp = current_day.replace(
                    hour=hour,
                    minute=minute,
                    second=second,
                )

                success = random.choices(
                    [True, False], weights=[0.93, 0.07], k=1)[0]
                failed_attempts = 0 if success else random.randint(1, 2)

                session_duration = (
                    random.randint(
                        10, 180) if success else random.randint(0, 5)
                )

                rows.append(
                    {
                        "timestamp": timestamp,
                        "user_id": profile.user_id,
                        "username": profile.username,
                        "department": profile.department,
                        "role": profile.role,
                        "privilege_level": profile.privilege_level,
                        "source_ip": random_ip(profile.home_ip_prefix),
                        "location": profile.location,
                        "device_type": profile.device_type,
                        "resource_accessed": choose_resource(profile),
                        "action": random.choice(["login", "access", "logout"]),
                        "login_success": success,
                        "failed_attempts": failed_attempts,
                        "session_duration_min": session_duration,
                        "is_attack": 0,
                        "attack_type": "normal",
                    }
                )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df

# save generated logs to CSV for later use in training and testing anomaly detection models


def save_raw_logs(df: pd.DataFrame, output_path: str = "data/raw/raw_logs.csv") -> None:
    df.to_csv(output_path, index=False)

# main function to generate and save synthetic normal logs, with fixed random seed for reproducibility and summary printout of generated data


def main() -> None:
    random.seed(42)
    Faker.seed(42)

    profiles = build_user_profiles(num_users=60)
    start_date = datetime(2026, 1, 1, 0, 0, 0)

    logs_df = generate_normal_logs(
        profiles=profiles,
        start_date=start_date,
        num_days=14,
        min_events_per_user_per_day=4,
        max_events_per_user_per_day=10,
    )

    save_raw_logs(logs_df)

# print summary of generated logs for verification

    print("Synthetic normal logs generated successfully.")
    print(f"Rows generated: {len(logs_df)}")
    print("Saved to: data/raw/raw_logs.csv")
    print(logs_df.head())


if __name__ == "__main__":
    main()

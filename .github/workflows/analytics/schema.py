from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class BaseEvent:
    event_type: str  = field(init=False)

    def to_dict(self):
        return asdict(self)

@dataclass
class CronEvent(BaseEvent):
    num_discord_members_total: int
    num_github_stars_total: int

    # count for each day
    num_python_downloads: int
    num_discord_members: int
    num_github_stars: int
    num_github_repo_views: int
    num_documentation_views: int = 0

    def __post_init__(self):
        self.event_type = "cron"

    def to_dict(self):
        return asdict(self)

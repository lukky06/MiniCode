"""Repository-scoped Memory V3 for MiniCode."""

from .consolidation import Phase2Consolidator, Phase2Result
from .events import RepositoryMemoryEventStore
from .extraction import Phase1BatchResult, Phase1Extractor, run_pending_phase1
from .migration import MemoryMigrationResult, migrate_v2_topics
from .recall import MemorySnapshotReader
from .repository_id import (
    RepositoryIdentity,
    RepositoryIdentityUnavailable,
    resolve_repository_identity,
)
from .snapshot import MEMORY_SNAPSHOT_METADATA, MemorySnapshot, MemorySnapshotStore
from .store import (
    RepositoryMemorySnapshotSource,
    RepositoryMemoryStore,
)
from .types import MemoryPipelineState, Stage1Record

__all__ = [
    "MEMORY_SNAPSHOT_METADATA",
    "MemoryMigrationResult",
    "MemoryPipelineState",
    "MemorySnapshot",
    "MemorySnapshotReader",
    "MemorySnapshotStore",
    "Phase1BatchResult",
    "Phase1Extractor",
    "Phase2Consolidator",
    "Phase2Result",
    "RepositoryIdentity",
    "RepositoryIdentityUnavailable",
    "RepositoryMemoryEventStore",
    "RepositoryMemorySnapshotSource",
    "RepositoryMemoryStore",
    "Stage1Record",
    "migrate_v2_topics",
    "resolve_repository_identity",
    "run_pending_phase1",
]

import tempfile
import unittest
from pathlib import Path

from s_brain.identity_memory_bridge import identity_from_memory
from s_brain.memory_store import MemoryStore


class IdentityMemoryBridgeTests(unittest.TestCase):
    def test_opt_in_and_scope_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore(Path(directory) / "memory.sqlite3") as store:
                store.remember(
                    scope="S", content="Example Person", source="explicit-user-statement",
                    consent="explicit", tags=("identity", "identity_name"),
                )
                store.remember(
                    scope="OTHER", content="Other Person", source="other-user",
                    consent="explicit", tags=("identity", "identity_name"),
                )
                self.assertNotIn(
                    "Example Person",
                    identity_from_memory(store, "Who am I?", scope="S").text,
                )
                answer = identity_from_memory(
                    store, "Who am I?", scope="S", enabled=True,
                )
                self.assertIn("Example Person", answer.text)
                self.assertNotIn("Other Person", answer.text)
                self.assertEqual(answer.sources, ("explicit-user-statement",))

    def test_unstructured_notes_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore(Path(directory) / "memory.sqlite3") as store:
                store.remember(
                    scope="S", content="Unverified Name", source="note",
                    consent="explicit", tags=("notes",),
                )
                answer = identity_from_memory(
                    store, "Who am I?", scope="S", enabled=True,
                )
                self.assertNotIn("Unverified Name", answer.text)


if __name__ == "__main__":
    unittest.main()

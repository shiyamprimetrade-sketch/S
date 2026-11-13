import unittest

from s_brain.grounded_identity import Fact, answer_with_context, detect_intent


class GroundedIdentityTests(unittest.TestCase):
    def test_identity_intent(self):
        self.assertEqual(detect_intent(" Who   am I? "), "personal_identity")

    def test_nonidentity_intent(self):
        self.assertEqual(detect_intent("write Python"), "general")

    def test_no_authorized_facts(self):
        result = answer_with_context("Who am I?", [])
        self.assertIn("don't have authorized", result.text)
        self.assertEqual(result.sources, ())

    def test_private_facts_not_leaked(self):
        result = answer_with_context(
            "Who am I?", [Fact("name", "Secret Name", "private-note")]
        )
        self.assertNotIn("Secret Name", result.text)

    def test_authorized_provenance(self):
        result = answer_with_context(
            "Who am I?",
            [
                Fact("name", "Example Person", "user-profile", True),
                Fact("projects", "Project Alpha", "user-message", True),
                Fact("unknown", "ignored", "user-message", True),
            ],
        )
        self.assertIn("Example Person", result.text)
        self.assertIn("Project Alpha", result.text)
        self.assertNotIn("ignored", result.text)
        self.assertEqual(result.sources, ("user-profile", "user-message"))

    def test_conflict_preserves_first(self):
        result = answer_with_context(
            "Who am I?",
            [Fact("name", "First", "a", True), Fact("name", "Second", "b", True)],
        )
        self.assertIn("First", result.text)
        self.assertNotIn("Second", result.text)


if __name__ == "__main__":
    unittest.main()

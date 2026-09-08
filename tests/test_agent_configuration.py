"""Contract checks for the shared Codex and Claude agent configuration."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("bddcv-worker", "bddcv-explorer", "bddcv-validator", "bddcv-reviewer")
EXPECTED_CODEX = {
    "bddcv-worker": ("gpt-5.6-luna", "max"),
    "bddcv-explorer": ("gpt-5.6-luna", "xhigh"),
    "bddcv-validator": ("gpt-5.6-terra", "max"),
    "bddcv-reviewer": ("gpt-5.6-sol", "high"),
}
EXPECTED_CLAUDE = {
    "bddcv-worker": ("sonnet", "high"),
    "bddcv-explorer": ("haiku", "high"),
    "bddcv-validator": ("sonnet", "high"),
    "bddcv-reviewer": ("opus", "high"),
}
SKILLS = ("bddcv-orchestration", "bddcv-experiment-validation")


def load_codex_agent(role: str) -> dict:
    path = ROOT / ".codex" / "agents" / f"{role}.toml"
    with path.open("rb") as file:
        return tomllib.load(file)


def load_claude_agent(role: str) -> tuple[dict[str, str], str]:
    path = ROOT / ".claude" / "agents" / f"{role}.md"
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    if match is None:
        raise AssertionError(f"Missing YAML frontmatter: {path}")

    frontmatter: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.startswith((" ", "-")):
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip('"\'')
    return frontmatter, match.group(2)


class AgentConfigurationTests(unittest.TestCase):
    def test_codex_agents_expose_expected_roles_and_models(self) -> None:
        for role, (model, effort) in EXPECTED_CODEX.items():
            with self.subTest(role=role):
                agent = load_codex_agent(role)
                self.assertEqual(agent["name"], role)
                self.assertTrue(agent["description"])
                self.assertTrue(agent["developer_instructions"])
                self.assertEqual(agent["model"], model)
                self.assertEqual(agent["model_reasoning_effort"], effort)

    def test_codex_read_only_roles_are_sandboxed(self) -> None:
        for role in ("bddcv-explorer", "bddcv-reviewer"):
            with self.subTest(role=role):
                self.assertEqual(load_codex_agent(role)["sandbox_mode"], "read-only")

    def test_claude_agents_expose_expected_roles_and_models(self) -> None:
        for role, (model, effort) in EXPECTED_CLAUDE.items():
            with self.subTest(role=role):
                frontmatter, body = load_claude_agent(role)
                self.assertEqual(frontmatter["name"], role)
                self.assertEqual(frontmatter["model"], model)
                self.assertEqual(frontmatter["effort"], effort)
                self.assertTrue(frontmatter["description"])
                self.assertTrue(body.strip())

    def test_claude_read_only_roles_exclude_write_tools(self) -> None:
        for role in ("bddcv-explorer", "bddcv-reviewer"):
            with self.subTest(role=role):
                frontmatter, _ = load_claude_agent(role)
                tools = {tool.strip() for tool in frontmatter["tools"].strip("[]").split(",")}
                self.assertTrue(tools)
                self.assertTrue(tools.isdisjoint({"Edit", "Write", "NotebookEdit"}))

    def test_canonical_skills_and_claude_routes_exist(self) -> None:
        for skill in SKILLS:
            with self.subTest(skill=skill):
                canonical = ROOT / ".agents" / "skills" / skill / "SKILL.md"
                router = ROOT / ".claude" / "skills" / skill / "SKILL.md"
                canonical_text = canonical.read_text(encoding="utf-8")
                router_text = router.read_text(encoding="utf-8")
                self.assertRegex(canonical_text, rf"(?m)^name: {re.escape(skill)}$")
                self.assertRegex(canonical_text, r"(?m)^description: .+$")
                self.assertIn(f".agents/skills/{skill}/SKILL.md", router_text)

    def test_orchestration_skill_requires_classification_and_confirmation(self) -> None:
        path = ROOT / ".agents" / "skills" / "bddcv-orchestration" / "SKILL.md"
        text = path.read_text(encoding="utf-8").lower()
        for term in ("easy", "difficult", "short", "long", "confirmation", "recommended", "model"):
            with self.subTest(term=term):
                self.assertIn(term, text)

    def test_context_window_is_not_overridden(self) -> None:
        paths = [ROOT / ".codex" / "config.toml"]
        paths.extend(ROOT / ".codex" / "agents" / f"{role}.toml" for role in ROLES)
        for path in paths:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("model_context_window", text)
                self.assertNotIn("model_auto_compact_token_limit", text)


if __name__ == "__main__":
    unittest.main()

"""The validation harness must not accidentally launch published sweeps."""

import json
import unittest
from unittest.mock import patch
from scripts.validate_result_tables import main, smoke_cell_sources


def notebook(*sources):
    return {'cells': [{'cell_type': 'code', 'source': [source]} for source in sources]}


class SmokeGuardTests(unittest.TestCase):
    def test_accepts_disabled_guard_without_executing_cells(self):
        source = 'RUN_PUBLISHED = False\nraise RuntimeError("must not execute")'
        self.assertEqual(smoke_cell_sources(notebook(source)), [(0, source)])

    def test_enabled_guard_is_rejected_before_any_cell_executes(self):
        saved = notebook('raise RuntimeError("first cell must not execute")', 'RUN_PUBLISHED = True')
        with patch('pathlib.Path.read_text', return_value=json.dumps(saved)), \
             patch('builtins.exec') as execute:
            with self.assertRaisesRegex(ValueError, 'No notebook cells were executed'):
                main()
        execute.assert_not_called()

    def test_missing_ambiguous_or_computed_guard_is_rejected(self):
        for sources in [('',), ('RUN_PUBLISHED = bool(0)',),
                        ('RUN_PUBLISHED = False', 'RUN_PUBLISHED = True')]:
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                smoke_cell_sources(notebook(*sources))

    def test_explicit_smoke_override_skips_published_code_without_editing_notebook(self):
        saved = notebook('RUN_PUBLISHED = True',
                         'if RUN_PUBLISHED:\n    raise RuntimeError("full sweep must not run")')
        before = json.dumps(saved)
        scope = {}
        for _, source in smoke_cell_sources(saved, force_smoke=True):
            exec(source, scope)
        self.assertIs(scope['RUN_PUBLISHED'], False)
        self.assertEqual(json.dumps(saved), before)


if __name__ == '__main__':
    unittest.main()

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import orchestrate
import agent


class FablePreflightTests(unittest.TestCase):
    def test_live_preflight_checks_model_access_before_constructing_robot(self):
        with patch.object(orchestrate.os,'environ',{'ANTHROPIC_API_KEY':'synthetic-test-key'}), \
                patch('anthropic.Anthropic') as constructor, patch.object(agent,'Robot') as robot:
            execute = orchestrate.prepare_fable('fable')
            constructor.return_value.__enter__.return_value.models.retrieve.assert_called_once_with(agent.MODEL)
            self.assertTrue(callable(execute))
            robot.assert_not_called()

    def test_api_failure_is_sanitized_and_does_not_construct_robot(self):
        import anthropic
        with patch.object(orchestrate.os,'environ',{'ANTHROPIC_API_KEY':'synthetic-test-key'}), \
                patch('anthropic.Anthropic') as constructor, patch.object(agent,'Robot') as robot:
            client = constructor.return_value.__enter__.return_value
            client.models.retrieve.side_effect = anthropic.APIConnectionError(request=Mock())
            with self.assertRaisesRegex(RuntimeError,'Fable API/model preflight failed') as raised:
                orchestrate.prepare_fable('fable')
            self.assertNotIn('synthetic-test-key',str(raised.exception))
            robot.assert_not_called()

    def test_http_status_is_reported_without_exposing_api_error_contents(self):
        import anthropic
        for status in (401, 403, 404):
            with self.subTest(status=status), \
                    patch.object(orchestrate.os,'environ',{'ANTHROPIC_API_KEY':'synthetic-test-key'}), \
                    patch('anthropic.Anthropic') as constructor, patch.object(agent,'Robot') as robot:
                client = constructor.return_value.__enter__.return_value
                client.models.retrieve.side_effect = anthropic.APIStatusError(
                    'synthetic-test-key', response=Mock(status_code=status,headers={}), body=None)
                with self.assertRaises(RuntimeError) as raised:
                    orchestrate.prepare_fable('fable')
                self.assertIn(f'HTTP {status}',str(raised.exception))
                self.assertIn(agent.MODEL,str(raised.exception))
                self.assertNotIn('synthetic-test-key',str(raised.exception))
                robot.assert_not_called()

    def test_offline_preflight_never_creates_an_api_client(self):
        with patch('anthropic.Anthropic') as constructor, patch.object(agent,'Robot') as robot:
            self.assertTrue(callable(orchestrate.prepare_fable('sweep')))
            constructor.assert_not_called()
            robot.assert_not_called()

    def test_unknown_planner_cannot_silently_select_sweep(self):
        with self.assertRaises(ValueError):
            orchestrate.prepare_fable('unknown')


if __name__ == '__main__':
    unittest.main()

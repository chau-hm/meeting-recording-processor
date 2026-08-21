import unittest

from meeting_recording_processor.quality import inspect_text


class QualityTests(unittest.TestCase):
    def test_blank_output_is_a_hard_failure(self) -> None:
        report = inspect_text("  \n\t")
        self.assertTrue(report.hard_failure)
        self.assertEqual(report.reasons, ("empty_output",))

    def test_exclamation_mark_collapse_is_a_hard_failure(self) -> None:
        report = inspect_text("!!!!!!!!!!")
        self.assertTrue(report.hard_failure)
        self.assertEqual(report.punctuation_or_symbol_ratio, 1.0)

    def test_normal_cantonese_passes_objective_gate(self) -> None:
        report = inspect_text("今日我哋會講 database migration，同埋下一步安排。")
        self.assertFalse(report.hard_failure)

    def test_exactly_ninety_percent_does_not_cross_threshold(self) -> None:
        report = inspect_text("字!!!!!!!!!")
        self.assertEqual(report.punctuation_or_symbol_ratio, 0.9)
        self.assertFalse(report.hard_failure)

    def test_active_audio_with_near_zero_text_fails(self) -> None:
        report = inspect_text("吓", active_audio_seconds=12.0)
        self.assertTrue(report.hard_failure)
        self.assertIn("near_zero_text_with_active_audio", report.reasons)


if __name__ == "__main__":
    unittest.main()

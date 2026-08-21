import unittest

from meeting_recording_processor.cli import build_parser


class CliTests(unittest.TestCase):
    def test_extract_defaults_to_auto(self) -> None:
        args = build_parser().parse_args(["extract", "meeting.m4a"])
        self.assertEqual(args.command, "extract")
        self.assertEqual(args.asr, "auto")
        self.assertEqual(args.language, "Cantonese")

    def test_export_parser(self) -> None:
        args = build_parser().parse_args(["export", "meeting.transcript.json"])
        self.assertEqual(args.command, "export")


if __name__ == "__main__":
    unittest.main()

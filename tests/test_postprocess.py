import unittest

from meeting_recording_processor.postprocess import clean_text, postprocess_result, split_text_for_cues
from meeting_recording_processor.schemas import BackendResult, TranscriptSegment


class PostprocessTests(unittest.TestCase):
    def test_clean_text_normalizes_controls_and_spaces(self) -> None:
        self.assertEqual(clean_text("  第一行\x00  \r\n\r\n 第二行  "), "第一行\n第二行")

    def test_model_words_are_grouped_without_losing_english_space(self) -> None:
        result = BackendResult(
            language="Cantonese",
            backend="qwen3",
            model="test",
            text="Hello world. 我哋開始。",
            segments=(
                TranscriptSegment(0.0, 0.5, "Hello"),
                TranscriptSegment(0.5, 1.0, "world."),
                TranscriptSegment(1.0, 1.5, "我哋"),
                TranscriptSegment(1.5, 2.0, "開始。"),
            ),
        )
        text, segments, _ = postprocess_result(
            result, audio_duration=2.0, converter=lambda value: value
        )
        self.assertEqual(text, "Hello world. 我哋開始。")
        self.assertIn("Hello world.", segments[0].text)
        self.assertTrue(all(segment.timing_source == "model" for segment in segments))

    def test_chunk_timing_is_explicitly_estimated(self) -> None:
        result = BackendResult(
            language="yue",
            backend="sensevoice",
            model="test",
            text="第一句。第二句。",
            segments=(TranscriptSegment(0.0, 10.0, "第一句。第二句。", "chunk"),),
        )
        _, segments, warnings = postprocess_result(
            result, audio_duration=10.0, converter=lambda value: value
        )
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(item.timing_source == "estimated_from_chunk" for item in segments))
        self.assertTrue(warnings)

    def test_long_text_splits_into_caption_sized_units(self) -> None:
        cues = split_text_for_cues("甲" * 100, max_characters=42)
        self.assertEqual([len(cue) for cue in cues], [42, 42, 16])


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support_bot import FAQ, FAQRetriever, answer_question, load_faqs


class FAQChatbotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.faqs = load_faqs()
        cls.retriever = FAQRetriever(cls.faqs)

    def test_faq_sample_data_loads_with_unique_ids(self):
        self.assertEqual(len(self.faqs), 10)
        self.assertEqual(len({faq.id for faq in self.faqs}), 10)

    def test_retrieves_order_tracking_from_paraphrase(self):
        results = self.retriever.search("where is my package right now")
        self.assertEqual(results[0].faq.id, "FAQ-001")
        self.assertGreater(results[0].score, 0)

    def test_retrieves_password_reset_from_synonyms(self):
        answer = answer_question("I am locked out and need a reset link", self.retriever)
        self.assertIn("secure reset link", answer)
        self.assertIn("[FAQ-005]", answer)

    def test_retrieves_payment_method(self):
        answer = answer_question("Can I pay with UPI?", self.retriever)
        self.assertIn("UPI", answer)
        self.assertIn("[FAQ-006]", answer)

    def test_answer_includes_citation(self):
        answer = answer_question("How many days to get my money back?", self.retriever)
        self.assertIn("[FAQ-004]", answer)
        self.assertIn("5 to 7 business days", answer)

    def test_unknown_question_uses_fallback(self):
        answer = answer_question("Can you repair my garden sprinkler?", self.retriever)
        self.assertIn("couldn't find an answer", answer)
        self.assertNotIn("Source:", answer)

    def test_blank_question_prompts_for_input(self):
        self.assertIn("enter a question", answer_question("  ", self.retriever).lower())

    def test_top_k_validation(self):
        with self.assertRaises(ValueError):
            self.retriever.search("shipping", top_k=0)

    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            FAQRetriever([])

    def test_invalid_faq_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faqs.json"
            path.write_text(json.dumps([{"id": "FAQ-1"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_faqs(path)

    def test_duplicate_ids_rejected(self):
        first = FAQ("FAQ-1", "Orders", "Question?", "Answer.")
        with self.assertRaises(ValueError):
            FAQRetriever([first, first])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faqs.json"
            record = {"id": "FAQ-1", "category": "Orders", "question": "Q?", "answer": "A."}
            path.write_text(json.dumps([record, record]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_faqs(path)


if __name__ == "__main__":
    unittest.main()

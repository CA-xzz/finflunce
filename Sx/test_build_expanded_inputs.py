import unittest
import pandas as pd

from build_expanded_inputs import merge_indices, merge_market, select_events


class InputIntegrityTests(unittest.TestCase):
    def test_cannot_splice_adjusted_price_providers_for_same_stock(self):
        old = pd.DataFrame({"code": ["000002"], "date": ["2023-01-04"], "close": [10.]})
        added = pd.DataFrame({"code": ["000002"], "date": ["2023-01-03"], "close": [20.]})
        with self.assertRaisesRegex(ValueError, "already has original"):
            merge_market(old, added)

    def test_existing_index_history_cannot_be_replaced(self):
        old = pd.DataFrame({"index_code": ["sh.000001"], "date": ["2023-01-04"],
                            "index_close": [100.], "index_ret": [.01]})
        added = old.assign(index_close=100.01)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            merge_indices(old, added)

    def test_missing_interior_index_session_added_without_modifying_neighbors(self):
        old = pd.DataFrame({"index_code": ["sh.000001"] * 2,
            "date": ["2023-01-03", "2023-01-05"], "index_close": [100.1234, 102.3456],
            "index_ret": [.001234, .01123]})
        added = pd.DataFrame({"index_code": ["sh.000001"], "date": ["2023-01-04"],
            "index_close": [101.21], "index_ret": [.01085]})
        merged = merge_indices(old, added)
        self.assertEqual(merged.index_close.tolist(), [100.1234, 101.21, 102.3456])

    def test_collision_excluded_without_changing_another_same_code_recommendation(self):
        raw = pd.DataFrame({"recommended_stock_code": ["000001", "000001"],
            "recommended_stock_name": ["上证指数", "平安银行"], "recommend_date": ["2023/1/3"] * 2,
            "bigv_name": ["a"] * 2, "laiyuan": ["微博"] * 2})
        decisions = pd.DataFrame({"source_row": [1, 0], "original_code": ["000001", "000001"],
            "canonical_code": ["000001", "000001"], "decision": ["keep", "exclude"],
            "reason": ["verified_equity", "index_code_collision"], "original_name": ["平安银行", "上证指数"],
            "original_date": ["2023/1/3"] * 2, "original_influencer": ["a"] * 2,
            "original_platform": ["微博"] * 2})
        kept, excluded = select_events(raw, decisions)
        self.assertEqual(kept.original_source_row.tolist(), [1])
        self.assertEqual(kept.recommended_stock_name.tolist(), ["平安银行"])
        self.assertEqual(excluded.exclusion_reason.tolist(), ["index_code_collision"])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            select_events(raw, decisions.iloc[:1])
        swapped = decisions.assign(source_row=[0, 1])
        with self.assertRaisesRegex(ValueError, "recommended_stock_name"):
            select_events(raw, swapped)


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from aci_patch_agent.eval_tasks import EVAL_TASKS
from aci_patch_agent.tasks import TASKS
from agent_edit_dpo.data import candidates
from agent_edit_dpo.protocol import parse_response, text_messages


class ProtocolTest(unittest.TestCase):
    def test_valid_tool_and_malformed_text(self):
        message = parse_response('{"tool":"test","arguments":{}}', "one")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "test")
        for raw in ['{"tool":"shell","arguments":{}}', '```json\n{}\n```', 'null', '[1]', '{"tool":"test","arguments":[],"extra":1}']:
            self.assertNotIn("tool_calls", parse_response(raw, "one"))

    def test_round_trip_and_observation(self):
        raw = '{"tool":"edit","arguments":{"start":1,"end":2,"replacement":"x\\n"}}'
        message = parse_response(raw, "one")
        converted = text_messages([message, {"role":"tool","content":"observation"}])
        self.assertEqual(json.loads(converted[0]["content"]), json.loads(raw))
        self.assertEqual(converted[1], {"role":"user","content":"Tool result:\nobservation"})

    def test_preferences_disjoint_from_final_tasks(self):
        rows = list(candidates())
        self.assertEqual(len(rows), 50)
        self.assertEqual(sum(r[3]["split"] == "train" for r in rows), 40)
        self.assertTrue({t.id for t in TASKS}.isdisjoint(t.id for t in EVAL_TASKS))
        for task, chosen, rejected, row in rows:
            self.assertNotEqual(chosen, rejected)
            compile(chosen, "chosen", "exec")
            compile(rejected, "rejected", "exec")
            self.assertEqual(json.loads(row["chosen"][0]["content"])["arguments"]["end"], len(task.source.splitlines()))

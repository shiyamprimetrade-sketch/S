"""Selection diagnostics catch low-loss invalid output without running source."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from s_brain.coding_curriculum import records, read_plan, batches, case_inputs, oracle
from s_brain.generation_validation import contract_matches, generation_metrics, selection_rank, semantic_token_weights

ROOT=Path(__file__).resolve().parents[2]


class GenerationValidationTests(unittest.TestCase):
    def setUp(self):
        self.plan=read_plan(ROOT/'experiments/coding-curriculum-v3/plan.json')
        self.rows=records(self.plan)
        self.row=next(r for r in self.rows if r['split']=='validation')
    def test_valid_contract_outweighs_low_loss_and_valid_but_wrong_source(self):
        self.assertGreater(selection_rank({'contract_matched':1,'format_passed':1},1.0),
                           selection_rank({'contract_matched':0,'format_passed':1},0.001))
        self.assertGreater(selection_rank({'contract_matched':0,'format_passed':1},1.0),
                           selection_rank({'contract_matched':0,'format_passed':0},0.001))
    def test_ast_validation_accepts_whitespace_and_rejects_other_behavior(self):
        files=json.loads(self.row['completion'])['files']
        self.assertTrue(contract_matches(files,self.row))
        files['app.py']=files['app.py'].replace('    return','  return')
        self.assertTrue(contract_matches(files,self.row))
        files['app.py']='import os\nos._exit(0)\n'+files['app.py']
        self.assertFalse(contract_matches(files,self.row))
        files['app.py']='def solve(p)\n invalid'
        self.assertFalse(contract_matches(files,self.row))
    def test_free_generation_reports_format_and_contract_separately(self):
        wrong=json.loads(self.row['completion']);wrong['files']['app.py']='def solve(p): return {}\n'
        rows=[self.row]*3
        with patch('s_brain.generation_validation.greedy_json',side_effect=[self.row['completion'],json.dumps(wrong),'not json']):
            result=generation_metrics(None,rows,200)
        self.assertEqual(result['tasks'],3);self.assertEqual(result['format_passed'],2)
        self.assertEqual(result['contract_matched'],1);self.assertFalse(result['source_executed'])
    def test_semantic_weights_cover_operator_and_constant_only(self):
        row=self.row;x,y=batches([row],512)
        weights=semantic_token_weights([row],y,8)
        self.assertTrue((weights[y==-100]==0).all())
        self.assertEqual(int((weights==8).sum()),len(str(row['constant']))+1)
        emphasized=bytes(y[weights==8].tolist()).decode()
        self.assertEqual(emphasized,'+'+str(row['constant']))
        self.assertTrue((weights[(y!=-100)&(weights!=8)]==1).all())
        with self.assertRaises(ValueError):semantic_token_weights([row],y,0)
    def test_boundary_cases_reject_divisor_that_small_inputs_accept(self):
        task={'constant':257};old=[-21,0,19];new=case_inputs(self.plan,task)
        self.assertTrue(all(oracle('floor_divide',257,x)==oracle('floor_divide',258,x) for x in old))
        self.assertTrue(any(oracle('floor_divide',257,x)!=oracle('floor_divide',258,x) for x in new))
        self.assertTrue(any(oracle('floor_divide',257,x)!=oracle('floor_divide',256,x) for x in new))
        self.assertEqual(len(new),self.plan['target']['cases_per_task'])
        self.assertEqual(case_inputs({},task),old)
    def test_new_constants_are_disjoint_from_prior_experiments(self):
        previous=set()
        for version in (1,2):
            previous|={r['constant'] for r in records(read_plan(ROOT/f'experiments/coding-curriculum-v{version}/plan.json'))}
        self.assertFalse(previous & {r['constant'] for r in self.rows})
        self.assertEqual(sum(r['split']=='train' for r in self.rows),1920)
        self.assertEqual(sum(r['split']=='validation' for r in self.rows),8)
        self.assertEqual(sum(r['split']=='test' for r in self.rows),8)


if __name__=='__main__':unittest.main()

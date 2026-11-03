import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from ml.continue_field import validate_parent_manifest
from ml.train_field import domain_metrics, validation_loss
from ml.prepare_expanded_field import assign_new_splits


class ExpandedFieldTests(unittest.TestCase):
    def test_new_domain_contributes_to_metrics_and_balanced_loss(self):
        logits=torch.full((11,12),-10.)
        logits[:,0]=10.
        targets=torch.tensor([0]*10+[1])
        rows=[{'domain':'PlantVillage'}]*10+[{'domain':'FieldSupplement'}]
        self.assertIn('FieldSupplement',domain_metrics(logits,targets,rows))
        self.assertAlmostEqual(validation_loss(logits,targets,rows),10.,places=3)

    def test_expansion_preserves_parent_holdouts_and_rejects_overlap(self):
        parent=[{'path':'held.jpg','sha256':'held','label':'Corn_healthy','split':'validation'}]
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'records.json';path.write_text(json.dumps(parent))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            source={'training':{'config':{'records_sha256':digest}}}
            audit={'parent_records_path':str(path),'parent_records_sha256':digest}
            rows=[dict(parent[0],audit_group='held'),{'path':'new.jpg','sha256':'new','label':'Corn_healthy','split':'train','audit_group':'new'}]
            validate_parent_manifest(source,'expanded',audit,rows,True)
            with self.assertRaisesRegex(ValueError,'different data manifest'):
                validate_parent_manifest(source,'expanded',audit,rows,False)
            with self.assertRaisesRegex(ValueError,'changed or removed'):
                validate_parent_manifest(source,'expanded',audit,rows[1:],True)
            rows[1]['audit_group']='held'
            with self.assertRaisesRegex(ValueError,'overlaps'):
                validate_parent_manifest(source,'expanded',audit,rows,True)

    def test_new_splits_are_reproducible_and_include_each_class(self):
        rows=[{'label':label,'group':label+str(i)} for label in ['a','b'] for i in range(10)]
        again=[dict(row) for row in rows]
        assign_new_splits(rows);assign_new_splits(again)
        self.assertEqual(rows,again)
        self.assertEqual(sum(row['split']=='validation' for row in rows),4)
        healthy=[{'label':'Corn_healthy','source_name':'CornEcuadorHealthy','group':str(i)} for i in range(10)]
        assign_new_splits(healthy)
        self.assertTrue(all(row['split']=='train' for row in healthy))


if __name__=='__main__':
    unittest.main()

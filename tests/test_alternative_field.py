import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ml.prepare_alternative_field import CCMT_RAW_CLASSES, CCMT_SEPTORIA_FOLDER, bari_target, exclusion_reason, prepare, select_ccmt


class AlternativeFieldTests(unittest.TestCase):
    def test_only_original_bari_and_pinned_raw_ccmt_classes_are_selected(self):
        self.assertEqual(bari_target('Potato Leaf Disease Dataset/Fungal Late Blight/orig_1.jpg'), 'Potato_Late_blight')
        self.assertIsNone(bari_target('Potato Leaf Disease Dataset/Healthy/aug_0_1.jpg'))
        self.assertIsNone(bari_target('../Healthy/orig_1.jpg'))
        raw={'id':'a','filename':'septoria leaf spot10_.jpg','folder_id':CCMT_SEPTORIA_FOLDER,
             'content_details':{'sha256_hash':'a'*64}}
        augmented=dict(raw,id='b',filename='0tomato_train_septoria.JPG',folder_id='augmented')
        metadata={'id':'bwh3zbpkpv','version':1,'files':[raw,augmented]}
        self.assertEqual(select_ccmt(metadata),[raw])
        wrong_name=dict(raw,id='misfiled',filename='leaf blight802_.jpg',
                        folder_id='7b44bb20-452c-468d-9caf-9149ea7b21e7',content_details={'sha256_hash':'b'*64})
        self.assertEqual(select_ccmt(dict(metadata,files=[raw,wrong_name])),[raw])
        conflict=dict(raw,id='c',filename='leaf blight10_.jpg',folder_id='other-raw-class')
        with self.assertRaisesRegex(ValueError,'Unexpected raw CCMT inventory'):
            select_ccmt(dict(metadata,files=[raw,conflict]))
        self.assertEqual(CCMT_RAW_CLASSES['7b44bb20-452c-468d-9caf-9149ea7b21e7'][1],'Unsupported')
        with self.assertRaises(ValueError):select_ccmt(dict(metadata,version=2))

    def test_pending_label_reviews_exclude_training_only(self):
        row={'split':'train','label':'Tomato_healthy','original_label':'tomato leaf','source_dataset':'plantdoc'}
        self.assertEqual(exclusion_reason(row),'generic_healthy_pending_review')
        self.assertIsNone(exclusion_reason(dict(row,split='validation')))
        self.assertEqual(exclusion_reason(dict(row,original_label='Healthy',source_dataset='plantdoc_detection_crop')),
                         'crop_lineage_pending_review')

    def test_preparation_keeps_holdout_and_rejects_new_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);parent=root/'parent';parent.mkdir()
            def image(name,seed):
                path=root/name
                Image.fromarray(np.random.default_rng(seed).integers(0,256,(96,96,3),dtype=np.uint8)).save(path)
                return path
            held=image('held.png',1);new=image('new.png',2)
            old=[{'path':str(held),'sha256':hashlib.sha256(held.read_bytes()).hexdigest(),'label':'Corn_healthy',
                  'split':'validation','group':'held','domain':'PlantVillage','source_dataset':'plantvillage'}]
            manifest=parent/'records.json';manifest.write_text(json.dumps(old));before=manifest.read_bytes()
            (parent/'audit.json').write_text(json.dumps({'records_sha256':hashlib.sha256(before).hexdigest()}))
            additions=[{'path':str(held),'label':'Corn_healthy','source_dataset':'new'},
                       {'path':str(new),'label':'Corn_healthy','source_dataset':'new'}]
            output=root/'output';prepare(root/'hub',parent,output,additions)
            rows=json.loads((output/'records.json').read_text());audit=json.loads((output/'audit.json').read_text())
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['sha256'],old[0]['sha256'])
            self.assertEqual(rows[0]['split'],'validation')
            self.assertEqual(rows[1]['split'],'train')
            self.assertEqual(audit['excluded'][0]['reason'],'duplicate_or_near_match_parent_or_new')
            self.assertEqual(manifest.read_bytes(),before)
            with self.assertRaises(ValueError):prepare(root/'hub',parent,output,additions)

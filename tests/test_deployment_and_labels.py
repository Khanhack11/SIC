import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from backend.field_inference import FieldInferenceEngine
from backend.inference_engine import CropDiseaseInferenceEngine
from ml.benchmark_deployment import summarize
from ml.inspect_labels import export
from ml.prepare_field_v2 import extract_plantdoc_train_crops


class DeploymentAndLabelTests(unittest.TestCase):
    def test_cuda_request_is_never_silently_run_on_cpu(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'CUDA unavailable'):
                CropDiseaseInferenceEngine(device='cuda',runtime='eager')
            engine=FieldInferenceEngine('not-loaded.pt',device='cuda')
            self.assertFalse(engine.weights_loaded)
            self.assertIn('CUDA is unavailable',engine.load_error)

    def test_label_export_preserves_input_and_shows_actual_numeric_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';data.mkdir()
            manifest=data/'records.json'
            rows=[{'path':'corn.jpg','label':'Corn_healthy','split':'train','domain':'Field','source_dataset':'source'},
                  {'path':'other.jpg','label':'Unsupported','split':'validation','domain':'Field'}]
            manifest.write_text(json.dumps(rows));before=manifest.read_bytes()
            (data/'audit.json').write_text(json.dumps({'records_sha256':hashlib.sha256(before).hexdigest()}))
            output=root/'review.csv';summary=export(data,output)
            with output.open(encoding='utf-8-sig',newline='') as f: exported=list(csv.DictReader(f))
            self.assertEqual([r['class_index'] for r in exported],['3','11'])
            self.assertTrue(all(not r['proposed_label'] and not r['reviewer'] for r in exported))
            self.assertEqual(manifest.read_bytes(),before)
            self.assertEqual(summary['rows'],2)
            with self.assertRaises(ValueError):export(data,output)
            manifest.write_text('[]')
            with self.assertRaisesRegex(ValueError,'audit'):export(data,root/'bad.csv')

    def test_latency_report_distinguishes_images_from_batches(self):
        result=summarize([10.,20.,30.],batch_size=4)
        self.assertEqual(result['mean_ms'],20.)
        self.assertEqual(result['images_per_second'],200.)
        self.assertGreater(result['p95_ms'],result['p50_ms'])

    def test_training_crop_inherits_annotation_and_cannot_use_holdout_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);analysis=root/'analysis';analysis.mkdir()
            image=root/'parent.jpg';Image.new('RGB',(200,200),'green').save(image)
            sha=hashlib.sha256(image.read_bytes()).hexdigest()
            record={'path':str(image),'sha256':sha,'domain':'PlantDoc','split':'train',
                    'label':'Corn_Common_rust','group':'same-leaf'}
            (analysis/'plantdoc_detection_inventory.jsonl').write_text(json.dumps(
                {'annotation_key':'TRAIN/parent.jpg','sha256':sha})+'\n')
            box={'official_split':'train','annotation_status':'valid_geometry','filename':'parent.jpg',
                 'class':'corn rust leaf','xmin':20,'ymin':20,'xmax':120,'ymax':120}
            with (analysis/'detection_boxes.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(box));writer.writeheader();writer.writerow(box)
            crops,_=extract_plantdoc_train_crops(root,[record],root/'crops')
            self.assertEqual(len(crops),1)
            self.assertEqual((crops[0]['label'],crops[0]['original_label'],crops[0]['group'],crops[0]['split']),
                             ('Corn_Common_rust','corn rust leaf','same-leaf','train'))
            self.assertEqual(hashlib.sha256(Path(crops[0]['path']).read_bytes()).hexdigest(),crops[0]['sha256'])
            crops,skipped=extract_plantdoc_train_crops(root,[dict(record,split='validation')],root/'held')
            self.assertEqual(crops,[])
            self.assertEqual(skipped[0]['reason'],'parent_not_in_plantdoc_train')
            crops,_=extract_plantdoc_train_crops(root,[dict(record,label='Corn_healthy')],root/'conflict')
            self.assertEqual(crops,[])

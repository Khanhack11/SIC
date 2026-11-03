"""Optional V3 serving engine; activation requires an explicit candidate path."""
import io
from pathlib import Path
import time
from PIL import Image
import torch
from backend.mobile_model import FIELD_ARCHITECTURE, FIELD_CLASSES, FIELD_PREPROCESSING, build_field_model, field_transform, decode_field_image
from backend.preprocessing import orient_rgb
from backend.inference_engine import ModelNotReadyError, InputValidationError


class FieldInferenceEngine:
    def __init__(self,path):
        self.runtime='eager';self.device=torch.device('cpu');self.calibrated=False
        self.weights_loaded=False;self.load_error=None;self.temperature=1.
        self.model=build_field_model().eval().to(memory_format=torch.channels_last)
        try:
            checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
            for key,value in {'architecture':FIELD_ARCHITECTURE,'class_names':FIELD_CLASSES,
                              'preprocessing':FIELD_PREPROCESSING}.items():
                if checkpoint.get(key)!=value:raise ValueError('Field checkpoint contract mismatch: '+key)
            self.temperature=float(checkpoint['temperature'])
            if not .05<=self.temperature<=10:raise ValueError('Invalid field temperature')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.calibrated=bool(checkpoint.get('calibration'))
            self.weights_loaded=True
        except Exception as error:self.load_error=str(error)

    def logits(self,images):
        if not self.weights_loaded:raise ModelNotReadyError(self.load_error or 'Field checkpoint unavailable')
        return self.model(images.to('cpu',memory_format=torch.channels_last))

    def predict(self,image_bytes):
        if not self.weights_loaded:raise ModelNotReadyError(self.load_error or 'Field checkpoint unavailable')
        start=time.perf_counter()
        try:
            tensor=field_transform()(decode_field_image(image_bytes)).unsqueeze(0)
        except (ValueError,OSError) as error:raise InputValidationError('Không giải mã được ảnh.') from error
        with torch.inference_mode():values=(self.logits(tensor)/self.temperature).softmax(1)[0].tolist()
        index=max(range(len(values)),key=values.__getitem__);label=FIELD_CLASSES[index]
        latency=(time.perf_counter()-start)*1000
        return {'predicted_class':label,'crop':label.split('_')[0] if label!='Unsupported' else None,
                'is_healthy':label.endswith('_healthy'),'confidence':values[index],
                'confidence_percentage':f'{values[index]*100:.2f}%', 'latency_ms':round(latency,2),
                'calibrated':self.calibrated,'benchmark_met':latency<200,
                'all_probabilities':dict(zip(FIELD_CLASSES,values))}

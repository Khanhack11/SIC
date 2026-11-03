"""Optional V3 serving engine; activation requires an explicit candidate path."""
import hashlib
import json
from pathlib import Path
import time
import torch
from backend.mobile_model import FIELD_ARCHITECTURE, FIELD_CLASSES, FIELD_PREPROCESSING, build_field_model, field_transform, decode_field_image
from backend.config import IMAGENET_MEAN, IMAGENET_STD
from backend.inference_engine import ModelNotReadyError, InputValidationError


def field_onnx_contract(checkpoint, weights_path):
    return {'architecture': FIELD_ARCHITECTURE, 'class_names': FIELD_CLASSES,
            'preprocessing': FIELD_PREPROCESSING, 'input_size': [3,224,224],
            'mean': IMAGENET_MEAN, 'std': IMAGENET_STD, 'precision': 'fp32',
            'temperature': float(checkpoint['temperature']),
            'calibrated': bool(checkpoint.get('calibration')),
            'checkpoint_sha256': hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()}


class FieldInferenceEngine:
    def __init__(self,path,onnx_path=None,device='cpu'):
        self.runtime='onnx' if onnx_path else 'eager';self.device=torch.device(device);self.calibrated=False
        self.weights_loaded=False;self.load_error=None;self.temperature=1.
        self.model=None
        try:
            if self.device.type not in {'cpu','cuda'}:raise ValueError('Field device must be cpu or cuda')
            if self.device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable; use .venv-gpu with an NVIDIA GPU')
            if onnx_path and self.device.type!='cpu':raise ValueError('This ONNX profile uses CPU; choose PyTorch eager for CUDA')
            checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
            for key,value in {'architecture':FIELD_ARCHITECTURE,'class_names':FIELD_CLASSES,
                              'preprocessing':FIELD_PREPROCESSING}.items():
                if checkpoint.get(key)!=value:raise ValueError('Field checkpoint contract mismatch: '+key)
            self.temperature=float(checkpoint['temperature'])
            if not .05<=self.temperature<=10:raise ValueError('Invalid field temperature')
            self.calibrated=bool(checkpoint.get('calibration'))
            if onnx_path:
                import onnxruntime as ort
                options=ort.SessionOptions()
                options.intra_op_num_threads=torch.get_num_threads();options.inter_op_num_threads=1
                options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
                options.add_session_config_entry('session.intra_op.allow_spinning','0')
                self.session=ort.InferenceSession(str(onnx_path),options,providers=['CPUExecutionProvider'])
                contract=json.loads(self.session.get_modelmeta().custom_metadata_map['sic_field_contract'])
                if contract!=field_onnx_contract(checkpoint,path):raise ValueError('Field ONNX contract/checkpoint mismatch')
                inputs=self.session.get_inputs();outputs=self.session.get_outputs()
                if (len(inputs)!=1 or inputs[0].name!='images' or inputs[0].type!='tensor(float)' or
                    inputs[0].shape[1:]!=[3,224,224] or len(outputs)!=1 or outputs[0].name!='logits' or outputs[0].type!='tensor(float)' or
                    len(outputs[0].shape)!=2 or outputs[0].shape[1]!=len(FIELD_CLASSES)):
                    raise ValueError('Field ONNX input/output mismatch')
            else:
                self.model=build_field_model().eval().to(device=self.device,memory_format=torch.channels_last if self.device.type=='cpu' else torch.contiguous_format)
                self.model.load_state_dict(checkpoint['state_dict'])
            self.weights_loaded=True
        except Exception as error:self.load_error=str(error)

    def logits(self,images):
        if not self.weights_loaded:raise ModelNotReadyError(self.load_error or 'Field checkpoint unavailable')
        if self.runtime=='onnx':
            return torch.from_numpy(self.session.run(['logits'],{'images':images.detach().cpu().contiguous().numpy()})[0])
        return self.model(images.to(self.device,memory_format=torch.channels_last if self.device.type=='cpu' else torch.contiguous_format))

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

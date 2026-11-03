"""Measure actual CPU/CUDA execution of the two retained models, without using test labels."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import struct
import time

import numpy as np
import torch

from backend.config import MODEL_WEIGHTS_PATH, ONNX_MODEL_PATH, FIELD_MODEL_WEIGHTS_PATH, FIELD_ONNX_MODEL_PATH
from backend.field_inference import FieldInferenceEngine
from backend.inference_engine import CropDiseaseInferenceEngine, preprocess_image
from backend.mobile_model import decode_field_image, field_transform


def summarize(values, batch_size=1):
    return {'iterations':len(values),'mean_ms':statistics.mean(values),'p50_ms':float(np.percentile(values,50)),
            'p95_ms':float(np.percentile(values,95)),'images_per_second':batch_size*1000/statistics.mean(values)}


def measure(args):
    if args.output.exists():raise ValueError('Choose a new output; existing evidence is preserved')
    if args.device=='cuda' and (args.runtime!='eager' or not torch.cuda.is_available()):
        raise ValueError('CUDA requires PyTorch eager in .venv-gpu; this ONNX installation is CPU-only')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    paths=sorted(Path('sample_images/demo').glob('*.jpg'))
    if len(paths)<4:raise ValueError('Four demo images are required')
    paths=paths[:4];raw=[p.read_bytes() for p in paths]
    libraries={'torch_python_binding':str(torch._C.__file__),
               'torch_cpu_dll':str(Path(torch.__file__).parent/'lib/torch_cpu.dll')}
    if args.device=='cuda':libraries['torch_cuda_dll']=str(Path(torch.__file__).parent/'lib/torch_cuda.dll')
    if args.runtime=='onnx':
        import onnxruntime as ort
        libraries['onnxruntime_binding']=str(next((Path(ort.__file__).parent/'capi').glob('onnxruntime_pybind11_state*.pyd')))
    for path in libraries.values():assert Path(path).is_file(),path
    hardware={'os':platform.platform(),'cpu_architecture':platform.machine(),'python_bits':struct.calcsize('P')*8,
              'cpu_identifier':platform.processor(),'torch':str(torch.__version__),'threads':2,
              'device':args.device,'runtime':args.runtime,'native_libraries':libraries}
    if args.device=='cuda':hardware.update(gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        compute_capability=list(torch.cuda.get_device_capability()),total_vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    def sync():
        if args.device=='cuda':torch.cuda.synchronize()
    reports=[]
    for name,weights,graph in [('baseline',MODEL_WEIGHTS_PATH,ONNX_MODEL_PATH),('candidate',FIELD_MODEL_WEIGHTS_PATH,FIELD_ONNX_MODEL_PATH)]:
        if name=='baseline':
            engine=CropDiseaseInferenceEngine(weights,device=args.device,runtime=args.runtime,onnx_path=graph)
            prepared=[preprocess_image(b).squeeze(0) for b in raw]
        else:
            engine=FieldInferenceEngine(weights,onnx_path=graph if args.runtime=='onnx' else None,device=args.device)
            prepared=[field_transform()(decode_field_image(b)) for b in raw]
        if not engine.weights_loaded:raise RuntimeError(engine.load_error)
        assert engine.device.type==args.device
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        batches={}
        with torch.inference_mode():
            for size in [1,4]:
                images=torch.stack(prepared[:size]).to(engine.device)
                for _ in range(args.warmup):engine.logits(images)
                times=[]
                for _ in range(args.iterations):
                    sync();started=time.perf_counter();output=engine.logits(images);sync()
                    times.append((time.perf_counter()-started)*1000)
                assert output.device.type==args.device and torch.isfinite(output).all()
                batches[str(size)]=summarize(times,size)
            for i in range(args.warmup):engine.predict(raw[i%len(raw)])
            times=[]
            for i in range(args.iterations):
                sync();started=time.perf_counter();engine.predict(raw[i%len(raw)]);sync()
                times.append((time.perf_counter()-started)*1000)
            predictions=[engine.predict(b)['predicted_class'] for b in raw]
        report={'model':name,'weights':str(weights),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                'temperature':engine.temperature,'parameter_count':sum(p.numel() for p in engine.model.parameters()) if engine.model is not None else None,
                'forward_only_by_batch':batches,'image_pipeline':summarize(times),'fixture_predictions':predictions}
        if args.runtime=='onnx':report['onnx_sha256']=hashlib.sha256(graph.read_bytes()).hexdigest()
        if args.device=='cuda':report['peak_allocated_vram_bytes']=torch.cuda.max_memory_allocated()
        reports.append(report);print(name,args.device,args.runtime,report['image_pipeline'],flush=True)
        del engine,prepared,images,output;gc.collect()
        if args.device=='cuda':torch.cuda.empty_cache()
    evidence={'hardware':hardware,'warmup':args.warmup,'iterations':args.iterations,
              'fixtures':{str(p):hashlib.sha256(b).hexdigest() for p,b in zip(paths,raw)},
              'scope':'Forward excludes preprocessing/transfer. Image pipeline includes decode, transform, device transfer, model and probabilities; excludes HTTP, advisory and network.',
              'quality_claim':False,'precision':'FP32; no autocast','results':reports}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(evidence,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--runtime',choices=['eager','onnx'],default='onnx')
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--iterations',type=int,default=50)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.warmup,args.iterations)<1:parser.error('Counts must be positive')
    measure(args)

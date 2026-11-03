"""Export an optional V3 FP32 graph; retain preprocessing outside the graph and bind its contract."""
import argparse
import json
from pathlib import Path

import onnx
import torch

from backend.field_inference import FieldInferenceEngine, field_onnx_contract


def export(weights, output):
    if output.exists(): raise ValueError('Preserve existing graph; choose a new output')
    engine=FieldInferenceEngine(weights)
    if not engine.weights_loaded: raise ValueError(engine.load_error)
    checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.partial.onnx')
    if temporary.exists(): raise ValueError('Previous partial export preserved; choose a new output')
    torch.onnx.export(engine.model.to(memory_format=torch.contiguous_format),torch.zeros(1,3,224,224),str(temporary),
                      input_names=['images'],output_names=['logits'],opset_version=17,dynamo=False,
                      dynamic_axes={'images':{0:'batch'},'logits':{0:'batch'}})
    graph=onnx.load(str(temporary))
    onnx.helper.set_model_props(graph,{'sic_field_contract':json.dumps(field_onnx_contract(checkpoint,weights))})
    onnx.checker.check_model(graph);onnx.save(graph,str(temporary));temporary.replace(output)
    print('Exported field FP32:',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,default=Path('models/mobilenetv3_field_continued_20261103.pt'))
    parser.add_argument('--output',type=Path,default=Path('models/mobilenetv3_field_continued_20261103.onnx'))
    args=parser.parse_args();torch.set_num_threads(2);export(args.weights,args.output)

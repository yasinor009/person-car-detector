"""Exercise a real tiny ONNX session + public evaluator entry point end-to-end."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest


def constant_model(path):
    onnx=pytest.importorskip('onnx')
    from onnx import helper,TensorProto,numpy_helper
    a=np.zeros((1,6,2),np.float32)
    a[0,:4,0]=[48,64,32,64];a[0,4,0]=.95
    a[0,:4,1]=[96,64,32,32];a[0,5,1]=.96
    node=helper.make_node('Constant',[],['output0'],value=numpy_helper.from_array(a))
    graph=helper.make_graph([node],'contract_test',[helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,128,128])],
                            [helper.make_tensor_value_info('output0',TensorProto.FLOAT,[1,6,2])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=10
    helper.set_model_props(model,{'names':"{0:'person',1:'car'}",'end2end':'False'})
    onnx.save(model,path)


def test_real_evaluator_keeps_150m_gate_failed_without_measurement(tmp_path):
    from evaluate_laptop import main
    p=tmp_path/'model.onnx';constant_model(p)
    root=tmp_path/'dataset';(root/'images').mkdir(parents=True);(root/'labels').mkdir()
    cv2.imwrite(str(root/'images/scene.png'),np.zeros((128,128,3),np.uint8))
    (root/'labels/scene.txt').write_text('0 0.375 0.5 0.25 0.5\n1 0.75 0.5 0.25 0.25\n')
    output=tmp_path/'report.json'
    code=main(['--dataset',str(root),'--model',str(p),'--split','test','--label-schema','two-class',
               '--device','cpu','--min-objects','1','--min-fps','0.01','--output',str(output),'--imgsz','128'])
    report=json.loads(output.read_text())
    assert report['metrics']['true_positives']==2
    assert report['metrics']['false_positives']==0
    assert report['acceptance']['general']['passed']
    assert not report['acceptance']['long_range']['passed']
    assert code==2

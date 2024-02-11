from loguru import logger
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import select_device
from ultralytics.models.yolo.detect import DetectionValidator
from typing import Dict
from pathlib import Path
import openvino as ov
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
import os
import torch
import nncf
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.engine.validator import BaseValidator as Validator
from openvino.runtime import serialize


def read_openvino_model(model_path, device = 'CPU'):
    """
    :param model_path: .xml file
    :return:
    """
    ov_model = ov.Core().read_model(model_path)
    if device != "CPU":
        ov_model.reshape({0: [1, 3, 640, 640]})
    det_compiled_model = ov.Core().compile_model(ov_model, device)
    return det_compiled_model


def create_folder(output_folder):
    isExist = os.path.exists(output_folder)
    if not isExist:
        os.makedirs(output_folder)


def validation_metric(
    compiled_model: ov.CompiledModel,
    validation_loader: torch.utils.data.DataLoader,
    validator: Validator,
    num_samples: int = None,
    log=True,
    key = 'metrics/mAP50(B)' # "metrics/mAP50-95(B)"
) -> float:
    validator.seen = 0
    validator.jdict = []
    validator.stats = []
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    num_outputs = len(compiled_model.outputs)

    counter = 0
    for batch_i, batch in enumerate(validation_loader):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        results = compiled_model(batch["img"])
        if num_outputs == 1:
            preds = torch.from_numpy(results[compiled_model.output(0)])
        else:
            preds = [
                torch.from_numpy(results[compiled_model.output(0)]),
                torch.from_numpy(results[compiled_model.output(1)]),
            ]
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
        counter += 1
    stats = validator.get_stats()
    if num_outputs == 1:
        stats_metrics = stats[key]
    else:
        stats_metrics = stats[key]
    if log:
        print(f"Validate: dataset length = {counter}, metric value = {stats_metrics:.3f}")
    return stats_metrics

def export2openvino(pathmodel, device = 'CPU'):
    core = ov.Core()
    pt_model = YOLO(pathmodel)
    ov_model_path = pathmodel.split('.pt')[0] + '_openvino_model/' + os.path.basename(pathmodel)[:-3] + '.xml'
    if not Path(ov_model_path).exists():
        pt_model.export(format="openvino", dynamic=True, half=False)
        logger.info('save ov model to {}'.format(ov_model_path))
    else:
        logger.info('model already exists in {}'.format(ov_model_path))
    ov_model = ov.Core().read_model(ov_model_path)

    if device!= "CPU":
        ov_model.reshape({0: [1, 3, 640, 640]})

    det_compiled_model = core.compile_model(ov_model, device)
    return ov_model, pt_model, det_compiled_model

def compile_validator(args, pt_modelpath, yaml_datapath, dir_calibration_runs = './calibration_valStats'):
    args.model = pt_modelpath
    args.data = yaml_datapath
    validator = DetectionValidator(args=args)
    validator.is_coco = False

    model = AutoBackend(
        validator.args.model,
        device=select_device(validator.args.device, validator.args.batch),
        dnn=validator.args.dnn,
        data=validator.args.data,
        fp16=validator.args.half,
    )

    validator.names = model.names
    validator.nc = len(model.names)
    validator.metrics.names = validator.names
    validator.metrics.plot = validator.args.plots
    validator.data = check_det_dataset(args.data)
    validator.training = False
    validator.stride = model.stride
    validator.save_dir = Path(dir_calibration_runs)
    return validator

def prepare_quantization_dataset(pt_modelpath, args, datapath, yaml_datapath):
    validator = compile_validator(args, pt_modelpath, yaml_datapath)
    data_loader = validator.get_dataloader(datapath, 1)
    def transform_fn(data_item: Dict):
        input_tensor = validator.preprocess(data_item)["img"].numpy()
        return input_tensor

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)
    return quantization_dataset, validator, data_loader

def get_accuracy(validation_metric, model_path, args, pt_modelpath, datapath='./', yaml_datapath='./datadata.yaml'):
    """
    model_path : xml
    pt_modelpath : pt
    """
    quantization_dataset, validator, data_loader = prepare_quantization_dataset(pt_modelpath, args, datapath, yaml_datapath)
    logger.info(f"Accuracy test for === {model_path} ")
    model_ov = read_openvino_model(model_path)
    metric = validation_metric(model_ov,validation_loader = data_loader, validator = validator)
    return metric

def save_quantized_model(quantized_det_model, DET_MODEL_NAME, info = 'basic quantization'):
    output_folder = f'{os.path.dirname(os.path.realpath(__file__))}/quantized_res/{DET_MODEL_NAME}_openvino_int8_model/'
    create_folder(output_folder)
    int8_model_det_path = f'{output_folder}{DET_MODEL_NAME}.xml'
    logger.info(info)
    logger.info(f"Quantized detection model will be saved to {int8_model_det_path}")
    serialize(quantized_det_model, str(int8_model_det_path))

from utils import *
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
import platform
from functools import partial


def quantize_basic(ov_model, pt_modelpath, args , ignored_scope = None, datapath='./', yaml_datapath='../data.yaml',  get_model_graph = False):
    quantization_dataset , _ , _= prepare_quantization_dataset(pt_modelpath=pt_modelpath, args=args, \
                                                        datapath=datapath, yaml_datapath=yaml_datapath)
    if get_model_graph:
        from nncf.common.utils.dot_file_rw import write_dot_graph
        graph = (GraphConverter.create_nncf_graph(ov_model))
        write_dot_graph(graph._get_graph_for_visualization(), "./graph/dump/graph_model.dot")
        logger.info(f"\n\nmodel graph saved to directory = ./graph/dump/graph_model.dot\n")

    quantized_det_model = nncf.quantize(
        ov_model,
        quantization_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=ignored_scope
    )
    DET_MODEL_NAME = os.path.basename(pt_modelpath)[:-3] + '_noAC'
    save_quantized_model(quantized_det_model, DET_MODEL_NAME)
    return quantized_det_model

def quantize_AC(ov_model, validation_metric,pt_modelpath, args, ignored_scope = None, datapath='./', yaml_datapath='../data.yaml' ):
    """
    * Besides the calibration dataset, a validation dataset is required to \
    compute the accuracy metric. Both datasets can refer to the same data in the simplest case.

    * Validation function, used to compute accuracy metric is required.\
     It can be a function that is already available in the source framework or a custom function.

    * Since accuracy validation is run several times during the quantization process, quantization \
    with accuracy control can take more time than the Basic 8-bit quantization flow.

    * The resulted model can provide smaller performance improvement than the Basic 8-bit quantization \
    flow because some of the operations are kept in the original precision.

    """

    quantization_dataset, validator, data_loader = prepare_quantization_dataset(pt_modelpath, args, datapath, yaml_datapath)
    validation_fn = partial(validation_metric, validator=validator, log=False)
    quantized_model = nncf.quantize_with_accuracy_control(
        ov_model,
        quantization_dataset,
        quantization_dataset,
        validation_fn=validation_fn,
        max_drop=0.01,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=128,
        advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(
            ranking_subset_size=25
        ),
        ignored_scope=ignored_scope
    )
    DET_MODEL_NAME = os.path.basename(pt_modelpath)[:-3] + '_AC'
    save_quantized_model(quantized_model, DET_MODEL_NAME, info = 'Quantization with Accuracy Control DONE')

    return quantized_model

def main_basic(pt_modelpath, args, get_model_graph = False, datapath = '../datasets/images', yaml_datapath='../data.yaml'):
    ov_model, pt_model, det_compiled_model = export2openvino(pt_modelpath)

    ignored_scope = nncf.IgnoredScope(
        types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
        names=[
            "/model.22/dfl/conv/Conv",  # in the post-processing subgraph
            "/model.22/Add",
            "/model.22/Add_1",
            "/model.22/Add_2",
            "/model.22/Add_3",
            "/model.22/Add_4",
            "/model.22/Add_5",
            "/model.22/Add_6",
            "/model.22/Add_7",
            "/model.22/Add_8",
            "/model.22/Add_9",
            "/model.22/Add_10"
        ]
    )
    quantized_basic_model = quantize_basic(ov_model=ov_model, pt_modelpath=pt_modelpath, args=args, ignored_scope = ignored_scope,datapath=datapath,  get_model_graph=get_model_graph)
def main_AC(validation_metric, pt_modelpath,args, datapath, yaml_datapath='../data.yaml'):
    ov_model, _, _ = export2openvino(pt_modelpath)
    quantized_AC_model = quantize_AC(ov_model = ov_model, datapath=datapath, validation_metric = validation_metric, pt_modelpath = pt_modelpath, args = args)
    return quantized_AC_model

def main():
    pt_modelpath = '../yolov8s.pt'
    yaml_datapath = '../data.yaml'
    datapath = '../datasets/images' # path to images folder
    args = get_cfg(cfg=DEFAULT_CFG)
    # RUN QUANTIZATION
    main_basic(pt_modelpath, args, datapath=datapath, yaml_datapath=yaml_datapath, get_model_graph=True)  # or main_AC(validation_metric, pt_modelpath, args)
    main_AC(validation_metric, pt_modelpath, args, datapath=datapath, yaml_datapath = yaml_datapath)

    # if you already quantized model:
    xml_noAC = './quantized_res/yolov8s_noAC_openvino_int8_model/yolov8s_noAC.xml'
    metric_noAC = get_accuracy(validation_metric=validation_metric, model_path=xml_noAC, args=args, \
                               pt_modelpath=pt_modelpath, yaml_datapath=yaml_datapath, datapath=datapath)
    logger.info(f'\nmetric noAC: {metric_noAC}')

if __name__ == "__main__":
    print("Python version:", platform.python_version())
    main()








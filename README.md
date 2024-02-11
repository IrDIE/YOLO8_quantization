# Faster your YOLOv8 (detection) inference with openVINO quantization

## Currently available: 
* Base quantization, quantization with accuracy control.
* Accuracy test for quantized model 

## How use:
* git clone
* refactor main() in `quantize.py` with your paths
* run `quantize.py`

---
### Base quantization (`quantize/main_basic()`)

Need:
* .pt model file
* calibration dataset (In standart YOLO format, with labels. OpenVINO usually use about 300 images)
* need to write standart .yaml file that describe dataset
  *  example of .yaml file:

```
#file: calibration.yaml
path: ./yolo_dataset # your dataset root dir

train: images # train images (relative to 'path') 
val: images # val images (relative to 'path') 

# Classes
names:
  0: car
  ...
```
* (optional) you can define `ignored_scope` in `main_basic()` or `main_AC()` functions like in *openVINO tutorial - https://docs.openvino.ai/2022.3/basic_qauntization_flow.html#set-up-an-environment*

<details>
  <summary>HOW SET LAYERS OF YOUR MODEL IN ignored_scope </summary>
  
  In OpenVino example you can see that layer names for `ignored_scope` parameter looks like this:
  ```python
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
  ```
  If you want to see all layer names of your model &rarr; pass  `get_model_graph = True` in `main_basic()` function.
  It will save .dot file in `graph/dump/graph_model.dot`. Looking at this file you can understand all layer names to pass it to `ignored_scope`.
  Also, you can visualise it file (for example in https://www.devtoolsdaily.com/graphviz/):
  
  | https://github.com/IrDIE/YOLO8_quantization/blob/master/readme_imgs/gramp_example.png |
  |---------------------|


  
</details>

---
### Quantization with accuracy control (`quantize/main_AC()`)

Need:
* all from **Base quantization**
* `validation_metric` - in this repo - mAp50 (but you can change it on your own, more info in **Prepare validation function** - https://docs.openvino.ai/2023.3/notebooks/122-yolov8-quantization-with-accuracy-control-with-output.html  )
* (optional) - you can use diffirent calibration and validation dataset. In this repo they are the same.

---

### Note - calibration can take a long time depending on your dataset size and `ignored_scope` (and others quantization parameters)

---
### Accuracy test after quantization (`utils/compare_accuracy()`)
```angular2html
compare_accuracy(validation_metric, model_path , args, pt_modelpath )
```
* validation_metric - same function as in **Quantization with accuracy control**
* model_path - path to .xml file (**IMPORTANT - .bin FILE SHOULD BE IN SAME DIRECTORY WHERE .xml SAVED**)
* args - `from ultralytics.utils import DEFAULT_CFG` `args = get_cfg(cfg=DEFAULT_CFG)`
* pt_modelpath - path to .pt 

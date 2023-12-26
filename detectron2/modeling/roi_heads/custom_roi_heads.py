from .roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.utils.registry import Registry
from .custom_fast_rcnn import CustomRCNNOutput

ROI_HEADS_REGISTRY = Registry("CustomROIHeads")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):
  def __init__(self, cfg, input_shape):
    super().__init__(cfg, 
                     input_shape,
                     box_predictor=CustomRCNNOutput(cfg))

        

    

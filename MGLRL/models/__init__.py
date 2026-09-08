from . import text_model
from . import vision_model
from . import MGLRL_model
from . import cnn_backbones


IMAGE_MODELS = {
    "classification": vision_model.ImageEncoder
}



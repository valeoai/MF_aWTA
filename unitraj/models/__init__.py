from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.mtr.MTR_wo_anchor import MotionTransformer as MotionTransformer_no_anchor
__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'MTR_wo_anchor': MotionTransformer_no_anchor
    
}


def build_model(config):
    print(f"################# {config.method.model_name} #################")
    model = __all__[config.method.model_name](
        config=config
    )

    return model

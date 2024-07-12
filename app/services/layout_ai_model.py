
import layoutparser as lp

class Detectron2LayoutModelSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if Detectron2LayoutModelSingleton._instance is None:
            Detectron2LayoutModelSingleton()
        return Detectron2LayoutModelSingleton._instance

    def __init__(self):
        if Detectron2LayoutModelSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
            )
            Detectron2LayoutModelSingleton._instance = self

    def detect(self, image):
        layout = self.model.detect(image)
        return layout

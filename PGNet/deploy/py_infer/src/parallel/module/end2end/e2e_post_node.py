import numpy as np

from ....infer import TextEnd2End
from ...framework import ModuleBase


class E2EPostNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super().__init__(args, msg_queue)
        self.text_detector = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_detector = TextEnd2End(self.args)
        self.text_detector.init(preprocess=False, model=False, postprocess=True)
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data
        res = self.text_detector.postprocess(data["pred"], data["shape_list"])

        infer_res_list = []
        for poly, score, text in zip(res["polys"][0], res["scores"][0], res["texts"][0]):
            if isinstance(poly, np.ndarray):
                poly = poly.astype(int).tolist()
            infer_res_list.append({"points": poly, "score": score, "transcription": text})

        input_data.infer_result = infer_res_list

        input_data.data = None

        if not self.args.vis_pipeline_save_dir:
            input_data.frame = None

        if not infer_res_list:
            input_data.skip = True

        self.send_to_next_module(input_data)

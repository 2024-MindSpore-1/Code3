from ....infer import TextEnd2End
from ...framework import ModuleBase


class E2EPreNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super().__init__(args, msg_queue)
        self.text_detector = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_detector = TextEnd2End(self.args)
        self.text_detector.init(preprocess=True, model=False, postprocess=False)
        super().init_self_args()
        return self.text_detector.get_params()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        image = input_data.frame[0]  # bs = 1 for det
        data = self.text_detector.preprocess(image)

        if not self.args.vis_pipeline_save_dir:
            input_data.frame = None

        input_data.data = data

        self.send_to_next_module(input_data)

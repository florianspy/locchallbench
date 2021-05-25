from easydict import EasyDict as edict
import PySimpleGUIQt as psg
from PIL import Image, ImageQt
import matplotlib.pyplot as plt
import numpy as np

# it creates a file
############
## CONFIG ##
############
# if you get error index out of bounce restart the whole app it can happen when something is copied away
# the reason you get the error out of bounce is that you put the gt data in the field where the evaluation data should have been put in.
# zero-size array to reduction operation maximum which has no identity you probably have a thing where no timestamps at all matched check if gt is the correct for this dataset
 
cfg = edict()

# window
cfg.window = edict()
cfg.window.title = "Benchmark"
cfg.window.no_titlebar = False

# ui-elements
cfg.uie = edict()

cfg.uie.eval_text = "Eval path:"
cfg.uie.gt_text = "Groundtruth path:"

cfg.uie.eval_key = "eval"
cfg.uie.gt_key = "gt"


cfg.uie.browse_btn = "browse"
cfg.uie.file_dialog = "Open File"
cfg.uie.eval_btn = "evaluate"

# fmt: off
cfg.uie.allowed_files = (("File", "*.txt",),)
# fmt: on

# image
cfg.img = edict()
cfg.img.size = (800, 800)

# error
cfg.error = edict()
cfg.error.eval_missing = "Parameter '{}' is missing, please enter it."

# THEME
psg.theme("Dark")


class GUI:
    def __init__(self, cfg):
        self.cfg = cfg

        # interface function to override
        self.on_eval = None

        self.out_img_path = "out.pdf"

    @property
    def layout(self):
        # fmt: off
        self.img = psg.Image(filename=None)

        return [
            [self.img],
            [psg.Text(self.cfg.uie.eval_text), psg.Input(key=self.cfg.uie.eval_key), psg.FileBrowse(self.cfg.uie.browse_btn, file_types=self.cfg.uie.allowed_files)],
            [psg.Text(self.cfg.uie.gt_text), psg.Input(key=self.cfg.uie.gt_key), psg.FileBrowse(self.cfg.uie.browse_btn, file_types=self.cfg.uie.allowed_files)],
            [psg.Button(cfg.uie.eval_btn)],
        ]
        # fmt: on

    def run(self):
        window = psg.Window(
            self.cfg.window.title, self.layout, no_titlebar=self.cfg.window.no_titlebar
        )

        # event loop
        while True:
            event, values = window.read()

            if event == psg.WIN_CLOSED or event == "Exit":
                break
            elif event == self.cfg.uie.eval_btn:
                data = self.on_eval_clicked(values)
                if data is None:
                    continue

                error = self.run_eval(*data)
                if error:
                    continue

                self.img.update(filename=self.out_img_path)

        window.close()

    def on_eval_clicked(self, values):
        eval_path = values[self.cfg.uie.eval_key]
        if not eval_path:
            psg.popup(self.cfg.error.eval_missing.format("Eval Path (string)"))
            return None

        gt_path = values[self.cfg.uie.gt_key]
        if not gt_path:
            psg.popup(self.cfg.error.eval_missing.format("Ground Truth Path (string)"))
            return None

        return eval_path, gt_path

    def run_eval(self, eval_path, gt_path):
        try:
            from evaluate import evaluate

            evaluate(eval_path, gt_path)
            return False
        except Exception as e:
            print(e)
            return True


if __name__ == "__main__":
    gui = GUI(cfg)
    gui.run()

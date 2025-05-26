import importlib
import warnings

from exp.exp_base import Exp_Basic

warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.model = self._get_model()

    def _get_model(self):
        last_dot_index = self.args.model.rfind('.')
        file = self.args.model[:last_dot_index]
        module = self.args.model[last_dot_index + 1:]
        model_class = getattr(importlib.import_module(f'{file}'), module)
        model = model_class(self.args).to(self.device)
        return model

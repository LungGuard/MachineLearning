from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

class CustomRichProgressBar(RichProgressBar):
    def __init__(self, comperator=None, mapper=None, refresh_rate=100, leave=False, theme=None, console_kwargs=None):
        
        self.comperator = comperator
        self.mapper = mapper

        # 2. THE FIX: If no theme is provided, instantiate the default one!
        if theme is None:
            theme = RichProgressBarTheme()

        # 3. Pass the valid theme up to the parent class
        super().__init__(
            refresh_rate=refresh_rate, 
            leave=leave, 
            theme=theme, 
            console_kwargs=console_kwargs
        )

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)

        formated_items = dict(sorted(items.items(), key=self.comperator)) if self.comperator else items
        formated_items = dict(map(self.mapper, formated_items.items())) if self.mapper else formated_items

        return formated_items
from RealMLP import RealMLP


class RealMLPWide(RealMLP):
    """Wide+RealMLP hybrid: RealMLP nonlinear branch plus linear wide branch."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("model_type", "RealMLPWide")
        kwargs.setdefault("wide_enabled", True)
        super().__init__(*args, **kwargs)

import torch

class TorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor", "cudagraphs"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "custom"

    torch._logging.set_logs(output_code=True)

    def patch(self, model, backend):
        m = model.clone()

        torch.set_float32_matmul_precision("medium")

        options = {
            'max_autotune': True,
            'triton.cudagraphs': True
        }

        m.add_object_patch("diffusion_model", torch.compile(model=m.get_model_object("diffusion_model"), backend=backend, options=options))
        return (m, )

NODE_CLASS_MAPPINGS = {
    "TorchCompileModelCustom": TorchCompileModel
}

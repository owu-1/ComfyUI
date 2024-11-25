import torch
import folder_paths
import comfy.utils

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

        options = {
            'max_autotune': True,
            'triton.cudagraphs': True
        }

        m.add_object_patch("diffusion_model", torch.compile(model=m.get_model_object("diffusion_model"), backend=backend, options=options))
        return (m, )


class AddFluxEstimator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "guidance_ckpt_name": (folder_paths.get_filename_list("checkpoints"), )
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "custom"

    def patch(self, model, guidance_ckpt_name):
        guidance_ckpt_file_path = folder_paths.get_full_path("checkpoints", guidance_ckpt_name)
        state_dict = comfy.utils.load_torch_file(guidance_ckpt_file_path)

        model.get_model_object("diffusion_model").distilled_guidance_layer.load_state_dict(state_dict)

        return (model, )


NODE_CLASS_MAPPINGS = {
    "TorchCompileModelCustom": TorchCompileModel,
    "AddFluxEstimator": AddFluxEstimator
}

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation


class Segmenter:
    def __init__(self):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-1.4", trust_remote_code=True
        )
        self.device = self._get_best_device()
        print("Segmenter using device:", self.device)
        self.model.to(self.device)

    def _get_best_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")

        # Apple Silicon (macOS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _preprocess_image(self, im: np.ndarray, model_input_size: list) -> torch.Tensor:
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        # orig_im_size=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        # im_tensor = torch.from_numpy(im).float()
        im_tensor = F.interpolate(
            torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
        )
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image

    def _postprocess_image(self, result: torch.Tensor, im_size: list) -> np.ndarray:
        result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array

    def segment(self, image: np.ndarray):
        orig_im_size = image.shape[0:2]
        model_input_size = [1024, 1024]
        # model_input_size = [512, 512]
        # model_input_size = [512, 512]
        # model_input_size = [768, 768]
        image = self._preprocess_image(image, model_input_size).to(self.device)

        # inference
        with torch.no_grad():
            result = self.model(image)

        # post process
        result_image = self._postprocess_image(result[0][0], orig_im_size)

        return result_image

if __name__ == "__main__":
    pass


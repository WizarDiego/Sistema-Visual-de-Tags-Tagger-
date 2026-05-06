import os
import re

from PIL import Image, ImageDraw
import torch

_SHARED_FLORENCE_MODELS = {}
_SHARED_FLORENCE_PROCESSORS = {}
_SHARED_MODEL_CACHE_PATHS = {}

FLORENCE_MODELS = {
    "microsoft-large": "microsoft/Florence-2-large",
    "promptgen-large-v2": "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
}


class AIModelManager:
    def __init__(self, models_dir=None, device_preference="auto", model_key="microsoft-large"):
        self.device_preference = device_preference
        self.device = self.resolve_device(device_preference)
        self.model_key = model_key if model_key in FLORENCE_MODELS else "microsoft-large"
        self.model_id = FLORENCE_MODELS[self.model_key]

        if models_dir:
            os.environ["HF_HOME"] = models_dir
            os.environ["TRANSFORMERS_CACHE"] = models_dir

        self.florence_model = None
        self.florence_processor = None
        self.florence_dtype = torch.float16 if "cuda" in self.device else torch.float32

    @staticmethod
    def resolve_device(device_preference):
        if device_preference == "cpu":
            return "cpu"
        if device_preference == "gpu":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def gpu_available():
        return torch.cuda.is_available()

    def load_florence2(self):
        global _SHARED_FLORENCE_MODELS, _SHARED_FLORENCE_PROCESSORS, _SHARED_MODEL_CACHE_PATHS

        if self.florence_model is not None and self.florence_processor is not None:
            return

        cache_key = (self.model_key, self.device)
        if cache_key in _SHARED_FLORENCE_MODELS and cache_key in _SHARED_FLORENCE_PROCESSORS:
            self.florence_model = _SHARED_FLORENCE_MODELS[cache_key]
            self.florence_processor = _SHARED_FLORENCE_PROCESSORS[cache_key]
            return

        if self.florence_model is None:
            print(f"Carregando {self.model_id}...")

            from huggingface_hub import snapshot_download

            from florence2_local.comfy_mock import inject_comfy_mock

            inject_comfy_mock()

            print(f"Baixando pesos de {self.model_id} (isso pode demorar na primeira vez)...")
            model_path = snapshot_download(
                repo_id=self.model_id,
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.py"],
                local_dir_use_symlinks=False,
            )

            from florence2_local.config import Florence2Config
            from florence2_local.model import Florence2
            from florence2_local.processing import Processor
            from safetensors.torch import load_file

            config_path = os.path.join(model_path, "config.json")
            checkpoint_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(model_path, "pytorch_model.bin")

            config = Florence2Config.from_json(config_path)
            dtype = self.florence_dtype

            import comfy.ops

            self.florence_model = Florence2(
                config,
                dtype=dtype,
                device=self.device,
                operations=comfy.ops,
            )

            print("Carregando pesos na memoria...")
            state_dict = (
                load_file(checkpoint_path)
                if checkpoint_path.endswith(".safetensors")
                else torch.load(checkpoint_path, map_location="cpu")
            )
            for key in [
                "language_model.model.encoder.embed_tokens.weight",
                "language_model.model.decoder.embed_tokens.weight",
            ]:
                if key in state_dict and "language_model.model.shared.weight" in state_dict:
                    state_dict.pop(key, None)

            self.florence_model.load_state_dict(state_dict, strict=False)
            self.florence_model.language_model.tie_weights()
            self.florence_model = self.florence_model.to(self.device).eval()

            self.florence_processor = Processor(model_path=model_path)
            _SHARED_FLORENCE_MODELS[cache_key] = self.florence_model
            _SHARED_FLORENCE_PROCESSORS[cache_key] = self.florence_processor
            _SHARED_MODEL_CACHE_PATHS[self.model_key] = model_path
            print(f"{self.model_id} carregado com sucesso.")

    def preload_models(self):
        self.load_florence2()
        return f"{self.model_id} carregado/baixado com sucesso."

    def is_florence_available_locally(self):
        if self.model_key in _SHARED_MODEL_CACHE_PATHS:
            return True
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=self.model_id,
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.py"],
                local_files_only=True,
                local_dir_use_symlinks=False,
            )
            return True
        except Exception:
            return False

    def _run_generation(self, image_path, prompt, max_new_tokens=96, num_beams=3, do_sample=False):
        self.load_florence2()

        image = Image.open(image_path).convert("RGB")
        import torchvision.transforms.functional as F

        img_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        inputs = self.florence_processor(text=prompt, images=img_tensor)
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(
            dtype=self.florence_dtype,
            device=self.device,
        )

        with torch.no_grad():
            generation_kwargs = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "repetition_penalty": 1.18,
                "no_repeat_ngram_size": 4,
            }
            try:
                generated_ids = self.florence_model.generate(**generation_kwargs)
            except TypeError:
                generation_kwargs.pop("repetition_penalty", None)
                generation_kwargs.pop("no_repeat_ngram_size", None)
                generated_ids = self.florence_model.generate(**generation_kwargs)

        generated_text = self.florence_processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        return image, generated_text.replace("</s>", "").replace("<s>", "").strip()

    def analyze_with_florence(self, image_path, task_prompt="<CAPTION>", max_new_tokens=96, num_beams=3):
        image, clean_results = self._run_generation(
            image_path,
            task_prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        parsed_answer = self.florence_processor.post_process_generation(
            clean_results,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        return parsed_answer[task_prompt]

    def generate_caption(self, image_path, task_prompt="<CAPTION>", max_new_tokens=512):
        return self.analyze_with_florence(
            image_path,
            task_prompt=task_prompt,
            max_new_tokens=max_new_tokens,
            num_beams=3,
        )

    def generate_tags(self, image_path, max_new_tokens=256):
        prompt = "<GENERATE_TAGS>" if self.model_key.startswith("promptgen") else "<CAPTION>"
        _, raw_tags = self._run_generation(image_path, prompt, max_new_tokens=max_new_tokens, num_beams=3)
        return normalize_tag_text(raw_tags)

    def generate_prompt_analysis(self, image_path, max_new_tokens=1024):
        prompt = "<ANALYZE>" if self.model_key.startswith("promptgen") else "<CAPTION>"
        _, raw_text = self._run_generation(
            image_path,
            prompt,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False,
        )
        return raw_text.replace("<ANALYZE>", "").replace("<MIXED_CAPTION_PLUS>", "").strip()

    def generate_bounding_boxes(self, image_path, max_new_tokens=96):
        return self.analyze_with_florence(
            image_path,
            task_prompt="<OD>",
            max_new_tokens=max_new_tokens,
            num_beams=3,
        )


def normalize_tag_text(raw_text):
    text = raw_text.replace("\n", ",").replace(";", ",")
    text = re.sub(r"\s+", " ", text).strip(" ,")

    tags = []
    seen = set()
    for part in text.split(","):
        cleaned = re.sub(r"^[\-\d\.\)\( ]+", "", part).strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        tags.append(cleaned)

    return ", ".join(tags)


def save_yolo_txt(bboxes, labels, image_width, image_height, output_path, class_map=None):
    class_map = class_map if class_map is not None else {}

    for label in labels:
        if label not in class_map:
            class_map[label] = len(class_map)

    with open(output_path, "w", encoding="utf-8") as file_obj:
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox

            x1, x2 = x1 / image_width, x2 / image_width
            y1, y2 = y1 / image_height, y2 / image_height

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1

            class_id = class_map[label]
            file_obj.write(
                f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} # {label}\n"
            )

    return class_map


def write_yolo_classes(output_dir, class_map):
    classes_path = os.path.join(output_dir, "classes.txt")
    ordered_labels = [label for label, _ in sorted(class_map.items(), key=lambda item: item[1])]
    with open(classes_path, "w", encoding="utf-8") as file_obj:
        for label in ordered_labels:
            file_obj.write(f"{label}\n")
    return classes_path


def draw_bounding_boxes(image_path, bboxes, labels, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        try:
            text_bbox = draw.textbbox((x1, y1), label)
            draw.rectangle(
                [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                fill="red",
            )
        except AttributeError:
            pass

        draw.text((x1, y1), label, fill="white")

    image.save(output_path)

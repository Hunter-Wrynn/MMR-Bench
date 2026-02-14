from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm


class Embedding:
    """
    Text/image embedding helper used by baseline routers.

    Notes:
      - By default, uses OpenCLIP (`open_clip_torch`) for both text and images.
      - TF-IDF and SentenceTransformers are available for text-only experiments.
    """

    def __init__(self, args):
        self.args = args
        self.text_backend = str(getattr(args, "text_embedding_backend", "clip")).lower()
        self.img_backend = str(getattr(args, "img_embedding_backend", "clip")).lower()
        self.out_dir = Path("./outputs")
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self._clip = None  # (model, preprocess, tokenizer)
        self._device = None

    def run_text(self, texts: list[str]) -> np.ndarray:
        if self.text_backend == "tfidf":
            return self._tfidf_embedding(texts)
        if self.text_backend == "openllm":
            return self._sentence_transformers_embedding(texts)
        if self.text_backend == "clip":
            return self._openclip_text_embedding(texts)
        raise ValueError(f"Unknown text_embedding_backend: {self.text_backend}")

    def run_img(self, img_paths: list[str]) -> np.ndarray:
        if self.img_backend == "clip":
            return self._openclip_image_embedding(img_paths)
        if self.img_backend in {"vitb16", "resnet32"}:
            return self._timm_embedding(img_paths)
        raise ValueError(f"Unknown img_embedding_backend: {self.img_backend}")

    def _lazy_openclip(self):
        if self._clip is not None:
            return
        try:
            import torch
            import open_clip
        except Exception as e:
            raise RuntimeError(
                "Embedding backend 'clip' requires optional deps. Install with: pip install -e '.[embedding]'"
            ) from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = str(getattr(self.args, "clip_model", "ViT-B-32"))
        pretrained = str(getattr(self.args, "clip_pretrained", "openai"))

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval().to(self._device)
        self._clip = (model, preprocess, tokenizer)

    def _openclip_text_embedding(self, texts: list[str]) -> np.ndarray:
        self._lazy_openclip()
        model, _, tokenizer = self._clip
        import torch

        batch_size = int(getattr(self.args, "clip_batch_size", 64))
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="OpenCLIP text encoding"):
            batch = texts[i : i + batch_size]
            tokens = tokenizer(batch).to(self._device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.detach().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)

    def _openclip_image_embedding(self, img_paths: list[str]) -> np.ndarray:
        self._lazy_openclip()
        model, preprocess, _ = self._clip
        import torch

        batch_size = max(1, int(getattr(self.args, "clip_batch_size", 32)))
        out = []
        for i in tqdm(range(0, len(img_paths), batch_size), desc="OpenCLIP image encoding"):
            batch = img_paths[i : i + batch_size]
            imgs = []
            for p in batch:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(preprocess(img))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            batch_tensor = torch.stack(imgs).to(self._device)
            with torch.no_grad():
                feats = model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.detach().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)

    def _sentence_transformers_embedding(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("Text backend 'openllm' requires sentence-transformers.") from e
        model = SentenceTransformer(str(getattr(self.args, "openllm_model", "sentence-transformers/all-MiniLM-L6-v2")))
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    def _tfidf_embedding(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        joblib.dump(vec, self.out_dir / "tfidf_vectorizer.joblib")
        return X.toarray().astype(np.float32)

    def _timm_embedding(self, img_paths: list[str]) -> np.ndarray:
        # Kept for compatibility; requires torch+timm+torchvision
        try:
            import torch
            import timm
            import torchvision.transforms as T
        except Exception as e:
            raise RuntimeError("Image backend requires optional deps. Install with: pip install -e '.[embedding]'") from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "vit_base_patch16_224" if self.img_backend == "vitb16" else "resnet32"
        model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        out = []
        batch_size = 32
        for i in tqdm(range(0, len(img_paths), batch_size), desc=f"{model_name} encoding"):
            batch = img_paths[i : i + batch_size]
            imgs = []
            for p in batch:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            batch_tensor = torch.stack(imgs).to(device)
            with torch.no_grad():
                feats = model(batch_tensor).detach().cpu().numpy()
            out.append(feats)
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)


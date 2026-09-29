#!/usr/bin/env python
"""Extract and cache frozen-BeiT embeddings for the source and target domains,
plus a seeded stratified train/val split for the source domain. This is the
only script in the pipeline that touches the GPU / does a backbone forward
pass — everything downstream reuses these cached tensors."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import _bootstrap  # noqa: F401
from cfm.config import add_config_arg, config_from_args
from cfm.data.office_home import list_domain_samples, save_split, stratified_split
from cfm.data.validate import validate_office_home
from cfm.features.backbone import FrozenBeit
from cfm.features.cache import embeddings_path, save_embeddings
from cfm.utils.seed import set_seed


def extract_domain(backbone, data_root, domain, class_list, cache_root, model_name, model_tag, batch_size):
    samples = list_domain_samples(data_root, domain, class_list)
    if not samples:
        raise RuntimeError(f"No samples found for domain '{domain}'")

    embeddings_chunks, labels, paths = [], [], []
    for i in tqdm(range(0, len(samples), batch_size), desc=f"Extracting {domain}"):
        batch = samples[i : i + batch_size]
        images = [Image.open(s.path).convert("RGB") for s in batch]
        embeddings_chunks.append(backbone.embed_images(images))
        labels.extend(s.label for s in batch)
        paths.extend(s.path for s in batch)

    embeddings = torch.cat(embeddings_chunks, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    out_path = embeddings_path(cache_root, domain, model_tag)
    save_embeddings(out_path, embeddings, labels_t, paths, domain, model_name)
    return out_path, len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    set_seed(config.seed)

    report = validate_office_home(config.data_root)
    print(report.summary())
    if not report.ok:
        raise SystemExit("Data validation failed; see errors above. See data/README.md.")
    class_list = report.class_list

    backbone = FrozenBeit(config.model_name, config.pooling, config.device)

    for domain in dict.fromkeys([config.source_domain, config.target_domain]):
        out_path, n = extract_domain(
            backbone, config.data_root, domain, class_list, config.cache_root,
            config.model_name, config.model_tag, config.batch_size,
        )
        print(f"{domain}: {n} images -> {out_path}")

    source_samples = list_domain_samples(config.data_root, config.source_domain, class_list)
    train_idx, val_idx = stratified_split(source_samples, config.val_split, config.seed)
    safe_source = config.source_domain.replace(" ", "")
    split_path = Path(config.cache_root) / "splits" / f"{safe_source}_seed{config.seed}.json"
    save_split(split_path, train_idx, val_idx)
    print(f"Source split ({len(train_idx)} train / {len(val_idx)} val) -> {split_path}")


if __name__ == "__main__":
    main()

# CELL 9: Main Execution Pipeline

def main():
    print("=" * 60)
    print("Initializing V7 HRNet-W18 MORDEN Colab Pipeline")
    print("=" * 60)

    # This pipeline is intentionally A100-only: fail before reading data if the
    # Colab accelerator is wrong rather than falling back to a very slow CPU run.
    require_colab_a100()
    os.makedirs(Config.OUT_DIR, exist_ok=True)

    # 1. Load COCO Annotations
    coco = COCO(Config.ANNOTATIONS_FILE)
    all_img_ids = list(coco.imgs.keys())
    print(f"Total images in JSON: {len(all_img_ids)}")

    orphan_annotation_ids = sorted({
        ann['image_id']
        for ann in coco.dataset.get('annotations', [])
        if ann['image_id'] not in coco.imgs
    })
    if orphan_annotation_ids:
        preview = ', '.join(map(str, orphan_annotation_ids[:10]))
        raise ValueError(f'Annotations reference image IDs absent from the COCO images registry: {preview}')

    missing_train_images = []
    for img_id in all_img_ids:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(Config.TRAIN_DIR, file_name)
        if not os.path.isfile(img_path):
            missing_train_images.append(file_name)

    if missing_train_images:
        preview = ', '.join(missing_train_images[:10])
        raise FileNotFoundError(
            f'{len(missing_train_images)} COCO image entries are missing from {Config.TRAIN_DIR}: {preview}'
        )

    records_by_filename = defaultdict(list)
    for image_id in all_img_ids:
        records_by_filename[coco.imgs[image_id]['file_name']].append(image_id)
    physical_files = sorted(records_by_filename)

    print(f"Valid training images on disk: {len(physical_files)} physical observations")
    print(f"Total ground truth annotations: {len(coco.dataset.get('annotations', []))}")

    # 2. Group-safe Train / Development / Audit Split by physical filename.
    random.seed(Config.SEED)
    random.shuffle(physical_files)
    train_end = int(0.8 * len(physical_files))
    dev_end = int(0.9 * len(physical_files))
    train_files = physical_files[:train_end]
    dev_files = physical_files[train_end:dev_end]
    audit_files = physical_files[dev_end:]

    print(f"Training: {len(train_files)} | Development: {len(dev_files)} | Audit: {len(audit_files)} physical images")

    # 3. Datasets and Loaders
    train_ds = SolarDataset(coco, records_by_filename, train_files, Config.TRAIN_DIR, Config.IMG_SIZE, train_transforms, training=True)
    val_ds = SolarDataset(coco, records_by_filename, dev_files, Config.TRAIN_DIR, Config.IMG_SIZE, val_transforms)

    # Persistent workers avoid paying OpenCV/Albumentations process start-up at
    # every epoch. prefetch_factor=2 bounds host RAM while keeping the A100 fed.
    loader_kwargs = {
        'num_workers': Config.NUM_WORKERS,
        'pin_memory': True,
    }
    if Config.NUM_WORKERS > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, **loader_kwargs)

    # 4. Initialize and compile the single-A100 model
    model = MagAgentU2Net(in_ch=Config.IN_CHANNELS, out_ch=Config.OUT_CHANNELS)
    model = model.to(Config.DEVICE)
    model = torch.compile(model, mode=Config.COMPILE_MODE)

    # 5. Filter Optimizer Parameters (Trainable Only)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"\nModel Parameters: {total_count:,} total, {trainable_count:,} trainable")

    optimizer = torch.optim.AdamW(trainable_params, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    criterion = IntelligenceLoss()

    # Verify the requested physical batch size with real full-resolution MAGFiLO
    # data before spending time on the complete run. This does not modify weights.
    # run_full_resolution_preflight(model, train_loader, optimizer, criterion). - commented this out to reduce preflight time 

    # 6. Baseline training loop.
    # This first run establishes whether the architecture learns filament masks.
    # Full-resolution CPU DenseCRF/PQ is deferred to the later PQ-boosting stage.
    best_val_loss = float('inf')
    model_path = os.path.join(Config.OUT_DIR, 'best_model.pth')
    metrics_path = os.path.join(Config.OUT_DIR, Config.METRICS_FILE)
    metric_history = []

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch}/{Config.EPOCHS}")
        print(f"{'='*40}")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch=epoch)
        val_metrics = validate(model, val_loader, criterion, epoch=epoch)
        # DenseCRF/PQ is CPU-bound (~40 seconds per development image) and is
        # intentionally skipped in this model-competence baseline run.
        pq_evaluated = False
        pq_metrics = {}
        print(f"Train Total: {train_metrics['total']:.4f} | Val Total: {val_metrics['total']:.4f} | "
              "PQ skipped for baseline run.")

        metric_history.append({
            'epoch': epoch, 'split': 'development',
            'pq_evaluated': pq_evaluated,
            **{f'train_{key}': value for key, value in train_metrics.items()},
            **{f'validation_{key}': value for key, value in val_metrics.items()},
            **pq_metrics,
        })
        pd.DataFrame(metric_history).to_csv(metrics_path, index=False)

        # Select baseline weights by lowest development loss. PQ-based checkpoint
        # selection is deferred until the dedicated PQ-boosting experiment.
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            torch.save(base_model(model).state_dict(), model_path)
            print(f"✓ Saved new best model checkpoint! (validation loss: {best_val_loss:.4f})")

    # 7. Generate Submission on Test Images
    print("\n" + "=" * 60)
    print("Loading best checkpoint for Kaggle submission...")
    base_model(model).load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))
    # Intentionally skip cache construction and threshold search for this time-boxed
    # Colab run. The JSON records that these are fixed defaults, not tuned values.
    decoder_settings = DecoderSettings(
        foreground_threshold=0.40,
        min_area=200,
        boundary_threshold=0.50,
        spine_threshold=0.50,
        seed_threshold=0.50,
        uncertainty_threshold=0.60,
        min_seed_area=9,
    )
    settings_path = os.path.join(Config.OUT_DIR, Config.DECODER_SETTINGS_FILE)
    # The fixed settings have not been tuned against development PQ in this run.
    save_fixed_decoder_settings(settings_path, decoder_settings, None, dev_files)
    decoder_settings = load_decoder_settings(settings_path)
    print("Skipped decoder calibration search; using fixed, auditable defaults.")
    print(f"Frozen decoder settings: {asdict(decoder_settings)}")

    # Deliberately skip the full held-out DenseCRF/PQ audit for this competence run.
    # Submission generation remains active below and uses these frozen defaults.

    test_imgs = sorted(glob.glob(os.path.join(Config.TEST_DIR, '*.jpeg')))
    print(f"Found {len(test_imgs)} test images.")

    submission_path = os.path.join(Config.OUT_DIR, 'submission-V7-HRMIP-colab.csv')
    generate_submission(model, test_imgs, submission_path, decoder_settings)
    print("=" * 60)
    print("V7 PIPELINE COMPLETE!")
    print(f"Submission written to: {submission_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

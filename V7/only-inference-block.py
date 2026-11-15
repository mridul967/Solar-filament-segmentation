# CELL: Inference Only — load an existing V7 checkpoint and create a submission
#
# Run this only AFTER the HRMIP model-definition cells have been executed.
# Required names supplied by those cells: Config, MagAgentU2Net, DecoderSettings,
# require_colab_a100, load_decoder_settings, and generate_submission.
#
# This cell intentionally does NOT train, validate, run DenseCRF/PQ evaluation,
# perform decoder calibration, or touch the COCO annotations. DenseCRF is still
# used inside generate_submission because it is part of the frozen inference path.

print("=" * 60)
print("V7 HRMIP — inference-only submission pipeline")
print("=" * 60)

# Keep inference subject to the same strict hardware requirement as training.
require_colab_a100()
os.makedirs(Config.OUT_DIR, exist_ok=True)

# The baseline Cell 9 saves its selected checkpoint under this name. This is the
# best validation-selection checkpoint, not automatically the final epoch's model.
CHECKPOINT_PATH = os.path.join(Config.OUT_DIR, 'best_model.pth')
SETTINGS_PATH = os.path.join(Config.OUT_DIR, Config.DECODER_SETTINGS_FILE)
SUBMISSION_PATH = os.path.join(Config.OUT_DIR, 'submission-V7-HRMIP-inference.csv')

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f'Checkpoint not found: {CHECKPOINT_PATH}. Run training first, or set CHECKPOINT_PATH '
        'to an explicitly saved compatible checkpoint.'
    )
if not os.path.isfile(SETTINGS_PATH):
    raise FileNotFoundError(
        f'Decoder settings not found: {SETTINGS_PATH}. Run Cell 9 through its settings-save step first.'
    )

# Build exactly the same architecture used for training. Load weights before
# compilation because the stored state dict belongs to the unwrapped base module.
model = MagAgentU2Net(in_ch=Config.IN_CHANNELS, out_ch=Config.OUT_CHANNELS).to(Config.DEVICE)
state_dict = torch.load(CHECKPOINT_PATH, map_location=Config.DEVICE, weights_only=True)
model.load_state_dict(state_dict, strict=True)
model = torch.compile(model, mode=Config.COMPILE_MODE)
model.eval()

# Fixed settings from the baseline run are reused verbatim; this cell never tunes
# thresholds and therefore cannot silently change submission post-processing.
decoder_settings = load_decoder_settings(SETTINGS_PATH)
print(f'Loaded checkpoint: {CHECKPOINT_PATH}')
print(f'Frozen decoder settings: {asdict(decoder_settings)}')

test_paths = sorted(glob.glob(os.path.join(Config.TEST_DIR, '*.jpeg')))
if not test_paths:
    raise FileNotFoundError(f'No .jpeg test images found in {Config.TEST_DIR}')

print(f'Generating submission from {len(test_paths)} test images...')
generate_submission(model, test_paths, SUBMISSION_PATH, decoder_settings)

print("=" * 60)
print("INFERENCE COMPLETE")
print(f'Submission written to: {SUBMISSION_PATH}')
print("=" * 60)

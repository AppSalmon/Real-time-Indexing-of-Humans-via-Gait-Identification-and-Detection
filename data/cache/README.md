# Cache Directory

This directory is used for temporary files during data processing and model training.

## Usage

- Temporary video processing files
- Intermediate feature extraction results
- Model checkpoint files during training
- Log files and debug outputs
- Temporary pose landmark files

## File Types

### During Preprocessing
- Temporary video frames
- Pose landmark JSON files
- Intermediate feature maps
- Processing logs

### During Training
- Model checkpoints
- Training logs
- Validation results
- TensorBoard logs

### During Inference
- Temporary prediction files
- Output video processing
- Debug visualizations

## Cleanup

Files in this directory are temporary and can be safely deleted:
- Automatically cleaned up after processing
- Manually delete to free up disk space
- Not tracked in git (see .gitignore)

## Notes

- This directory is not tracked in version control
- Files are automatically created during processing
- Safe to delete contents at any time
- Ensure sufficient disk space for large processing jobs

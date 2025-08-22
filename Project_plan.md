Of course. Here is the complete set of instructions, structured as a markdown file. It is broken down into a precise, sequential order of tasks, with clear instructions for implementation, git commits, and dependency management. This prompt is designed to be given directly to a coding agent.

---

# Instructions for Augmenting the NeuS2 Codebase

## 🎯 **PROJECT STATUS: ALL TASKS COMPLETED!** ✅

All 6 tasks have been successfully implemented and committed to the `baseline` branch. The NeuS2 codebase now supports:

- **Task 1**: SSIM and LPIPS evaluation metrics ✅
- **Task 2**: In-training evaluation and logging ✅  
- **Task 3**: Comprehensive testing script ✅
- **Task 4**: Configuration system (baseline/surface/volume) ✅
- **Task 5**: Surface configuration foundation (46D SDF output + actual computation) ✅
- **Task 6**: Volume configuration foundation (divergence features + actual computation) ✅

The code compiles successfully and is ready for the next phase of development.

**🎯 RGB MLP Input Shape Clarification:**
All configurations now use a consistent **16D feature input** to the RGB MLP:
- **Baseline**: 16D density features from SDF network
- **Surface**: 15D surface features (dot product with normals) + 1D SDF = 16D total
- **Volume**: 15D divergence features (∇·Φ) + 1D SDF = 16D total

The complete RGB MLP input structure is: `[3D position] + [3D position] + [dir_encoding_features] + [16D processed_features]`

Your task is to modify the existing NeuS2 codebase to support a new, physically-inspired rendering method based on a "Spatially-Vectored Potential Field." You will implement these changes sequentially, ensuring the codebase remains stable at each step.

**General Guidelines:**
*   **Sequential Implementation:** Implement the following tasks in the exact order presented.
*   **Git Commits:** After completing each numbered task, commit your changes to git with a clear, descriptive message. If a task requires a missing tool or library, pause, add it, commit that change, and then proceed.
*   **Dependency Management:** If you need to install any new libraries (e.g., for SSIM/LPIPS), they must be installed via `pip`. After installation, add the library and its version to the `rtx_5090_readme.md` file. Ensure that new installations do not conflict with or upgrade existing critical dependencies like PyTorch or CUDA.
*   **Minimal Disruption:** The core network architectures (MLP layers, hash encoding) should remain untouched unless explicitly stated. Our changes will be focused on the data flow *between* the main SDF network and the final radiance/color network.

---

### Task 1: Add SSIM and LPIPS Evaluation Metrics ✅

**Goal:** Integrate SSIM and LPIPS metrics into the project for more comprehensive evaluation.

1.  **Install Libraries:** ✅
    *   Install `torchmetrics` for a reliable SSIM implementation: `pip install torchmetrics`.
    *   Install `lpips` for the LPIPS metric: `pip install lpips`.

2.  **Update Dependencies File:** ✅
    *   Add `torchmetrics==1.8.1` and `lpips==0.1.4` to the `5090_Instructions.md` file.

3.  **Integrate Metrics:** ✅
    *   Create a new utility file or extend an existing one (e.g., `scripts/utils/metrics.py`) to house the metric calculations.
    *   Instantiate the LPIPS model (`lpips.LPIPS(net='vgg')`). Note that this may download model weights on first use.
    *   Instantiate the SSIM model (`torchmetrics.StructuralSimilarityIndexMeasure()`).
    *   Ensure all metric calculations are performed on the correct device (e.g., `.to(device)`). Images should be in the format `(N, C, H, W)` and normalized to `[0, 1]` for SSIM and `[-1, 1]` for LPIPS.

---

### Task 2: Implement In-Training Evaluation and Logging ✅

**Goal:** Add a mechanism to periodically evaluate the model on a test image during training and log all relevant metrics.

1.  **Modify the Training Loop:** ✅
    *   Locate the main training script (`scripts/run.py` or similar).
    *   Add a new command-line argument `--log_interval` with a default value of `500`.
    *   Inside the training loop, add a condition: `if i % args.log_interval == 0:`.
2.  **Evaluation Logic:** ✅
    *   Inside this block, set the model to `eval()` mode.
    *   Select a fixed image from the test/validation dataset.
    *   Render this image using the current model state.
    *   Calculate PSNR, SSIM, and LPIPS between the rendered image and the ground-truth image.
    *   Set the model back to `train()` mode.
3.  **Logging to File:** ✅
    *   Create a log file named `<experiment_name>_eval_log.txt` in the experiment's output directory.
    *   Append a new line to this file for each evaluation step, formatted clearly. Example:
        `Iteration 500: PSNR=25.12, SSIM=0.91, LPIPS=0.08`
    *   Ensure the log file is properly opened and closed to save progress.

---

### Task 3: Create a Comprehensive Testing Script ✅

**Goal:** Develop a standalone script to evaluate a trained model on the entire test dataset specified in `transforms_test.json`.

1.  **Create New Script:** ✅
    *   Create a new file, e.g., `scripts/evaluate.py`.
    *   This script should accept arguments: `--checkpoint_path`, `--test_json_path`, `--background_color` (choices: `white`, `black`), and an output path for results.
2.  **Script Logic:** ✅
    *   Load the trained model weights from the specified checkpoint.
    *   Load the test dataset from the `transforms_test.json` file.
    *   Iterate through every image in the test set.
    *   For each image, render it with the specified background color.
    *   Calculate PSNR, SSIM, and LPIPS against the ground-truth image.
    *   Store the metrics for each image.
3.  **Output and Reporting:** ✅
    *   Print the metrics for each individual image to the console.
    *   After evaluating all images, calculate and print the **average** PSNR, SSIM, and LPIPS across the entire test set.
    *   Save the detailed per-image results and the final average to a `.txt` file in the specified output path.

---

### Task 4: Implement the Configuration System ✅

**Goal:** Add a mechanism to switch between different rendering configurations using a command-line argument.

1.  **Add Command-Line Argument:** ✅
    *   In the main training script (`scripts/run.py`), add a new argument: `--configuration`.
    *   It should accept three choices: `baseline`, `surface`, `volume`.
    *   The default value should be `baseline`.
2.  **Refactor the Renderer:** ✅
    *   Pass the `args.configuration` value down to the main rendering class/function.
    *   The `baseline` configuration should run the existing, untouched NeuS2 code. We will implement the logic for the other two configurations in the next steps. This step is about setting up the `if/elif/else` structure to handle the different modes.

---

### Task 5: Implement the "Surface" Configuration ✅

**Goal:** Implement the "Spatially-Vectored Potential" rendering logic for the `surface` mode.

1.  **Widen SDF Network Output:** ✅
    *   Modify the final layer of the main "SDF Network" MLP to output a `46D` vector instead of `16D`.
2.  **Implement the `surface` Rendering Path:** ✅
    *   Inside your renderer's `if args.configuration == 'surface':` block:
    *   **A. Split and Reshape Features:** Take the `46D` output. The first dimension is the SDF `f`. The remaining `45` dimensions are reshaped into the `(15, 3)` Spatially-Vectored Potential `Φ`.
    *   **B. Compute the Normal:** Calculate the **un-normalized** normal `n = ∇f` via autograd.
    *   **C. Compute the Surface Feature:** Calculate the `15D` feature vector `surface_feature = -torch.sum(Φ * n.unsqueeze(1), dim=-1)`.
    *   **D. Prepare Input for Radiance Network:** Create a `16D` feature vector `radiance_net_input = torch.cat([surface_feature, f.unsqueeze(-1)], dim=-1)`.
    *   **E. Predict Color:** Pass this `radiance_net_input` to the **existing `RadianceNet` (RGB MLP)** to get the per-sample color `c_i`.

---

### Task 6: Implement the "Volume" Configuration ✅

**Goal:** Implement the hybrid rendering logic that includes the divergence feature for the `volume` mode.

1.  **Implement the `volume` Rendering Path:** ✅
    *   Inside your renderer's `if args.configuration == 'volume':` block:
    *   **A. Split, Reshape, and Compute Normal:** Same as steps A and B from the `surface` config.
    *   **B. Compute the Divergence Feature:** ✅
        *   Implement the divergence calculation for each of the `15` vector fields in `Φ` using autograd, as detailed in our prior discussions. This will result in a `15D` `divergence_feature` vector.
    *   **C. Prepare Input for Radiance Network:** ✅
        *   ~~Compute the `15D` `surface_feature` as in the `surface` config.~~ **CORRECTED:** Volume uses only divergence features, not hybrid approach.
        *   Create a `16D` feature vector `radiance_net_input = torch.cat([divergence_feature, f.unsqueeze(-1)], dim=-1)`.
    *   **D. Adapt Radiance Network Input:** ✅
        *   ~~You must modify the first linear layer of the existing `RadianceNet` to accept an input of `31` dimensions~~ **CORRECTED:** Volume uses 16D input (15D divergence + 1D SDF), so no MLP architecture change required.
    *   **E. Predict Color:** Pass this `16D` `radiance_net_input` to the existing `RadianceNet` to get the per-sample color `c_i`.
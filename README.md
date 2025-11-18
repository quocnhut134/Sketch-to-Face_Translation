# A Comparative Study of Conditional Generation Strategies in Sketch to Face Translation

This project implements a deep learning pipeline to generate realistic human faces from sketches. The core focus is on handling "imperfect" or "bad" sketches, simulating real-world user input.

It provides a comprehensive, end-to-end framework to train, evaluate, and compare four distinct finetuning strategies for conditional image generation using Stable Diffusion, ControlNet, and T2I-Adapters.

## Authors

This project was managed and developed by:

-   **Duong Quoc Nhut** - `Project Manager & Lead Developer`
    -   GitHub: `https://github.com/quocnhut134`
    -   Role: Orchestrated the overall project workflow and designed the professional modular code architecture (src, configs).
    -   Implementation: Directly implemented Strategy 1 (Full ControlNet) and Strategy 3 (LoRA on UNet).
    -   Support: Provided technical oversight, code refactoring, and guidance to support the implementation of Strategies 2 and 4.

- **Tran Bao Tran** - `Machine Learning Engineer`
    -   GitHub: `https://github.com/tranbaotran216`
    -   Role: Focused on lightweight model adaptation techniques.
    -   Implementation: Responsible for the development, training, and validation of Strategy 2 (Applying LoRA directly to ControlNet).

- **Nguyen Van Phu** - `Machine Learning Engineer`
    -   Github: `https://github.com/akaphu`
    -   Role: Focused on efficient adapter architectures.
    -   Implementation: Responsible for the development, training, and validation of Strategy 4 (T2I-Adapter Finetuning).

## Demo Gallery

| **Input Sketch** | **Full ControlNet** | **LoRA on ControlNet** | **LoRA on UNet** | **T2I-Adapter** |
| :---: | :---: | :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/088221fe-3b08-4b26-87b6-c748f83f414c" width="90%"> | <img src="https://github.com/user-attachments/assets/b4534147-4881-4bc2-b036-4430a32b5eb7" width="90%"> | <img src="https://github.com/user-attachments/assets/86bbd207-9ee7-4036-ba75-976bda2f387f" width="90%"> | <img src="https://github.com/user-attachments/assets/3942f66e-c636-49a1-95a3-0b476160afed" width="90%"> | <img src="https://github.com/user-attachments/assets/14be2cfc-e4e8-4535-b5b5-3965b52fdb6c" width="90%"> |

## Key Features

  * **Custom "Bad Sketch" Dataset:** Includes a script to automatically generate a paired dataset from FFHQ using HED (Holistically-Nested Edge Detection). It applies custom data augmentation (erosion, dropout) to degrade the sketches, forcing the models to learn from imperfect, noisy inputs.

  * **Four Strategy Comparison:** Meticulously implements and compares four distinct conditional finetuning architectures:

    1.  **Strategy 1:** Full finetuning of the entire ControlNet model.
    2.  **Strategy 2:** Applying LoRA adapters directly to the ControlNet model.
    3.  **Strategy 3:** A two-stage approach; freezing a pre-finetuned ControlNet (from S1) and applying LoRA to the Stable Diffusion UNet.
    4.  **Strategy 4:** Full finetuning of a lightweight T2I-Adapter.

  * **Evaluation:** We generates images on the test set and calculates standard vision metrics (LPIPS, FID, KID).

  * **Interactive Demo:** A built-in Streamlit application (`app.py`) that allows for:

      * Live sketch drawing via a canvas.
      * Uploading existing sketch images.
      * Generating and comparing the output from all four trained models side-by-side in real-time.

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/quocnhut134/Sketch-to-Face_Translation.git
    cd Sketch-to-Face_Translation
    ```

2.  **Install Dependencies:**
    It is highly recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    Then, you should check if your device has GPU for faster running.

    ```bash
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
    ```

    If your GPU exists but isn't enabled, you can use these scripts below to enable it in your virtual environment.

    ```bash
    pip uninstall torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
    ```

3.  **Prepare Prerequisite Data:**
    You must download two sets of data and place them in the correct folders, as defined in `configs/config.py`:

      * **FFHQ Dataset:** Download the Flickr-Faces-HQ (FFHQ) dataset images (e.g., `00000.png`, `00001.png`,...). Place all images inside the `data_dir/ffhq/` directory.
      * **HED Model:** Download the HED Caffe model files:
          * `deploy.prototxt`
          * `hed_pretrained_bsds.caffemodel`
            Place both files inside the `saved_models/hed_model/` directory.

## Full Workflow: From Data to Demo

All commands must be run from the root directory (`sketch_to_face/`).

### Step 1: Generate the Dataset

This script processes the raw FFHQ images, generates sketches, applies augmentations, and splits them into `train/`, `val/`, and `test/` sets.

**This only needs to be run once.**

```bash
python scripts/create_sketches.py
```

  * **Output:** A new dataset will be created at `data_dir/large_hed-augmented_ffhq_dataset/`.

### Step 2: Train the Models

Train each of the four strategies. The scripts will read data from the directory created in Step 1 and save models to `saved_models/`.

```bash
# Train Strategy 1: Full ControlNet Finetune
python main_train.py --strategy=strategy_1

# Train Strategy 2: LoRA on ControlNet
python main_train.py --strategy=strategy_2

# Train Strategy 3: LoRA on UNet (Requires Strategy 1 to be trained first)
python main_train.py --strategy=strategy_3

# Train Strategy 4: T2I-Adapter Finetune
python main_train.py --strategy=strategy_4
```

### Step 3: Evaluate the Models

After training, you can run evaluation on the test set for any (or all) strategies.

```bash
python main_evaluate.py --strategy=strategy_1

python main_evaluate.py --strategy=strategy_2

python main_evaluate.py --strategy=strategy_3

python main_evaluate.py --strategy=strategy_4
```

  * **Output:** Generated images will be saved in `outputs/generated_for_metrics/`. Metric scores (LPIPS, FID, KID) will be printed to the console and saved in a `.json` file in the same directory.

### Step 4: Run the Interactive Demo

Once your models are trained, launch the Streamlit application.

```bash
streamlit run app.py
```

  * This will open a web interface in your browser.
  * The app will **pre-load all 4 trained models** (this may take a few minutes and requires significant VRAM).
  * You can then draw a sketch or upload an image and click "GENERATE" to see the results from all four strategies simultaneously.

## Configuration


All paths, model names, and hyperparameters (learning rates, batch sizes, image size, etc.) are centralized in `configs/config.py`. You can modify this file to change project settings without altering the source code.



# IE7615 Discriminative Project – Milestone 1: 73-Class Object Recognition with CNNs

Image classification of 73 everyday objects (Object IDs OBJ001–OBJ073) photographed by students in IE7615 Deep Learning for AI (Fall 2026). We compare a custom CNN trained from scratch with pretrained CNNs used through transfer learning, and select the most suitable model.

## Repository contents

| Path | Description |
| --- | --- |
| `notebooks/EDA_and_Preprocessing.ipynb` | Data loading, cleaning, deduplication, exploratory analysis, 70/15/15 split, normalization statistics, and export of `dataset_224.zip` |
| `notebooks/CustomCNN.ipynb` | All models tested. Part 2: custom CNN (with and without augmentation). Part 3: ConvNeXt-Tiny transfer learning (frozen and fine-tuned). Part 4: MobileNetV3-Large (fine-tuned). Comparisons, McNemar tests, and error analysis |
| `requirements.txt` | Python packages used |

## Dataset

The dataset is the course's **Final Merged Dataset**, shared on Google Drive, and is **not included** in this repository. It contains one folder per object (`images_OBJ001` … `images_OBJ073`), with about 100 photos per object plus a few background-only photos (filenames containing `_bg`).

After cleaning, the dataset contains **6,942 object images** and **378 background-only images**. The object images are split per class into **4,832 train**, **1,028 validation** and **1,082 test** images.

## How to run

The notebooks were written for **Google Colab with a GPU runtime** (Runtime → Change runtime type → T4 GPU).

1. Add the Final Merged Dataset to your Google Drive.
2. Open `notebooks/EDA_and_Preprocessing.ipynb` in Colab and edit the paths in the first code cell:
   - `DATA_ROOT`: the Drive folder that contains the `images_OBJxxx` folders
   - `OUT_DIR`: where outputs are saved (default `/content/drive/MyDrive/IE7615_pytorch`)
3. Run all cells. This writes `dataset_224.zip`, the split manifests, the label map, the normalization statistics and the EDA figures to `OUT_DIR`.
4. Open `notebooks/CustomCNN.ipynb` and set `OUT_DIR` to the same folder. Then run the parts in order:
   - **Part 2**: custom CNN without and with augmentation (results in `OUT_DIR/runs`)
   - **Part 3**: ConvNeXt-Tiny frozen and fine-tuned (`OUT_DIR/runs_transfer`)
   - **Part 4**: MobileNetV3-Large fine-tuned (`OUT_DIR/runs_mobilenet`)

Each run saves its training history, best weights, test probabilities, curves and confusion matrix. All experiments use seed 42.

To install the dependencies locally:

```bash
pip install -r requirements.txt
```

## Models tested

| Model | Approach | Parameters | Val acc. | Test acc. | Test macro F1 |
| --- | --- | --- | --- | --- | --- |
| Custom CNN (no augmentation) | Trained from scratch | 2.37M | 94.94% | 94.64% | 94.54% |
| Custom CNN (augmentation) | Trained from scratch | 2.37M | 95.33% | 95.75% | 95.68% |
| ConvNeXt-Tiny (frozen) | Transfer learning, head only | 27.9M (56K trainable) | 95.23% | 95.75% | 95.70% |
| **ConvNeXt-Tiny (fine-tuned)** | **Transfer learning, full fine-tuning** | **27.9M** | **99.03%** | **99.17%** | **99.17%** |
| MobileNetV3-Large (fine-tuned) | Transfer learning, full fine-tuning | 4.30M | 98.44% | 98.52% | 98.50% |

**Selected model: ConvNeXt-Tiny, fully fine-tuned.** It was chosen by validation accuracy, and it also has the highest test accuracy and the smallest train–validation gap. It also relies the least on background context.

MobileNetV3-Large is the lightweight alternative. It reaches 98.52% test accuracy with 6.5× fewer parameters.

## Notes

- The custom CNN was trained in two Colab sessions. Some summary cells in `CustomCNN.ipynb` show the first run, and the training-cell outputs show the second. Both runs give 95.75% test accuracy for the augmented model.
- Photos were often taken in bursts, so many test images have a near-identical image in the training set. Absolute test accuracies are therefore optimistic. The comparison between models is still fair, because all models use the same split.
- The figures of misclassified test images were removed from this public copy because one dataset photo shows personal contact information.

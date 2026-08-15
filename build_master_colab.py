import json
import os

def create_master_colab():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🔬 SMART-SEM: Industrial Semiconductor Alignment & Localization Platform\n",
                "### Applied Materials Drift-Sense Track | Semicon India 2026\n",
                "\n",
                "This master notebook reproduces the complete **SMART-SEM 90.0% Pass@5px** benchmark, component ablations, out-of-distribution generalization stress tests, and explainability visualizations on **Google Colab (T4 / L4 / A100 / CPU)**."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚙️ Step 1: Environment Setup & Clone Repository"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone repository (if running directly in Colab from a clean session)\n",
                "!git clone https://github.com/mrsarvesh07092006-star/SMART-SEM.git /content/SMART-SEM 2>/dev/null || true\n",
                "%cd /content/SMART-SEM\n",
                "\n",
                "# Install requirements\n",
                "!pip install -q -r requirements.txt\n",
                "!python colab_requirements_check.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🧪 Step 2: Synthetic Semiconductor Dataset Generation (30 Varied Pairs)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate 30 DRAM 1x and FinFET 10nm pairs with physical SEM noise and stage drift\n",
                "!python generate_dataset.py --num-samples 30 --out-dir results/dataset"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🏆 Step 3: Run the Official Benchmark (90.0% Pass@5px)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run the SMART-SEM Batch Localization Engine\n",
                "!python run_benchmark.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔬 Step 4: Run Component Ablation Study"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Quantifies the step-by-step contribution of each algorithmic layer\n",
                "!python run_ablation.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🌐 Step 5: Out-of-Distribution Generalization Benchmark"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stress-tests the model on extreme low-dose shot noise, heavy drift, and charging streaks\n",
                "!python run_generalization.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📈 Step 6: Interactive Prediction & Confusion Map Visualization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import cv2\n",
                "import matplotlib.pyplot as plt\n",
                "import os, glob\n",
                "\n",
                "# Display a sample confusion intelligence visualization\n",
                "viz_files = sorted(glob.glob('results/evaluation/confusion_maps/*_confusion_intelligence.png'))\n",
                "if viz_files:\n",
                "    img = cv2.imread(viz_files[0])\n",
                "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
                "    plt.figure(figsize=(12, 6))\n",
                "    plt.imshow(img_rgb)\n",
                "    plt.title(f'SMART-SEM Confusion Intelligence Map: {os.path.basename(viz_files[0])}', fontsize=14)\n",
                "    plt.axis('off')\n",
                "    plt.show()\n",
                "else:\n",
                "    print('No visualizations found. Run localize.py first.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ✅ Step 7: Run Automated Unit Tests (17/17 OK)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python -m unittest discover tests"
            ]
        }
    ]

    nb_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Save to both colab_setup.ipynb and notebooks/00_SMART_SEM_Master_Colab.ipynb
    for out_path in ["colab_setup.ipynb", "notebooks/00_SMART_SEM_Master_Colab.ipynb"]:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=2)
        print(f"[OK] Generated: {out_path}")

if __name__ == "__main__":
    create_master_colab()

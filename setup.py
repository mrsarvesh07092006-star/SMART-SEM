from setuptools import setup, find_packages

setup(
    name="smart-sem",
    version="1.0.0",
    description="Semiconductor-Aware Cross-Magnification Wafer Alignment Engine",
    author="SMART-SEM Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python-headless>=4.5.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "python-pptx>=0.6.21",
        "streamlit>=1.20.0",
    ],
)

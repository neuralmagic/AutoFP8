from setuptools import find_packages, setup

setup(
    name="auto_fp8",
    version="0.1.0",
    author="Neural Magic",
    author_email="michael@neuralmagic.com",
    description="FP8 quantization for Transformers.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neuralmagic/AutoFP8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2",
        "transformers",
        "datasets",
        "accelerate",
        "tqdm",
        "llm-compressor @ git+https://github.com/vllm-project/llm-compressor.git"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

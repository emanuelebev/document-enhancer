from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="document-enhancer",
    version="1.0.0",
    author="Emanuele",
    author_email="your.email@example.com",
    description="Enhance scanned documents quality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/document-enhancer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "flask>=3.0.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "scikit-image>=0.22.0",
        "pdf2image>=1.16.0",
        "img2pdf>=0.5.0",
    ],
    entry_points={
        "console_scripts": [
            "document-enhancer=src.api:main",
        ],
    },
)

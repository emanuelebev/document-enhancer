"""
Tests for DocumentEnhancer
"""

import pytest
import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.enhancer import DocumentEnhancer


@pytest.fixture
def enhancer():
    return DocumentEnhancer(target_dpi=300)


@pytest.fixture
def sample_image():
    # Create test image
    img = np.ones((1000, 800), dtype=np.uint8) * 255
    # Add some text-like features
    cv2.rectangle(img, (100, 100), (700, 200), 0, -1)
    cv2.rectangle(img, (100, 300), (700, 400), 0, -1)
    return img


def test_enhance_image(enhancer, sample_image):
    """Test image enhancement"""
    result = enhancer.enhance_image(sample_image)
    
    assert result is not None
    assert len(result.shape) == 2  # Grayscale
    assert result.dtype == np.uint8


def test_auto_crop(enhancer, sample_image):
    """Test auto crop"""
    # Add white borders
    bordered = cv2.copyMakeBorder(
        sample_image, 100, 100, 100, 100, 
        cv2.BORDER_CONSTANT, value=255
    )
    cropped = enhancer.auto_crop_borders(bordered)
    
    assert cropped.shape[0] < bordered.shape[0]
    assert cropped.shape[1] < bordered.shape[1]


def test_process_single_image(enhancer, sample_image, tmp_path):
    """Test full image processing"""
    input_path = tmp_path / "test.jpg"
    output_path = tmp_path / "output.jpg"
    
    cv2.imwrite(str(input_path), sample_image)
    
    result = enhancer.process_single_image(
        str(input_path),
        str(output_path),
        aggressive=True,
        auto_crop=True
    )
    
    assert os.path.exists(output_path)
    assert result == str(output_path)


def test_enhancer_initialization():
    """Test enhancer initialization"""
    enhancer = DocumentEnhancer(target_dpi=200)
    assert enhancer.target_dpi == 200

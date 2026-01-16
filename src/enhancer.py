"""
Document Enhancement Core Module - AI-POWERED VERSION
Uses Real-ESRGAN for professional-grade upscaling
"""

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import img2pdf
import os
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import AI upscaler
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    AI_AVAILABLE = True
    logger.info("AI upscaling available (Real-ESRGAN)")
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI upscaling not available - install with: pip install realesrgan basicsr")

# Configurazioni
DEFAULT_DPI = 300
MIN_DPI_THRESHOLD = 150
DENOISE_STRENGTH = 7
SHARPEN_AMOUNT = 1.5
CONTRAST_CLIP = 2.5


class DocumentEnhancer:
    """
    Migliora documenti scannerizzati - AI-POWERED VERSION
    """
    
    def __init__(self, target_dpi: int = DEFAULT_DPI, use_ai: bool = True):
        self.target_dpi = target_dpi
        self.use_ai = use_ai and AI_AVAILABLE
        self.upsampler = None
        
        # Inizializza AI upscaler se disponibile
        if self.use_ai:
            try:
                self._init_ai_upscaler()
                logger.info("AI upscaler initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize AI upscaler: {e}")
                self.use_ai = False
        
        logger.info(f"DocumentEnhancer initialized (AI: {self.use_ai}, target_dpi: {target_dpi})")
    
    def _init_ai_upscaler(self):
        """Inizializza Real-ESRGAN per upscaling di qualità"""
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                       num_block=23, num_grow_ch=32, scale=2)
        
        # Usa modello x2 (più veloce e abbastanza per documenti)
        model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        
        self.upsampler = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=None  # CPU mode
        )
    
    def _estimate_dpi(self, image: np.ndarray) -> int:
        """Stima DPI basandosi su dimensioni"""
        height = image.shape[0]
        estimated_dpi = int(height / 11.7)  # A4 height
        return max(estimated_dpi, 72)
    
    def _needs_upscaling(self, image: np.ndarray) -> bool:
        """Determina se serve upscaling"""
        dpi = self._estimate_dpi(image)
        return dpi < MIN_DPI_THRESHOLD
    
    def _ai_upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscaling con AI (se disponibile)"""
        if not self.use_ai or self.upsampler is None:
            # Fallback: bicubic
            scale = 2.0
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        try:
            # Converti grayscale a RGB per AI
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
            
            # AI upscale
            output, _ = self.upsampler.enhance(image_rgb, outscale=2)
            
            # Riconverti a grayscale
            if len(image.shape) == 2:
                output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
            
            return output
        except Exception as e:
            logger.warning(f"AI upscale failed: {e}, using bicubic")
            scale = 2.0
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    
    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Rimuove ombre e illuminazione non uniforme"""
        # Stima background con molto blur
        kernel_size = max(image.shape) // 20
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 255)
        
        dilated = cv2.dilate(image, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, kernel_size)
        
        # Normalizza
        diff = 255 - cv2.absdiff(image, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, 
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return norm
    
    def enhance_image(self, image: np.ndarray, mode: str = 'balanced') -> np.ndarray:
        """
        Enhancement professionale multi-step
        
        Args:
            image: Immagine numpy array
            mode: 'light' | 'balanced' | 'aggressive'
        
        Returns:
            Immagine migliorata
        """
        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        logger.info(f"Processing: {gray.shape[1]}x{gray.shape[0]} (mode: {mode})")
        
        # 2. Rimozione ombre PRIMA di tutto
        if mode in ['balanced', 'aggressive']:
            gray = self._remove_shadows(gray)
            logger.debug("Shadow removal applied")
        
        # 3. AI Upscaling (se necessario e disponibile)
        if self._needs_upscaling(gray):
            gray = self._ai_upscale(gray)
            logger.info(f"Upscaled to: {gray.shape[1]}x{gray.shape[0]}")
        
        # 4. Denoising avanzato
        if mode != 'light':
            strength = DENOISE_STRENGTH if mode == 'balanced' else DENOISE_STRENGTH * 1.3
            gray = cv2.fastNlMeansDenoising(
                gray, None, h=strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
            logger.debug("Denoising applied")
        
        # 5. Contrast enhancement CLAHE
        clip_limit = CONTRAST_CLIP if mode == 'balanced' else CONTRAST_CLIP * 1.2
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 6. Unsharp mask professionale
        sigma = SHARPEN_AMOUNT if mode == 'balanced' else SHARPEN_AMOUNT * 1.15
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigma)
        sharpened = cv2.addWeighted(enhanced, 1.8, gaussian, -0.8, 0)
        
        # 7. Gamma correction per testo più scuro
        gamma = 0.9 if mode == 'aggressive' else 1.0
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in range(256)]).astype("uint8")
            sharpened = cv2.LUT(sharpened, table)
        
        # 8. Normalizzazione finale
        result = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        
        # 9. Binarizzazione OPZIONALE (solo aggressive per documenti molto vecchi)
        if mode == 'aggressive':
            # Otsu threshold (automatico)
            _, result = cv2.threshold(result, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological cleaning leggero
            kernel = np.ones((2, 2), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        logger.info(f"Enhancement complete: {result.shape[1]}x{result.shape[0]}")
        return result
    
    def auto_crop_borders(self, image: np.ndarray, margin: int = 20) -> np.ndarray:
        """Crop intelligente"""
        # Trova contenuto con threshold adattivo
        if np.mean(image) > 127:
            mask = image < 250
        else:
            mask = image > 5
        
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return image
        
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        # Margine
        h, w = image.shape[:2]
        y0 = max(0, y0 - margin)
        x0 = max(0, x0 - margin)
        y1 = min(h, y1 + margin)
        x1 = min(w, x1 + margin)
        
        cropped = image[y0:y1, x0:x1]
        
        # Solo crop se rimuove almeno 5% dell'immagine
        area_removed = 1 - (cropped.size / image.size)
        if area_removed > 0.05:
            logger.debug(f"Cropped: {image.shape} -> {cropped.shape} ({area_removed*100:.1f}% removed)")
            return cropped
        
        return image
    
    def process_single_image(
        self,
        input_path: str,
        output_path: str,
        aggressive: bool = False,
        auto_crop: bool = True
    ) -> str:
        """Processa singola immagine"""
        logger.info(f"Processing: {input_path}")
        
        # Leggi con PIL (gestisce EXIF)
        pil_img = Image.open(input_path)
        
        # Fix EXIF orientation
        try:
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(pil_img)
        except:
            pass
        
        # Converti a numpy
        img = np.array(pil_img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Modalità
        mode = 'aggressive' if aggressive else 'balanced'
        
        # Enhance
        enhanced = self.enhance_image(img, mode=mode)
        
        # Crop
        if auto_crop:
            enhanced = self.auto_crop_borders(enhanced)
        
        # Salva alta qualità
        success = cv2.imwrite(output_path, enhanced, 
                             [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if not success:
            raise IOError(f"Failed to write: {output_path}")
        
        logger.info(f"Saved: {output_path}")
        return output_path
    
    def process_pdf(
        self,
        input_pdf: str,
        output_pdf: str,
        aggressive: bool = False,
        auto_crop: bool = True,
        dpi: int = 300
    ) -> str:
        """Processa PDF"""
        logger.info(f"Processing PDF: {input_pdf}")
        
        try:
            images = convert_from_path(input_pdf, dpi=dpi)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
        
        num_pages = len(images)
        logger.info(f"PDF: {num_pages} pages")
        
        # Temp directory unica
        import tempfile
        import time
        temp_id = f"pdf_{int(time.time() * 1000)}"
        temp_dir = tempfile.mkdtemp(prefix=temp_id)
        
        try:
            enhanced_paths = []
            mode = 'aggressive' if aggressive else 'balanced'
            
            for i, pil_img in enumerate(images):
                logger.info(f"Page {i+1}/{num_pages}")
                
                # PIL to numpy
                img_array = np.array(pil_img)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Enhance
                enhanced = self.enhance_image(img_array, mode=mode)
                
                # Crop
                if auto_crop:
                    enhanced = self.auto_crop_borders(enhanced)
                
                # Save
                temp_path = os.path.join(temp_dir, f"page_{i:04d}.jpg")
                cv2.imwrite(temp_path, enhanced, 
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                enhanced_paths.append(temp_path)
            
            # Create PDF
            logger.info("Creating PDF...")
            with open(output_pdf, "wb") as f:
                f.write(img2pdf.convert(enhanced_paths))
            
            logger.info(f"Saved: {output_pdf}")
            return output_pdf
        
        finally:
            # Cleanup
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)



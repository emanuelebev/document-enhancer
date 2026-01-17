"""
Document Enhancement Core Module - VERSIONE PROFESSIONALE
Ottimizzato per:
- Rilevamento preciso del foglio
- Rimozione ombre senza macchie
- Leggibilità testo massima
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

# Configurazioni ottimizzate
DEFAULT_DPI = 300
MIN_DPI_THRESHOLD = 150


class DocumentEnhancer:
    """
    Migliora documenti scannerizzati con tecniche professionali
    """
    
    def __init__(self, target_dpi: int = DEFAULT_DPI):
        self.target_dpi = target_dpi
        logger.info(f"DocumentEnhancer initialized (target_dpi: {target_dpi})")
    
    def _find_document_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Trova i 4 angoli del documento in modo preciso
        """
        h, w = image.shape[:2]
        
        # Converti a grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Preprocessing: blur per ridurre rumore
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Edge detection con Canny
        edges = cv2.Canny(blurred, 50, 200)
        
        # 3. Dilata per connettere bordi interrotti
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 4. Trova contorni
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 5. Ordina per area (dal più grande)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 6. Cerca il primo contorno rettangolare
        for contour in contours[:5]:
            # Area minima
            area = cv2.contourArea(contour)
            if area < (w * h) * 0.25:  # Almeno 25% dell'immagine
                continue
            
            # Approssima a poligono
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Se ha 4 vertici -> è il foglio!
            if len(approx) == 4:
                logger.info(f"Document found: {len(approx)} corners")
                return approx.reshape(4, 2)
        
        return None
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Ordina 4 punti: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Somma: top-left ha somma minima, bottom-right massima
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Differenza: top-right ha diff minima, bottom-left massima
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _perspective_transform(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Applica perspective transform per raddrizzare il documento
        """
        rect = self._order_points(corners)
        (tl, tr, br, bl) = rect
        
        # Calcola larghezza
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Calcola altezza
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Punti destinazione (rettangolo perfetto)
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Calcola e applica transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def _remove_shadows_gentle(self, image: np.ndarray) -> np.ndarray:
        """
        Rimozione ombre - VERSIONE CHE FUNZIONAVA
        """
        # Kernel moderato
        kernel_size = max(image.shape) // 15
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 255)
        
        # Background blur
        bg_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Divisione normalizzata (metodo che funzionava)
        bg_blur = np.where(bg_blur == 0, 1, bg_blur)
        normalized = cv2.divide(image.astype(np.float32), bg_blur.astype(np.float32), scale=255.0)
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Blend con originale 75/25
        blended = cv2.addWeighted(normalized, 0.75, image, 0.25, 0)
        
        return blended
    
    def _enhance_text_readability(self, image: np.ndarray) -> np.ndarray:
        """
        Migliora la leggibilità - VERSIONE CHE FUNZIONAVA
        """
        # 1. Denoising leggero
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, 
                                            templateWindowSize=7, 
                                            searchWindowSize=21)
        
        # 2. CLAHE per contrasto
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        contrasted = clahe.apply(denoised)
        
        # 3. Sharpening per caratteri nitidi
        gaussian = cv2.GaussianBlur(contrasted, (0, 0), 1.5)
        sharpened = cv2.addWeighted(contrasted, 1.8, gaussian, -0.8, 0)
        
        # 4. Gamma per scurire caratteri
        gamma = 0.85
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        darkened = cv2.LUT(sharpened, table)
        
        # 5. Normalizzazione finale
        normalized = cv2.normalize(darkened, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def enhance_image(self, image: np.ndarray, mode: str = 'balanced') -> np.ndarray:
        """
        Enhancement completo del documento
        
        Args:
            image: Immagine numpy array
            mode: 'light' | 'balanced' | 'strong'
        
        Returns:
            Immagine migliorata
        """
        original_shape = image.shape
        logger.info(f"Processing: {original_shape[1]}x{original_shape[0]} (mode: {mode})")
        
        # 1. Trova e ritaglia il documento
        corners = self._find_document_corners(image)
        
        if corners is not None:
            logger.info("Applying perspective transform")
            image = self._perspective_transform(image, corners)
        else:
            logger.warning("Document corners not found, using original image")
        
        # 2. Converti a grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 3. Rimozione ombre delicata
        if mode in ['balanced', 'strong']:
            gray = self._remove_shadows_gentle(gray)
            logger.debug("Gentle shadow removal applied")
        
        # 4. Migliora leggibilità testo
        enhanced = self._enhance_text_readability(gray)
        
        # 5. Per mode 'strong': binarizzazione adattiva
        if mode == 'strong':
            enhanced = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            logger.debug("Adaptive threshold applied")
        
        logger.info(f"Enhancement complete: {enhanced.shape[1]}x{enhanced.shape[0]}")
        return enhanced
    
    def process_single_image(
        self,
        input_path: str,
        output_path: str,
        aggressive: bool = False,
        auto_crop: bool = True
    ) -> str:
        """
        Processa singola immagine
        
        Args:
            input_path: Path immagine input
            output_path: Path immagine output
            aggressive: Se True usa mode 'strong', altrimenti 'balanced'
            auto_crop: Se True cerca di rilevare automaticamente il documento
        
        Returns:
            Path dell'immagine salvata
        """
        logger.info(f"Processing: {input_path}")
        
        # Leggi immagine
        pil_img = Image.open(input_path)
        
        # Fix EXIF orientation
        try:
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(pil_img)
        except:
            pass
        
        # Converti a numpy array
        img = np.array(pil_img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Modalità enhancement
        mode = 'strong' if aggressive else 'balanced'
        
        # Enhance (include auto_crop se attivo)
        enhanced = self.enhance_image(img, mode=mode)
        
        # Salva con alta qualità
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
        """
        Processa PDF completo
        
        Args:
            input_pdf: Path PDF input
            output_pdf: Path PDF output
            aggressive: Se True usa mode 'strong'
            auto_crop: Se True cerca di rilevare automaticamente le pagine
            dpi: DPI per conversione PDF -> immagini
        
        Returns:
            Path del PDF salvato
        """
        logger.info(f"Processing PDF: {input_pdf}")
        
        try:
            images = convert_from_path(input_pdf, dpi=dpi)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
        
        num_pages = len(images)
        logger.info(f"PDF: {num_pages} pages")
        
        # Directory temporanea
        import tempfile
        import time
        import shutil
        
        temp_id = f"pdf_{int(time.time() * 1000)}"
        temp_dir = tempfile.mkdtemp(prefix=temp_id)
        
        try:
            enhanced_paths = []
            mode = 'strong' if aggressive else 'balanced'
            
            for i, pil_img in enumerate(images):
                logger.info(f"Processing page {i+1}/{num_pages}")
                
                # PIL to numpy
                img_array = np.array(pil_img)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Enhance
                enhanced = self.enhance_image(img_array, mode=mode)
                
                # Save
                temp_path = os.path.join(temp_dir, f"page_{i:04d}.jpg")
                cv2.imwrite(temp_path, enhanced, 
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                enhanced_paths.append(temp_path)
            
            # Crea PDF finale
            logger.info("Creating PDF...")
            with open(output_pdf, "wb") as f:
                f.write(img2pdf.convert(enhanced_paths))
            
            logger.info(f"Saved: {output_pdf}")
            return output_pdf
        
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
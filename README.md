# 📄 Document Enhancer

Migliora automaticamente la qualità di documenti scannerizzati (PDF, JPG, PNG, TIFF) rendendoli più leggibili.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## 🎯 Funzionalità

- ✅ Enhancement intelligente di documenti scannerizzati
- ✅ Supporto PDF, JPG, PNG, TIFF
- ✅ Upscaling automatico a 300 DPI
- ✅ Rimozione noise e artefatti
- ✅ Correzione illuminazione non uniforme
- ✅ Deskewing automatico (raddrizzamento)
- ✅ Auto-crop bordi
- ✅ Batch processing
- ✅ API REST
- ✅ Interfaccia web drag & drop

## 🚀 Quick Start

### Opzione 1: Docker (Raccomandato)

```bash
# Clone repository
git clone https://github.com/yourusername/document-enhancer.git
cd document-enhancer

# Build & Run
docker-compose up -d

# Apri browser
open http://localhost:5000
```

### Opzione 2: Setup Locale

```bash
# 1. Clone repository
git clone https://github.com/yourusername/document-enhancer.git
cd document-enhancer

# 2. Crea virtual environment
python3 -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install system dependencies
# Ubuntu/Debian:
sudo apt-get install poppler-utils

# macOS:
brew install poppler

# Windows: Scarica poppler da https://github.com/oschwartz10612/poppler-windows/releases/

# 5. Run
python src/api.py

# 6. Apri browser
open http://localhost:5000
```

## 📖 Utilizzo

### Interfaccia Web

1. Apri `http://localhost:5000`
2. Trascina i file o clicca per selezionare
3. Scegli opzioni enhancement
4. Clicca "Migliora Documenti"
5. Download automatico del risultato

### API REST

#### Singolo file

```bash
curl -X POST http://localhost:5000/enhance \
  -F "file=@document.pdf" \
  -F "aggressive=true" \
  -F "auto_crop=true" \
  -o enhanced_document.pdf
```

#### Batch processing

```bash
curl -X POST http://localhost:5000/enhance/batch \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.jpg" \
  -F "files=@doc3.png" \
  -F "aggressive=true" \
  -o enhanced_documents.zip
```

### Python API

```python
from src.enhancer import DocumentEnhancer

# Inizializza
enhancer = DocumentEnhancer(target_dpi=300)

# Processa immagine
enhancer.process_single_image(
    input_path='scanned_document.jpg',
    output_path='enhanced_document.jpg',
    aggressive=True,
    auto_crop=True
)

# Processa PDF
enhancer.process_pdf(
    input_pdf='scanned_document.pdf',
    output_pdf='enhanced_document.pdf',
    aggressive=True,
    auto_crop=True
)
```

## ⚙️ Parametri

- **aggressive** (bool, default: `true`): Enhancement più spinto per scansioni molto rovinate
- **auto_crop** (bool, default: `true`): Rimuove automaticamente bordi bianchi/neri
- **target_dpi** (int, default: `300`): DPI target per output

## 🔧 Configurazione

Copia `.env.example` a `.env` e modifica:

```bash
# Flask
FLASK_ENV=production
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Enhancement
DEFAULT_DPI=300
MAX_FILE_SIZE=52428800  # 50MB
ALLOWED_EXTENSIONS=pdf,jpg,jpeg,png,tiff,bmp

# Logging
LOG_LEVEL=INFO
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Interfaccia web |
| `/health` | GET | Health check |
| `/enhance` | POST | Migliora singolo documento |
| `/enhance/batch` | POST | Batch processing |

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_enhancer.py::test_enhance_image

# Con coverage
python -m pytest --cov=src tests/
```

## 📁 Output

I file processati mantengono il formato originale:
- **PDF** → PDF enhanced
- **JPG/PNG/TIFF** → JPG enhanced (alta qualità, 95%)

## 🐳 Docker

### Build

```bash
docker build -t document-enhancer .
```

### Run

```bash
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  --name doc-enhancer \
  document-enhancer
```

### Docker Compose

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f
```

## 🔍 Come Funziona

Il pipeline di enhancement applica 9 step sequenziali:

1. **Grayscale Conversion** - Converte in scala di grigi
2. **Upscaling** - Porta a minimo 300 DPI (3000px height per A4)
3. **Denoising** - Rimuove grana con algoritmo Non-Local Means
4. **Contrast Enhancement** - CLAHE per migliorare contrasto locale
5. **Shadow Removal** - Normalizza illuminazione dividendo per background stimato
6. **Sharpening** - Kernel convolution per bordi più netti
7. **Adaptive Binarization** - Separa testo da sfondo in modo intelligente
8. **Morphological Cleaning** - Rimuove noise residuo
9. **Deskewing** - Raddrizza documento se necessario

## 🛠️ Tech Stack

- **Backend**: Python 3.8+, Flask
- **Image Processing**: OpenCV, NumPy, Pillow, scikit-image
- **PDF Handling**: pdf2image, img2pdf, poppler
- **Frontend**: Vanilla JS, HTML5, CSS3
- **Container**: Docker, docker-compose

## 📝 Limitazioni

- **Dimensione file**: Max 50MB (configurabile)
- **Formati**: PDF, JPG, PNG, TIFF, BMP
- **Memoria**: ~2GB RAM per PDF di 50+ pagine
- **Performance**: ~5-10 sec/pagina su CPU standard

## 🤝 Contributing

Pull requests benvenute! Per cambiamenti maggiori, apri prima una issue.

## 📄 License

MIT License

---

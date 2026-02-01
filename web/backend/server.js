/**
 * Menu OCR API Server
 * Handles image upload and menu extraction
 */

import express from 'express';
import multer from 'multer';
import cors from 'cors';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join, resolve } from 'path';
import fs from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Get project root (two levels up from web/backend)
const PROJECT_ROOT = resolve(__dirname, '../..');

// Setup upload directory
const UPLOAD_DIR = join(PROJECT_ROOT, 'web/uploads');
const OUTPUT_DIR = join(PROJECT_ROOT, 'web/output');

// Ensure directories exist
await fs.mkdir(UPLOAD_DIR, { recursive: true });
await fs.mkdir(OUTPUT_DIR, { recursive: true });

// Configure multer for image uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files allowed'));
    }
  }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static(UPLOAD_DIR));
app.use('/output', express.static(OUTPUT_DIR));

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Process menu image
app.post('/api/extract', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  const imagePath = req.file.path;
  const basename = req.file.filename.replace(/\.[^.]+$/, '');
  const outputJson = join(OUTPUT_DIR, `${basename}.json`);
  const outputImage = join(OUTPUT_DIR, `${basename}_annotated.jpg`);

  try {
    // Run Python extraction script
    const result = await runExtraction(imagePath, outputJson, outputImage);
    
    // Read results
    const menuJson = JSON.parse(await fs.readFile(outputJson, 'utf-8'));
    
    res.json({
      success: true,
      menu: menuJson,
      images: {
        original: `/uploads/${req.file.filename}`,
        annotated: `/output/${basename}_annotated.jpg`
      },
      processingTime: result.time
    });
  } catch (error) {
    console.error('Extraction error:', error);
    res.status(500).json({ 
      error: 'Extraction failed',
      details: error.message 
    });
  }
});

// Run Python extraction
function runExtraction(imagePath, outputJson, outputImage) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    
    const pythonScript = join(PROJECT_ROOT, 'scripts/api_extract.py');
    const venvPython = join(PROJECT_ROOT, '.venv/bin/python');
    const args = [
      pythonScript,
      '--image', imagePath,
      '--output-json', outputJson,
      '--output-image', outputImage
    ];
    
    const proc = spawn(venvPython, args, {
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONPATH: PROJECT_ROOT }
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      const time = Date.now() - startTime;
      if (code === 0) {
        resolve({ time, stdout, stderr });
      } else {
        reject(new Error(`Process exited with code ${code}: ${stderr}`));
      }
    });

    proc.on('error', (err) => {
      reject(err);
    });
  });
}

// List recent extractions
app.get('/api/extractions', async (req, res) => {
  try {
    const files = await fs.readdir(OUTPUT_DIR);
    const jsonFiles = files.filter(f => f.endsWith('.json'));
    
    const extractions = await Promise.all(
      jsonFiles.slice(-20).map(async (filename) => {
        const content = await fs.readFile(join(OUTPUT_DIR, filename), 'utf-8');
        const stats = await fs.stat(join(OUTPUT_DIR, filename));
        return {
          id: filename.replace('.json', ''),
          menu: JSON.parse(content),
          createdAt: stats.mtime
        };
      })
    );
    
    res.json(extractions.sort((a, b) => 
      new Date(b.createdAt) - new Date(a.createdAt)
    ));
  } catch (error) {
    res.json([]);
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Menu OCR API running on http://localhost:${PORT}`);
  console.log(`Project root: ${PROJECT_ROOT}`);
});

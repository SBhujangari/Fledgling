import { Router, Request, Response } from 'express';
import { spawn } from 'child_process';
import path from 'path';

const router = Router();

/**
 * POST /api/lab-autologger/extract
 * Extract ISA-Tab compliant metadata from unstructured lab notes
 */
router.post('/extract', async (req: Request, res: Response) => {
  try {
    const { labNotes } = req.body;

    if (!labNotes || typeof labNotes !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Lab notes are required'
      });
    }

    // Call Python script for extraction using OpenAI
    const scriptPath = path.join(__dirname, '../../..', 'slm_swap', 'lab_autologger.py');

    const pythonProcess = spawn('python', [
      scriptPath,
      '--lab-notes', labNotes
    ], {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1'
      }
    });

    let stdoutData = '';
    let stderrData = '';

    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Lab autologger stderr:', stderrData);
        return res.status(500).json({
          success: false,
          error: `Extraction failed with code ${code}: ${stderrData}`
        });
      }

      try {
        // Parse the extraction result from stdout
        const resultMatch = stdoutData.match(/<<<EXTRACTION_RESULT>>>([\s\S]*?)<<<END_RESULT>>>/);

        if (!resultMatch) {
          console.error('No result marker found in output:', stdoutData);
          return res.status(500).json({
            success: false,
            error: 'Failed to parse extraction result'
          });
        }

        const result = JSON.parse(resultMatch[1].trim());

        res.json({
          success: true,
          data: result.data,
          confidence_score: result.confidence_score || 0.9,
          raw_notes: labNotes
        });
      } catch (error) {
        console.error('Failed to parse extraction result:', error);
        console.error('Stdout:', stdoutData);
        return res.status(500).json({
          success: false,
          error: 'Failed to parse extraction result'
        });
      }
    });
  } catch (error) {
    console.error('Lab autologger error:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/lab-autologger/schema
 * Get the ISA-Tab schema definition
 */
router.get('/schema', (_req: Request, res: Response) => {
  res.json({
    schema: {
      study_identifier: 'string',
      study_title: 'string',
      study_description: 'string',
      study_submission_date: 'string (ISO 8601)',
      study_public_release_date: 'string (ISO 8601)',
      study_file_name: 'string',
      study_factors: 'array of { name, type, values[] }',
      study_assays: 'array of { measurement_type, technology_type, technology_platform }',
      study_protocols: 'array of { name, type, description, parameters[] }',
      study_contacts: 'array of { name, affiliation, email, role }'
    },
    description: 'ISA-Tab (Investigation, Study, Assay) standard format for experiment metadata'
  });
});

/**
 * POST /api/lab-autologger/validate
 * Validate ISA-Tab data against the schema
 */
router.post('/validate', async (req: Request, res: Response) => {
  try {
    const { data } = req.body;

    if (!data) {
      return res.status(400).json({
        valid: false,
        errors: ['No data provided']
      });
    }

    const errors: string[] = [];

    // Basic validation
    const requiredFields = ['study_identifier', 'study_title', 'study_description'];
    for (const field of requiredFields) {
      if (!data[field]) {
        errors.push(`Missing required field: ${field}`);
      }
    }

    // Validate arrays
    if (!Array.isArray(data.study_contacts)) {
      errors.push('study_contacts must be an array');
    }
    if (!Array.isArray(data.study_assays)) {
      errors.push('study_assays must be an array');
    }
    if (!Array.isArray(data.study_protocols)) {
      errors.push('study_protocols must be an array');
    }
    if (!Array.isArray(data.study_factors)) {
      errors.push('study_factors must be an array');
    }

    res.json({
      valid: errors.length === 0,
      errors
    });
  } catch (error) {
    res.status(500).json({
      valid: false,
      errors: [error instanceof Error ? error.message : 'Unknown error']
    });
  }
});

/**
 * GET /api/lab-autologger/examples
 * Get example lab notes for testing
 */
router.get('/examples', (_req: Request, res: Response) => {
  res.json({
    examples: [
      {
        title: 'Temperature Growth Study',
        notes: `Lab Notebook - Oct 15, 2024
Researcher: Dr. Sarah Chen (sarah.chen@biotech.edu), MIT Biology Dept

Study: Effect of Temperature on E. coli Growth Rates
Running a series of growth curve experiments to measure how different temperatures affect E. coli strain DH5α growth.

Protocol:
- Inoculated 5 flasks with 50mL LB medium each
- Incubated at different temps: 25°C, 30°C, 37°C, 42°C, 45°C
- Measured OD600 every hour for 12 hours
- Used BioTek plate reader

Preliminary observations: 37°C showing optimal growth, 45°C shows significant growth inhibition.`
      },
      {
        title: 'Drug Response Assay',
        notes: `EXPERIMENT LOG - 2024-10-20
PI: Dr. James Wilson, Pharmacology Dept, Stanford

Testing compound XYZ-142 cytotoxicity on HeLa cells

Methods:
- Plated cells at 10,000 cells/well in 96-well plates
- Treated with XYZ-142 at concentrations: 0, 1, 10, 100, 1000 µM
- Incubated 48h at 37°C, 5% CO2
- Measured viability using MTT assay
- Read absorbance at 570nm on Tecan plate reader

Results show dose-dependent response, IC50 appears to be around 50 µM`
      }
    ]
  });
});

/**
 * POST /api/lab-autologger/batch
 * Extract ISA-Tab data from multiple lab notes
 */
router.post('/batch', async (req: Request, res: Response) => {
  try {
    const { labNotesArray } = req.body;

    if (!Array.isArray(labNotesArray)) {
      return res.status(400).json({
        success: false,
        error: 'labNotesArray must be an array of strings'
      });
    }

    // Process each lab note sequentially (could be parallelized)
    const results = [];
    for (const notes of labNotesArray) {
      // Call the extraction endpoint internally
      // In a real implementation, extract the logic into a shared function
      results.push({
        success: true,
        notes: notes.substring(0, 100) + '...',
        message: 'Batch processing not fully implemented yet'
      });
    }

    res.json({
      success: true,
      count: results.length,
      results
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/lab-autologger/health
 * Check if OpenAI API key is configured
 */
router.get('/health', (_req: Request, res: Response) => {
  const hasOpenAIKey = !!process.env.OPENAI_API_KEY;

  res.json({
    status: hasOpenAIKey ? 'ready' : 'not_configured',
    message: hasOpenAIKey
      ? 'OpenAI API key configured'
      : 'OpenAI API key not found in environment',
    model: 'gpt-4o-mini'
  });
});

export default router;

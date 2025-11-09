import fs from 'fs';
import path from 'path';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const SLM_DIR = path.join(REPO_ROOT, 'slm_swap');
const CATALOG_PATH = path.join(SLM_DIR, 'model_catalog.json');
const SELECTION_PATH = path.join(SLM_DIR, 'model_selection.json');

export type SlmSource = 'huggingface' | 'local_adapter' | 'local';

export interface CatalogModel {
  id: string;
  label: string;
  source: SlmSource;
  description?: string;
  modelId?: string;
  path?: string;
  capabilities?: string[];
  default?: boolean;
}

export interface CatalogModelWithStatus extends CatalogModel {
  available: boolean;
  location: string | null;
}

export interface SelectionState {
  modelId: string;
  selectedAt: string;
}

function assertCatalog(): void {
  if (!fs.existsSync(CATALOG_PATH)) {
    throw new Error(`Missing model catalog at ${CATALOG_PATH}`);
  }
}

function readCatalog(): CatalogModel[] {
  assertCatalog();
  const raw = fs.readFileSync(CATALOG_PATH, 'utf-8');
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error('Model catalog must be an array.');
  }
  return parsed as CatalogModel[];
}

function readSelection(): SelectionState | null {
  if (!fs.existsSync(SELECTION_PATH)) {
    return null;
  }
  const raw = fs.readFileSync(SELECTION_PATH, 'utf-8');
  return JSON.parse(raw) as SelectionState;
}

function writeSelection(state: SelectionState): void {
  fs.writeFileSync(SELECTION_PATH, JSON.stringify(state, null, 2));
}

function resolveLocation(model: CatalogModel): { available: boolean; location: string | null } {
  if (model.source === 'huggingface') {
    return { available: Boolean(model.modelId), location: model.modelId ?? null };
  }

  const relativePath = model.path ?? '';
  const absolutePath = path.resolve(REPO_ROOT, relativePath);
  const insideRepo = !path.relative(REPO_ROOT, absolutePath).startsWith('..');
  if (!insideRepo) {
    return { available: false, location: absolutePath };
  }
  const exists = fs.existsSync(absolutePath);
  return { available: exists, location: absolutePath };
}

export function listModels(): CatalogModelWithStatus[] {
  const catalog = readCatalog();
  return catalog.map((entry) => {
    const status = resolveLocation(entry);
    return {
      ...entry,
      available: status.available,
      location: status.location,
    };
  });
}

export function getSelectedModel(): SelectionState | null {
  return readSelection();
}

export function selectModel(modelId: string): SelectionState {
  const catalog = readCatalog();
  const exists = catalog.some((entry) => entry.id === modelId);
  if (!exists) {
    throw new Error(`Unknown model id "${modelId}".`);
  }
  const state: SelectionState = {
    modelId,
    selectedAt: new Date().toISOString(),
  };
  writeSelection(state);
  return state;
}

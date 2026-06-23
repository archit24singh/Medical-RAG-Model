#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
  process.exit(1);
}

let input;
try {
  input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (e) {
  console.error('Failed to parse input JSON:', e.message);
  process.exit(1);
}

const { fileNodes, importEdges, allEdges } = input;

// ─── A. Directory Grouping ─────────────────────────────────────────────────
function getFilePath(node) {
  return node.filePath || '';
}

// Compute common prefix among all file paths
const allPaths = fileNodes.map(n => getFilePath(n));

function commonPrefix(paths) {
  if (!paths.length) return '';
  const parts = paths.map(p => p.split('/'));
  const minLen = Math.min(...parts.map(p => p.length));
  let prefix = [];
  for (let i = 0; i < minLen - 1; i++) {
    const seg = parts[0][i];
    if (parts.every(p => p[i] === seg)) {
      prefix.push(seg);
    } else break;
  }
  return prefix.join('/');
}

const commonPfx = commonPrefix(allPaths);
const prefixParts = commonPfx ? commonPfx.split('/').length : 0;

function getDirectoryGroup(filePath) {
  const parts = filePath.split('/');
  if (prefixParts > 0) {
    // strip common prefix
    const remaining = parts.slice(prefixParts);
    if (remaining.length === 0) return 'root';
    // Return first dir segment after prefix
    if (remaining.length === 1) return 'root'; // file at prefix level
    return remaining[0];
  } else {
    if (parts.length === 1) return 'root';
    return parts[0];
  }
}

const directoryGroups = {};
for (const node of fileNodes) {
  const fp = getFilePath(node);
  let group = getDirectoryGroup(fp);
  // Normalize: map specific sub-paths
  if (!directoryGroups[group]) directoryGroups[group] = [];
  directoryGroups[group].push(node.id);
}

// ─── B. Node Type Grouping ─────────────────────────────────────────────────
const nodeTypeGroups = {};
for (const node of fileNodes) {
  const t = node.type || 'file';
  if (!nodeTypeGroups[t]) nodeTypeGroups[t] = [];
  nodeTypeGroups[t].push(node.id);
}

// ─── C. Import Adjacency / Fan-in / Fan-out ──────────────────────────────
const fanIn = {};
const fanOut = {};
const nodeById = {};
for (const node of fileNodes) {
  nodeById[node.id] = node;
  fanIn[node.id] = 0;
  fanOut[node.id] = 0;
}

for (const edge of importEdges) {
  if (edge.type === 'imports') {
    fanOut[edge.source] = (fanOut[edge.source] || 0) + 1;
    fanIn[edge.target] = (fanIn[edge.target] || 0) + 1;
  }
}

// ─── D. Cross-Category Dependency Analysis ────────────────────────────────
const crossCategoryMap = {};
for (const edge of allEdges) {
  const srcNode = nodeById[edge.source];
  const tgtNode = nodeById[edge.target];
  if (!srcNode || !tgtNode) continue;
  const fromType = srcNode.type;
  const toType = tgtNode.type;
  const key = `${fromType}->${toType}:${edge.type}`;
  crossCategoryMap[key] = (crossCategoryMap[key] || 0) + 1;
}

const crossCategoryEdges = Object.entries(crossCategoryMap).map(([key, count]) => {
  const [types, edgeType] = key.split(':');
  const [fromType, toType] = types.split('->');
  return { fromType, toType, edgeType, count };
});

// ─── E. Inter-Group Import Frequency ─────────────────────────────────────
const nodeGroupMap = {};
for (const node of fileNodes) {
  const fp = getFilePath(node);
  nodeGroupMap[node.id] = getDirectoryGroup(fp);
}

const interGroupMap = {};
for (const edge of importEdges) {
  if (edge.type !== 'imports') continue;
  const fromGroup = nodeGroupMap[edge.source];
  const toGroup = nodeGroupMap[edge.target];
  if (!fromGroup || !toGroup || fromGroup === toGroup) continue;
  const key = `${fromGroup}->${toGroup}`;
  interGroupMap[key] = (interGroupMap[key] || 0) + 1;
}

const interGroupImports = Object.entries(interGroupMap).map(([key, count]) => {
  const [from, to] = key.split('->');
  return { from, to, count };
}).sort((a, b) => b.count - a.count);

// ─── F. Intra-Group Import Density ───────────────────────────────────────
const intraGroupDensity = {};
for (const group of Object.keys(directoryGroups)) {
  intraGroupDensity[group] = { internalEdges: 0, totalEdges: 0, density: 0 };
}

for (const edge of importEdges) {
  if (edge.type !== 'imports') continue;
  const fromGroup = nodeGroupMap[edge.source];
  const toGroup = nodeGroupMap[edge.target];
  if (!fromGroup || !toGroup) continue;
  if (fromGroup === toGroup) {
    if (intraGroupDensity[fromGroup]) intraGroupDensity[fromGroup].internalEdges++;
    if (intraGroupDensity[fromGroup]) intraGroupDensity[fromGroup].totalEdges++;
  } else {
    if (intraGroupDensity[fromGroup]) intraGroupDensity[fromGroup].totalEdges++;
    if (intraGroupDensity[toGroup]) intraGroupDensity[toGroup].totalEdges++;
  }
}

for (const group of Object.keys(intraGroupDensity)) {
  const g = intraGroupDensity[group];
  g.density = g.totalEdges > 0 ? parseFloat((g.internalEdges / g.totalEdges).toFixed(3)) : 0;
}

// ─── G. Directory Pattern Matching ───────────────────────────────────────
const dirPatterns = {
  routes: 'api', api: 'api', controllers: 'api', endpoints: 'api', handlers: 'api',
  serializers: 'api', controller: 'api', routers: 'api', blueprints: 'api',
  services: 'service', core: 'service', lib: 'service', domain: 'service', logic: 'service',
  composables: 'service', signals: 'service', mailers: 'service', jobs: 'service', channels: 'service',
  internal: 'service',
  models: 'data', db: 'data', data: 'data', persistence: 'data', repository: 'data',
  entities: 'data', migrations: 'data', sql: 'data', database: 'data', schema: 'data', entity: 'data',
  components: 'ui', views: 'ui', pages: 'ui', ui: 'ui', layouts: 'ui', screens: 'ui',
  frontend: 'ui',
  middleware: 'middleware', plugins: 'middleware', interceptors: 'middleware', guards: 'middleware',
  utils: 'utility', helpers: 'utility', common: 'utility', shared: 'utility', tools: 'utility',
  pkg: 'utility', templatetags: 'utility',
  config: 'config', constants: 'config', env: 'config', settings: 'config',
  management: 'config', commands: 'config',
  '__tests__': 'test', test: 'test', tests: 'test', spec: 'test', specs: 'test',
  types: 'types', interfaces: 'types', schemas: 'types', contracts: 'types', dtos: 'types',
  dto: 'types', request: 'types', response: 'types',
  hooks: 'hooks',
  store: 'state', state: 'state', reducers: 'state', actions: 'state', slices: 'state',
  assets: 'assets', static: 'assets', public: 'assets',
  bin: 'entry', cmd: 'entry',
  docs: 'documentation', documentation: 'documentation', wiki: 'documentation',
  deploy: 'infrastructure', deployment: 'infrastructure', infra: 'infrastructure',
  infrastructure: 'infrastructure', k8s: 'infrastructure', kubernetes: 'infrastructure',
  helm: 'infrastructure', charts: 'infrastructure', terraform: 'infrastructure',
  tf: 'infrastructure', docker: 'infrastructure',
  '.github': 'ci-cd', '.gitlab': 'ci-cd', '.circleci': 'ci-cd',
  rag: 'service', scripts: 'utility',
};

const patternMatches = {};
for (const group of Object.keys(directoryGroups)) {
  patternMatches[group] = dirPatterns[group.toLowerCase()] || 'unknown';
}

// Also check file-level patterns for root-level files
function filePatternLabel(filePath) {
  const name = path.basename(filePath);
  const ext = path.extname(filePath).toLowerCase();
  if (/\.(test|spec)\.[a-z]+$/.test(name) || /^test_/.test(name)) return 'test';
  if (/\.d\.ts$/.test(name)) return 'types';
  if (/^(index\.(ts|js)|__init__\.py)$/.test(name)) return 'entry';
  if (name === 'manage.py') return 'entry';
  if (name === 'wsgi.py' || name === 'asgi.py') return 'config';
  if (/^main\.(go|rs)$/.test(name) || name === 'lib.rs') return 'entry';
  if (name === 'Application.java' || name === 'Program.cs') return 'entry';
  if (name === 'config.ru') return 'entry';
  if (['Cargo.toml','go.mod','Gemfile','pom.xml','build.gradle','composer.json'].includes(name)) return 'config';
  if (name === 'Dockerfile' || /^docker-compose/.test(name)) return 'infrastructure';
  if (/\.(tf|tfvars)$/.test(name)) return 'infrastructure';
  if (name === 'Jenkinsfile' || /^\.gitlab-ci/.test(name)) return 'ci-cd';
  if (/\.sql$/.test(name)) return 'data';
  if (/\.(graphql|gql|proto)$/.test(name)) return 'types';
  if (/\.(md|rst)$/.test(name)) return 'documentation';
  if (name === 'Makefile') return 'infrastructure';
  if (name === 'requirements.txt') return 'documentation';
  if (/\.(yaml|yml|json|toml|ini|cfg|env)$/.test(name) && !name.startsWith('.')) return 'config';
  return null;
}

// ─── H. Deployment Topology Detection ────────────────────────────────────
const infraFiles = [];
let hasDockerfile = false, hasCompose = false, hasK8s = false, hasTerraform = false, hasCI = false;

for (const node of fileNodes) {
  const fp = getFilePath(node);
  const name = path.basename(fp);
  if (name === 'Dockerfile' || name.startsWith('Dockerfile.')) { hasDockerfile = true; infraFiles.push(fp); }
  if (/^docker-compose/.test(name)) { hasCompose = true; infraFiles.push(fp); }
  if (/\.(tf|tfvars)$/.test(name)) { hasTerraform = true; infraFiles.push(fp); }
  if (fp.includes('.github/workflows') || /^\.gitlab-ci/.test(name) || name === 'Jenkinsfile') { hasCI = true; infraFiles.push(fp); }
  if (fp.includes('k8s/') || fp.includes('kubernetes/')) { hasK8s = true; infraFiles.push(fp); }
}

const deploymentTopology = { hasDockerfile, hasCompose, hasK8s, hasTerraform, hasCI, infraFiles };

// ─── I. Data Pipeline Detection ───────────────────────────────────────────
const schemaFiles = [];
const migrationFiles = [];
const dataModelFiles = [];
const apiHandlerFiles = [];

for (const node of fileNodes) {
  const fp = getFilePath(node);
  const name = path.basename(fp);
  if (/\.(graphql|gql|proto|sql)$/.test(name) || name.includes('schema')) schemaFiles.push(fp);
  if (fp.includes('migration') || fp.includes('migrate')) migrationFiles.push(fp);
  if (fp.includes('model') || fp.includes('db/') || fp.includes('entity')) dataModelFiles.push(fp);
  if (fp.includes('route') || fp.includes('controller') || fp.includes('endpoint') || fp.includes('handler') || fp.includes('main.py')) apiHandlerFiles.push(fp);
}

const dataPipeline = { schemaFiles, migrationFiles, dataModelFiles, apiHandlerFiles };

// ─── J. Documentation Coverage ───────────────────────────────────────────
const groupsWithDocs = new Set();
for (const node of fileNodes) {
  if (node.type === 'document') {
    const fp = getFilePath(node);
    const group = getDirectoryGroup(fp);
    groupsWithDocs.add(group);
  }
}

const totalGroups = Object.keys(directoryGroups).length;
const undocumentedGroups = Object.keys(directoryGroups).filter(g => !groupsWithDocs.has(g));
const docCoverage = {
  groupsWithDocs: groupsWithDocs.size,
  totalGroups,
  coverageRatio: parseFloat((groupsWithDocs.size / totalGroups).toFixed(2)),
  undocumentedGroups
};

// ─── K. Dependency Direction ──────────────────────────────────────────────
const pairCounts = {};
for (const edge of importEdges) {
  if (edge.type !== 'imports') continue;
  const fromGroup = nodeGroupMap[edge.source];
  const toGroup = nodeGroupMap[edge.target];
  if (!fromGroup || !toGroup || fromGroup === toGroup) continue;
  const fwd = `${fromGroup}->${toGroup}`;
  const rev = `${toGroup}->${fromGroup}`;
  pairCounts[fwd] = (pairCounts[fwd] || 0) + 1;
}

const dependencyDirection = [];
const seen = new Set();
for (const [key, count] of Object.entries(pairCounts)) {
  const [a, b] = key.split('->');
  const pairKey = [a,b].sort().join(':');
  if (seen.has(pairKey)) continue;
  seen.add(pairKey);
  const rev = `${b}->${a}`;
  const revCount = pairCounts[rev] || 0;
  if (count >= revCount) {
    dependencyDirection.push({ dependent: a, dependsOn: b });
  } else {
    dependencyDirection.push({ dependent: b, dependsOn: a });
  }
}

// ─── File Stats ───────────────────────────────────────────────────────────
const filesPerGroup = {};
for (const [group, ids] of Object.entries(directoryGroups)) {
  filesPerGroup[group] = ids.length;
}
const nodeTypeCounts = {};
for (const [type, ids] of Object.entries(nodeTypeGroups)) {
  nodeTypeCounts[type] = ids.length;
}

const fileStats = {
  totalFileNodes: fileNodes.length,
  filesPerGroup,
  nodeTypeCounts
};

// ─── Output ───────────────────────────────────────────────────────────────
const results = {
  scriptCompleted: true,
  directoryGroups,
  nodeTypeGroups,
  crossCategoryEdges,
  interGroupImports,
  intraGroupDensity,
  patternMatches,
  deploymentTopology,
  dataPipeline,
  docCoverage,
  dependencyDirection,
  fileStats,
  fileFanIn: fanIn,
  fileFanOut: fanOut
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log('Analysis complete. Output written to', outputPath);
  process.exit(0);
} catch (e) {
  console.error('Failed to write output:', e.message);
  process.exit(1);
}

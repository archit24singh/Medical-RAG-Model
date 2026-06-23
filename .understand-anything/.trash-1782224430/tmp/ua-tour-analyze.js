#!/usr/bin/env node
// Tour Analysis Script for medical-rag-system

const fs = require('fs');

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node ua-tour-analyze.js <input.json> <output.json>');
  process.exit(1);
}

let input;
try {
  input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (e) {
  console.error('Failed to parse input JSON:', e.message);
  process.exit(1);
}

const { nodes, edges, layers } = input;

// Build node map
const nodeMap = {};
for (const node of nodes) {
  nodeMap[node.id] = node;
}

// A. Fan-In Ranking
const fanIn = {};
for (const node of nodes) { fanIn[node.id] = 0; }
for (const edge of edges) {
  if (fanIn[edge.target] !== undefined) {
    fanIn[edge.target]++;
  }
}
const fanInRanking = Object.entries(fanIn)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 20)
  .map(([id, count]) => ({
    id,
    fanIn: count,
    name: nodeMap[id] ? nodeMap[id].name : id
  }));

// B. Fan-Out Ranking
const fanOut = {};
for (const node of nodes) { fanOut[node.id] = 0; }
for (const edge of edges) {
  if (fanOut[edge.source] !== undefined) {
    fanOut[edge.source]++;
  }
}
const fanOutRanking = Object.entries(fanOut)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 20)
  .map(([id, count]) => ({
    id,
    fanOut: count,
    name: nodeMap[id] ? nodeMap[id].name : id
  }));

// C. Entry Point Candidates
const entryFilenames = [
  'index.ts','index.js','main.ts','main.js','app.ts','app.js',
  'server.ts','server.js','mod.rs','main.go','main.py','main.rs',
  'manage.py','app.py','wsgi.py','asgi.py','run.py','__main__.py',
  'Application.java','Main.java','Program.cs','config.ru','index.php',
  'App.swift','Application.kt','main.cpp','main.c'
];

const totalNodes = nodes.length;
const fanOutValues = Object.values(fanOut).sort((a, b) => a - b);
const fanInValues = Object.values(fanIn).sort((a, b) => a - b);
const fanOutTop10Threshold = fanOutValues[Math.floor(fanOutValues.length * 0.9)];
const fanInBottom25Threshold = fanInValues[Math.floor(fanInValues.length * 0.25)];

const candidateScores = [];
for (const node of nodes) {
  let score = 0;

  if (node.type === 'document' && node.name === 'README.md' && node.filePath === 'README.md') {
    score += 5;
    candidateScores.push({ id: node.id, score, name: node.name, summary: node.summary });
    continue;
  }
  if (node.type === 'document' && node.name.endsWith('.md') && !node.filePath.includes('/')) {
    score += 2;
  }

  if (node.type === 'file') {
    if (entryFilenames.includes(node.name)) score += 3;
    const depth = node.filePath.split('/').length - 1;
    if (depth <= 1) score += 1;
    if (fanOut[node.id] >= fanOutTop10Threshold) score += 1;
    if (fanIn[node.id] <= fanInBottom25Threshold) score += 1;
  }

  if (score > 0) {
    candidateScores.push({ id: node.id, score, name: node.name, summary: node.summary });
  }
}
candidateScores.sort((a, b) => b.score - a.score);
const entryPointCandidates = candidateScores.slice(0, 5);

// D. BFS from top code entry point
// Find top code (non-document) entry point
let bfsStart = null;
for (const c of candidateScores) {
  const node = nodeMap[c.id];
  if (node && node.type === 'file') {
    bfsStart = c.id;
    break;
  }
}

const bfsResult = { startNode: bfsStart, order: [], depthMap: {}, byDepth: {} };
if (bfsStart) {
  // Build adjacency for imports/calls edges
  const adj = {};
  for (const node of nodes) { adj[node.id] = []; }
  for (const edge of edges) {
    if ((edge.type === 'imports' || edge.type === 'calls') && adj[edge.source] !== undefined) {
      adj[edge.source].push(edge.target);
    }
  }

  const visited = new Set();
  const queue = [{ id: bfsStart, depth: 0 }];
  visited.add(bfsStart);

  while (queue.length > 0) {
    const { id, depth } = queue.shift();
    bfsResult.order.push(id);
    bfsResult.depthMap[id] = depth;
    if (!bfsResult.byDepth[depth]) bfsResult.byDepth[depth] = [];
    bfsResult.byDepth[depth].push(id);

    for (const neighbor of (adj[id] || [])) {
      if (!visited.has(neighbor) && nodeMap[neighbor]) {
        visited.add(neighbor);
        queue.push({ id: neighbor, depth: depth + 1 });
      }
    }
  }
}

// E. Non-code file inventory
const nonCodeFiles = {
  documentation: [],
  infrastructure: [],
  data: [],
  config: []
};

for (const node of nodes) {
  if (node.type === 'document') {
    nonCodeFiles.documentation.push({ id: node.id, name: node.name, type: node.type, summary: node.summary });
  } else if (['service','pipeline','resource'].includes(node.type)) {
    nonCodeFiles.infrastructure.push({ id: node.id, name: node.name, type: node.type, summary: node.summary });
  } else if (['table','schema','endpoint'].includes(node.type)) {
    nonCodeFiles.data.push({ id: node.id, name: node.name, type: node.type, summary: node.summary });
  } else if (node.type === 'config') {
    nonCodeFiles.config.push({ id: node.id, name: node.name, type: node.type, summary: node.summary });
  }
}

// F. Tightly Coupled Clusters
// Build bidirectional edge sets
const edgePairs = new Set();
const edgeSet = new Set();
for (const edge of edges) {
  edgeSet.add(`${edge.source}|||${edge.target}`);
}

const bidirectional = [];
for (const edge of edges) {
  const reverseKey = `${edge.target}|||${edge.source}`;
  const forwardKey = `${edge.source}|||${edge.target}`;
  if (edgeSet.has(reverseKey) && !edgePairs.has(reverseKey)) {
    edgePairs.add(forwardKey);
    bidirectional.push([edge.source, edge.target]);
  }
}

// Build clusters from bidirectional pairs, then expand
const clusterSets = [];
for (const [a, b] of bidirectional) {
  let merged = false;
  for (const cluster of clusterSets) {
    if (cluster.has(a) || cluster.has(b)) {
      cluster.add(a);
      cluster.add(b);
      merged = true;
      break;
    }
  }
  if (!merged) {
    clusterSets.push(new Set([a, b]));
  }
}

// Expand: add nodes connected to 2+ cluster members
for (const cluster of clusterSets) {
  const members = Array.from(cluster);
  for (const node of nodes) {
    if (cluster.has(node.id)) continue;
    let connections = 0;
    for (const edge of edges) {
      if ((edge.source === node.id && cluster.has(edge.target)) ||
          (edge.target === node.id && cluster.has(edge.source))) {
        connections++;
      }
    }
    if (connections >= 2 && cluster.size < 5) {
      cluster.add(node.id);
    }
  }
}

// Count internal edges for each cluster
const clusters = clusterSets.map(cluster => {
  const members = Array.from(cluster);
  let edgeCount = 0;
  for (const edge of edges) {
    if (cluster.has(edge.source) && cluster.has(edge.target)) edgeCount++;
  }
  return { nodes: members, edgeCount };
}).sort((a, b) => b.edgeCount - a.edgeCount).slice(0, 10);

// G. Layer list
const layerData = {
  count: layers.length,
  list: layers.map(l => ({ id: l.id, name: l.name, description: l.description }))
};

// H. Node Summary Index
const nodeSummaryIndex = {};
for (const node of nodes) {
  nodeSummaryIndex[node.id] = {
    name: node.name,
    type: node.type,
    summary: node.summary
  };
}

const output = {
  scriptCompleted: true,
  entryPointCandidates,
  fanInRanking,
  fanOutRanking,
  bfsTraversal: bfsResult,
  nonCodeFiles,
  clusters,
  layers: layerData,
  nodeSummaryIndex,
  totalNodes: nodes.length,
  totalEdges: edges.length
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log('Analysis complete. Results written to', outputPath);
} catch (e) {
  console.error('Failed to write output:', e.message);
  process.exit(1);
}

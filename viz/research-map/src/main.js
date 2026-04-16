/**
 * main.js
 *
 * Entry point. Sets up Three.js scene, camera, WebGLRenderer,
 * CSS2DRenderer, background grid, and wires up canvas controls,
 * nodes, and edges.
 */

import * as THREE from 'three';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { InfiniteCanvas } from './canvas.js';
import { NodeManager } from './nodes.js';
import { EdgeManager } from './edges.js';
import { nodes as graphNodes, edges as graphEdges } from './graph-data.js';

// =========================================================
// Scene setup
// =========================================================

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f172a);

// =========================================================
// Camera — OrthographicCamera for true 2D projection
// =========================================================

const viewWidth = window.innerWidth;
const viewHeight = window.innerHeight;

const camera = new THREE.OrthographicCamera(
  -viewWidth / 2,   // left
   viewWidth / 2,   // right
   viewHeight / 2,  // top
  -viewHeight / 2,  // bottom
  1,               // near
  1000             // far
);
camera.position.z = 10;
camera.zoom = 0.8;
camera.updateProjectionMatrix();

// =========================================================
// WebGL renderer
// =========================================================

const glRenderer = new THREE.WebGLRenderer({ antialias: true });
glRenderer.setPixelRatio(window.devicePixelRatio);
glRenderer.setSize(viewWidth, viewHeight);

const rendererContainer = document.getElementById('renderer-container');
rendererContainer.appendChild(glRenderer.domElement);

// =========================================================
// CSS2D renderer (for node cards and edge labels)
// =========================================================

const css2dRenderer = new CSS2DRenderer();
css2dRenderer.setSize(viewWidth, viewHeight);
css2dRenderer.domElement.style.position = 'absolute';
css2dRenderer.domElement.style.top = '0';
css2dRenderer.domElement.style.left = '0';
css2dRenderer.domElement.style.width = '100%';
css2dRenderer.domElement.style.height = '100%';
css2dRenderer.domElement.style.pointerEvents = 'none';

const css2dContainer = document.getElementById('css2d-container');
css2dContainer.appendChild(css2dRenderer.domElement);

// =========================================================
// Background grid
// =========================================================

function buildBackgroundGrid(gridSize = 60, halfExtent = 4000) {
  const material = new THREE.LineBasicMaterial({
    color: 0x1e293b,
    transparent: true,
    opacity: 0.6,
    depthTest: false,
  });

  const points = [];

  // Vertical lines
  for (let x = -halfExtent; x <= halfExtent; x += gridSize) {
    points.push(new THREE.Vector3(x, -halfExtent, -1));
    points.push(new THREE.Vector3(x,  halfExtent, -1));
  }

  // Horizontal lines
  for (let y = -halfExtent; y <= halfExtent; y += gridSize) {
    points.push(new THREE.Vector3(-halfExtent, y, -1));
    points.push(new THREE.Vector3( halfExtent, y, -1));
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  // Use LineSegments (pairs of points) for the grid
  const grid = new THREE.LineSegments(geometry, material);
  return grid;
}

const backgroundGrid = buildBackgroundGrid();
scene.add(backgroundGrid);

// =========================================================
// Infinite canvas controls
// =========================================================

const infiniteCanvas = new InfiniteCanvas(glRenderer.domElement, camera);

// =========================================================
// Edge manager (must be added before nodes so edges render behind)
// =========================================================

// =========================================================
// Node manager
// =========================================================

const nodeManager = new NodeManager(scene, (clickedNodeData) => {
  edgeManager.updateHighlight(nodeManager.selectedId);
});

nodeManager.buildNodes(graphNodes);

// =========================================================
// Edge manager (added after nodes so getNodeCenter works)
// =========================================================

const edgeManager = new EdgeManager(scene, nodeManager);
edgeManager.buildEdges(graphEdges);

// Click on background deselects
glRenderer.domElement.addEventListener('click', (event) => {
  if (event.target === glRenderer.domElement) {
    edgeManager.updateHighlight(null);
  }
});

// =========================================================
// Resize handling
// =========================================================

function onResize() {
  const width = window.innerWidth;
  const height = window.innerHeight;

  glRenderer.setSize(width, height);
  css2dRenderer.setSize(width, height);
  infiniteCanvas.onResize(width, height);
}

window.addEventListener('resize', onResize);

// =========================================================
// Render loop
// =========================================================

function animate() {
  requestAnimationFrame(animate);
  glRenderer.render(scene, camera);
  css2dRenderer.render(scene, camera);
}

animate();

// =========================================================
// Initial view: center the graph roughly at its centroid
// =========================================================

(function centerInitialView() {
  const avgX = graphNodes.reduce((sum, n) => sum + n.x, 0) / graphNodes.length;
  const avgY = graphNodes.reduce((sum, n) => sum + n.y, 0) / graphNodes.length;
  camera.position.x = avgX;
  camera.position.y = avgY;
  camera.zoom = 0.75;
  camera.updateProjectionMatrix();
})();

/**
 * edges.js
 *
 * Builds and manages Three.js edges between nodes.
 * Each edge is a quadratic bezier curve with an arrowhead,
 * plus a CSS2DObject label at the midpoint.
 */

import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

/** Edge type → hex color */
const EDGE_COLORS = {
  'builds-on': 0x94a3b8,
  'uses':       0x60a5fa,
  'optimizes':  0x34d399,
  'enables':    0xfbbf24,
  'applies-to': 0xf87171,
};

const EDGE_OPACITY_DEFAULT = 0.35;
const EDGE_OPACITY_HIGHLIGHTED = 0.9;
const ARROW_SIZE = 14;
const CURVE_SEGMENTS = 48;

/**
 * Compute the quadratic bezier control point for an edge.
 * We push the control point perpendicular to the midpoint to create
 * a slight arc and avoid overlap with other edges.
 */
function computeControlPoint(sourceX, sourceY, targetX, targetY, curvature = 0.25) {
  const midX = (sourceX + targetX) / 2;
  const midY = (sourceY + targetY) / 2;

  // Perpendicular direction
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const length = Math.sqrt(dx * dx + dy * dy) || 1;
  const perpX = -dy / length;
  const perpY = dx / length;

  const offset = length * curvature;
  return new THREE.Vector3(midX + perpX * offset, midY + perpY * offset, 0);
}

/**
 * Sample a quadratic bezier at parameter t in [0, 1].
 */
function bezierPoint(p0, p1, p2, t) {
  const mt = 1 - t;
  return new THREE.Vector3(
    mt * mt * p0.x + 2 * mt * t * p1.x + t * t * p2.x,
    mt * mt * p0.y + 2 * mt * t * p1.y + t * t * p2.y,
    0
  );
}

/**
 * Build an arrowhead as a small triangle mesh.
 * The arrow tip is at `tipPoint`, pointing in `direction`.
 */
function buildArrowhead(tipPoint, direction, color) {
  const perpX = -direction.y;
  const perpY = direction.x;

  const tip = tipPoint;
  const base1 = new THREE.Vector3(
    tip.x - direction.x * ARROW_SIZE + perpX * (ARROW_SIZE * 0.45),
    tip.y - direction.y * ARROW_SIZE + perpY * (ARROW_SIZE * 0.45),
    0
  );
  const base2 = new THREE.Vector3(
    tip.x - direction.x * ARROW_SIZE - perpX * (ARROW_SIZE * 0.45),
    tip.y - direction.y * ARROW_SIZE - perpY * (ARROW_SIZE * 0.45),
    0
  );

  const geometry = new THREE.BufferGeometry();
  const vertices = new Float32Array([
    tip.x,   tip.y,   0,
    base1.x, base1.y, 0,
    base2.x, base2.y, 0,
  ]);
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

  const material = new THREE.MeshBasicMaterial({
    color,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: EDGE_OPACITY_DEFAULT,
    depthTest: false,
  });

  return new THREE.Mesh(geometry, material);
}

/**
 * Build a CSS2DObject label for the edge midpoint.
 */
function buildEdgeLabel(labelText) {
  const div = document.createElement('div');
  div.className = 'edge-label';
  div.textContent = labelText;
  return new CSS2DObject(div);
}

/**
 * EdgeManager creates and manages all edges in the scene.
 */
export class EdgeManager {
  /**
   * @param {THREE.Scene} scene
   * @param {import('./nodes.js').NodeManager} nodeManager
   */
  constructor(scene, nodeManager) {
    this.scene = scene;
    this.nodeManager = nodeManager;

    /** @type {Array<{ line: THREE.Line, arrowMesh: THREE.Mesh, labelObject: CSS2DObject, material: THREE.LineBasicMaterial, arrowMaterial: THREE.MeshBasicMaterial, sourceId: string, targetId: string }>} */
    this.edges = [];
  }

  /**
   * Build all edges from graph data.
   * @param {object[]} edgeDataArray
   */
  buildEdges(edgeDataArray) {
    for (const edgeData of edgeDataArray) {
      const sourceCenter = this.nodeManager.getNodeCenter(edgeData.source);
      const targetCenter = this.nodeManager.getNodeCenter(edgeData.target);

      const p0 = new THREE.Vector3(sourceCenter.x, sourceCenter.y, 0);
      const p2 = new THREE.Vector3(targetCenter.x, targetCenter.y, 0);
      const p1 = computeControlPoint(p0.x, p0.y, p2.x, p2.y, 0.22);

      const hexColor = EDGE_COLORS[edgeData.type] ?? 0x94a3b8;

      // --- Curve line ---
      const points = [];
      for (let i = 0; i <= CURVE_SEGMENTS; i++) {
        points.push(bezierPoint(p0, p1, p2, i / CURVE_SEGMENTS));
      }

      const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
      const lineMaterial = new THREE.LineBasicMaterial({
        color: hexColor,
        transparent: true,
        opacity: EDGE_OPACITY_DEFAULT,
        depthTest: false,
      });
      const line = new THREE.Line(lineGeometry, lineMaterial);
      this.scene.add(line);

      // --- Arrowhead at target end ---
      // Compute direction at t=0.97 (just before the tip) to get arrow direction
      const tipPoint = bezierPoint(p0, p1, p2, 1.0);
      const nearTip = bezierPoint(p0, p1, p2, 0.97);
      const direction = new THREE.Vector3()
        .subVectors(tipPoint, nearTip)
        .normalize();

      const arrowMesh = buildArrowhead(tipPoint, direction, hexColor);
      this.scene.add(arrowMesh);

      // --- Label at midpoint (t=0.5) ---
      const midPoint = bezierPoint(p0, p1, p2, 0.5);
      const labelObject = buildEdgeLabel(edgeData.label);
      labelObject.position.copy(midPoint);
      this.scene.add(labelObject);

      this.edges.push({
        line,
        arrowMesh,
        labelObject,
        material: lineMaterial,
        arrowMaterial: arrowMesh.material,
        sourceId: edgeData.source,
        targetId: edgeData.target,
      });
    }
  }

  /**
   * Highlight edges connected to the selected node, dim others.
   * Pass null to reset all to default opacity.
   * @param {string|null} selectedNodeId
   */
  updateHighlight(selectedNodeId) {
    for (const edge of this.edges) {
      const isConnected =
        selectedNodeId === null ||
        edge.sourceId === selectedNodeId ||
        edge.targetId === selectedNodeId;

      const opacity = isConnected ? EDGE_OPACITY_HIGHLIGHTED : EDGE_OPACITY_DEFAULT * 0.5;
      edge.material.opacity = opacity;
      edge.arrowMaterial.opacity = opacity;
    }
  }
}

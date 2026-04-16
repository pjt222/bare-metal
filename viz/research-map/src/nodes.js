/**
 * nodes.js
 *
 * Builds and manages CSS2DObject node cards for each graph node.
 * Each card is a styled HTML div rendered by CSS2DRenderer,
 * positioned at the node's (x, y) world coordinates.
 */

import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

/** Maps phase identifier to a display label */
const PHASE_LABELS = {
  phase1: 'Phase 1',
  phase2: 'Phase 2',
  phase3: 'Phase 3',
  phase4: 'Phase 4',
  concept: 'Concept',
  hardware: 'Hardware',
};

/**
 * Creates an HTML node card element for a given node data object.
 * @param {object} nodeData - A node from graph-data.js
 * @returns {HTMLElement}
 */
function buildCardElement(nodeData) {
  const card = document.createElement('div');
  card.className = 'node-card';
  card.dataset.id = nodeData.id;
  card.dataset.phase = nodeData.phase;

  // Phase badge
  const badge = document.createElement('div');
  badge.className = 'node-phase-badge';
  badge.textContent = PHASE_LABELS[nodeData.phase] ?? nodeData.phase;
  card.appendChild(badge);

  // Title
  const title = document.createElement('div');
  title.className = 'node-title';
  title.textContent = nodeData.label;
  card.appendChild(title);

  // Summary
  const summary = document.createElement('div');
  summary.className = 'node-summary';
  summary.textContent = nodeData.summary;
  card.appendChild(summary);

  // Formulas
  if (nodeData.formulas && nodeData.formulas.length > 0) {
    const formulasContainer = document.createElement('div');
    formulasContainer.className = 'node-formulas';
    for (const formulaText of nodeData.formulas) {
      const formulaEl = document.createElement('div');
      formulaEl.className = 'node-formula';
      formulaEl.textContent = formulaText;
      formulasContainer.appendChild(formulaEl);
    }
    card.appendChild(formulasContainer);
  }

  // Metrics
  if (nodeData.metrics) {
    const metricsEl = document.createElement('div');
    metricsEl.className = 'node-metrics';
    metricsEl.textContent = nodeData.metrics;
    card.appendChild(metricsEl);
  }

  return card;
}

/**
 * NodeManager creates and tracks all node cards in the scene.
 */
export class NodeManager {
  /**
   * @param {THREE.Scene} scene
   * @param {Function} onNodeClick - Called with (nodeData) when a card is clicked
   */
  constructor(scene, onNodeClick) {
    this.scene = scene;
    this.onNodeClick = onNodeClick;

    /** @type {Map<string, { object: THREE.Object3D, css2d: CSS2DObject, data: object }>} */
    this.nodeMap = new Map();

    this._selectedId = null;
  }

  /**
   * Instantiate all nodes from graph data.
   * @param {object[]} nodes
   */
  buildNodes(nodes) {
    for (const nodeData of nodes) {
      const cardElement = buildCardElement(nodeData);

      // Wire up click
      cardElement.addEventListener('click', (event) => {
        event.stopPropagation();
        this._selectNode(nodeData.id);
        this.onNodeClick(nodeData);
      });

      // CSS2DObject positions the div at a point in 3D space
      const css2dObject = new CSS2DObject(cardElement);
      css2dObject.position.set(nodeData.x, nodeData.y, 0);

      // Invisible Three.js anchor point — needed to keep CSS2DObject in scene
      const anchorGeometry = new THREE.BufferGeometry();
      anchorGeometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3)
      );
      const anchorObject = new THREE.Points(
        anchorGeometry,
        new THREE.PointsMaterial({ size: 0, visible: false })
      );
      anchorObject.position.set(nodeData.x, nodeData.y, 0);
      anchorObject.add(css2dObject);

      this.scene.add(anchorObject);

      this.nodeMap.set(nodeData.id, {
        object: anchorObject,
        css2d: css2dObject,
        element: cardElement,
        data: nodeData,
      });
    }
  }

  /**
   * Returns the world-space center (x, y) of a node by id.
   * @param {string} id
   * @returns {{ x: number, y: number }}
   */
  getNodeCenter(id) {
    const entry = this.nodeMap.get(id);
    if (!entry) return { x: 0, y: 0 };
    return { x: entry.data.x, y: entry.data.y };
  }

  /**
   * Selects or deselects a node visually.
   * @param {string} id
   */
  _selectNode(id) {
    // Deselect previous
    if (this._selectedId !== null) {
      const prev = this.nodeMap.get(this._selectedId);
      if (prev) prev.element.classList.remove('selected');
    }

    if (this._selectedId === id) {
      // Toggle off
      this._selectedId = null;
      return;
    }

    this._selectedId = id;
    const current = this.nodeMap.get(id);
    if (current) current.element.classList.add('selected');
  }

  /** Returns the currently selected node id, or null. */
  get selectedId() {
    return this._selectedId;
  }
}

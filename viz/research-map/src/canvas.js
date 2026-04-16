/**
 * canvas.js
 *
 * Manages the infinite canvas: OrthographicCamera, pan, zoom,
 * touch support, and smooth animated transitions.
 */

import * as THREE from 'three';

const ZOOM_MIN = 0.15;
const ZOOM_MAX = 4.0;
const ZOOM_SPEED = 0.001;
const PAN_BUTTON = 0; // left mouse button for panning background
const MIDDLE_BUTTON = 1;

export class InfiniteCanvas {
  /**
   * @param {HTMLElement} domElement - The element that receives pointer events
   * @param {THREE.OrthographicCamera} camera
   */
  constructor(domElement, camera) {
    this.domElement = domElement;
    this.camera = camera;

    // Internal state
    this._isPanning = false;
    this._panStartPointer = new THREE.Vector2();
    this._panStartCamera = new THREE.Vector2();
    this._targetZoom = camera.zoom;
    this._currentZoom = camera.zoom;
    this._zoomAnimating = false;

    // Track which DOM element was under the pointer at mousedown
    // so we only pan when clicking the background (not a node card)
    this._panAllowed = false;

    this._bindEvents();
  }

  // -------------------------------------------------------
  // Event binding
  // -------------------------------------------------------

  _bindEvents() {
    this.domElement.addEventListener('mousedown', this._onMouseDown.bind(this));
    window.addEventListener('mousemove', this._onMouseMove.bind(this));
    window.addEventListener('mouseup', this._onMouseUp.bind(this));
    this.domElement.addEventListener('wheel', this._onWheel.bind(this), { passive: false });

    // Touch
    this.domElement.addEventListener('touchstart', this._onTouchStart.bind(this), { passive: false });
    this.domElement.addEventListener('touchmove', this._onTouchMove.bind(this), { passive: false });
    this.domElement.addEventListener('touchend', this._onTouchEnd.bind(this));
  }

  // -------------------------------------------------------
  // Mouse handlers
  // -------------------------------------------------------

  _onMouseDown(event) {
    const isLeftClick = event.button === PAN_BUTTON;
    const isMiddleClick = event.button === MIDDLE_BUTTON;

    if (!isLeftClick && !isMiddleClick) return;

    // Only pan when clicking directly on the canvas (not on a node card)
    const targetIsCanvas =
      event.target === this.domElement ||
      event.target.tagName === 'CANVAS';

    if (!targetIsCanvas && !isMiddleClick) return;

    this._isPanning = true;
    this._panAllowed = true;
    this._panStartPointer.set(event.clientX, event.clientY);
    this._panStartCamera.set(this.camera.position.x, this.camera.position.y);

    this.domElement.style.cursor = 'grabbing';
  }

  _onMouseMove(event) {
    if (!this._isPanning || !this._panAllowed) return;

    const deltaX = event.clientX - this._panStartPointer.x;
    const deltaY = event.clientY - this._panStartPointer.y;

    // Convert screen-space delta to world-space delta
    const viewWidth = this.domElement.clientWidth;
    const viewHeight = this.domElement.clientHeight;
    const frustumWidth = (this.camera.right - this.camera.left) / this.camera.zoom;
    const frustumHeight = (this.camera.top - this.camera.bottom) / this.camera.zoom;

    const worldDeltaX = -(deltaX / viewWidth) * frustumWidth;
    const worldDeltaY = (deltaY / viewHeight) * frustumHeight;

    this.camera.position.x = this._panStartCamera.x + worldDeltaX;
    this.camera.position.y = this._panStartCamera.y + worldDeltaY;
  }

  _onMouseUp() {
    this._isPanning = false;
    this._panAllowed = false;
    this.domElement.style.cursor = '';
  }

  // -------------------------------------------------------
  // Wheel handler (zoom)
  // -------------------------------------------------------

  _onWheel(event) {
    event.preventDefault();

    const zoomDelta = -event.deltaY * ZOOM_SPEED;
    this._targetZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, this._targetZoom * (1 + zoomDelta * 2)));

    // Zoom toward the cursor position
    this._zoomTowardPointer(event.clientX, event.clientY, this._targetZoom);
  }

  /**
   * Adjusts camera position so that the world point under the cursor
   * stays fixed as we zoom.
   */
  _zoomTowardPointer(clientX, clientY, newZoom) {
    const viewWidth = this.domElement.clientWidth;
    const viewHeight = this.domElement.clientHeight;

    // Normalized device coords [-1, 1]
    const ndcX = (clientX / viewWidth) * 2 - 1;
    const ndcY = -(clientY / viewHeight) * 2 + 1;

    // World position under cursor before zoom
    const frustumWidthBefore = (this.camera.right - this.camera.left) / this.camera.zoom;
    const frustumHeightBefore = (this.camera.top - this.camera.bottom) / this.camera.zoom;
    const worldXBefore = this.camera.position.x + ndcX * (frustumWidthBefore / 2);
    const worldYBefore = this.camera.position.y + ndcY * (frustumHeightBefore / 2);

    // Apply zoom
    this.camera.zoom = newZoom;
    this.camera.updateProjectionMatrix();

    // World position under cursor after zoom
    const frustumWidthAfter = (this.camera.right - this.camera.left) / newZoom;
    const frustumHeightAfter = (this.camera.top - this.camera.bottom) / newZoom;
    const worldXAfter = this.camera.position.x + ndcX * (frustumWidthAfter / 2);
    const worldYAfter = this.camera.position.y + ndcY * (frustumHeightAfter / 2);

    // Shift camera to compensate
    this.camera.position.x += worldXBefore - worldXAfter;
    this.camera.position.y += worldYBefore - worldYAfter;
  }

  // -------------------------------------------------------
  // Touch handlers
  // -------------------------------------------------------

  _onTouchStart(event) {
    event.preventDefault();
    if (event.touches.length === 1) {
      // Single-finger pan
      this._isPanning = true;
      this._panAllowed = true;
      this._panStartPointer.set(event.touches[0].clientX, event.touches[0].clientY);
      this._panStartCamera.set(this.camera.position.x, this.camera.position.y);
    } else if (event.touches.length === 2) {
      // Pinch-zoom start
      this._isPanning = false;
      this._touchPinchStart = this._pinchDistance(event.touches);
      this._touchZoomStart = this.camera.zoom;
      this._touchMidStart = this._touchMidpoint(event.touches);
    }
  }

  _onTouchMove(event) {
    event.preventDefault();
    if (event.touches.length === 1 && this._isPanning) {
      const deltaX = event.touches[0].clientX - this._panStartPointer.x;
      const deltaY = event.touches[0].clientY - this._panStartPointer.y;

      const viewWidth = this.domElement.clientWidth;
      const viewHeight = this.domElement.clientHeight;
      const frustumWidth = (this.camera.right - this.camera.left) / this.camera.zoom;
      const frustumHeight = (this.camera.top - this.camera.bottom) / this.camera.zoom;

      this.camera.position.x = this._panStartCamera.x - (deltaX / viewWidth) * frustumWidth;
      this.camera.position.y = this._panStartCamera.y + (deltaY / viewHeight) * frustumHeight;
    } else if (event.touches.length === 2) {
      const currentDist = this._pinchDistance(event.touches);
      const scale = currentDist / this._touchPinchStart;
      const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, this._touchZoomStart * scale));
      const mid = this._touchMidpoint(event.touches);
      this._zoomTowardPointer(mid.x, mid.y, newZoom);
    }
  }

  _onTouchEnd(event) {
    if (event.touches.length === 0) {
      this._isPanning = false;
      this._panAllowed = false;
    }
  }

  _pinchDistance(touches) {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  _touchMidpoint(touches) {
    return {
      x: (touches[0].clientX + touches[1].clientX) / 2,
      y: (touches[0].clientY + touches[1].clientY) / 2,
    };
  }

  // -------------------------------------------------------
  // Focus on a world-space point (animated)
  // -------------------------------------------------------

  focusOn(worldX, worldY, zoom = 1.0, durationMs = 400) {
    const startX = this.camera.position.x;
    const startY = this.camera.position.y;
    const startZoom = this.camera.zoom;
    const endZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom));
    const startTime = performance.now();

    const animate = (now) => {
      const elapsed = now - startTime;
      const t = Math.min(elapsed / durationMs, 1);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - t, 3);

      this.camera.position.x = startX + (worldX - startX) * ease;
      this.camera.position.y = startY + (worldY - startY) * ease;
      this.camera.zoom = startZoom + (endZoom - startZoom) * ease;
      this.camera.updateProjectionMatrix();

      if (t < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }

  // -------------------------------------------------------
  // Handle window resize
  // -------------------------------------------------------

  onResize(width, height) {
    const halfW = width / 2;
    const halfH = height / 2;
    this.camera.left = -halfW;
    this.camera.right = halfW;
    this.camera.top = halfH;
    this.camera.bottom = -halfH;
    this.camera.updateProjectionMatrix();
  }
}

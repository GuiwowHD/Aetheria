/**
 * Aetheria — input aggregation.
 *
 * One place that turns mice, touch, pens, keyboards and gyroscopes into the six
 * verbs the renderer understands: orbit, zoom, slice, ignite, toggle, reset.
 * Keeping unification here means the renderers stay free of device branching and
 * the UI can advertise exact key bindings.
 *
 * Notable behaviours:
 *   - A tap ignites a supernova, but only if the pointer moved less than 6 px and
 *     was released within 400 ms, so dragging never fires one by accident.
 *   - Two-finger pinch zooms (distance) and slices the hyperplane (vertical
 *     centroid motion) simultaneously, which is the only sane way to expose both
 *     on a phone.
 *   - Device orientation is used only when the platform actually reports it
 *     (iOS requires an explicit permission request, handled by `enableGyro`).
 */

export interface InputHandlers {
  orbit(dx: number, dy: number, plane?: 'xw' | 'yw' | 'zw'): void;
  zoom(delta: number): void;
  slice(delta: number): void;
  ignite(ndcX: number, ndcY: number): void;
  key(action: KeyAction): void;
  /** Fired on any interaction, used to wake the auto-hiding UI. */
  activity(): void;
}

export type KeyAction =
  | 'pause'
  | 'reset'
  | 'toggle-ui'
  | 'export-png'
  | 'record'
  | 'help'
  | 'plane-xw'
  | 'plane-yw'
  | 'plane-zw'
  | 'none';

export class InputController {
  private el: HTMLElement;
  private handlers: InputHandlers;
  private pointers = new Map<number, { x: number; y: number; t: number; moved: number }>();
  private activeDrag: number | null = null;
  private pinchDist = 0;
  private pinchY = 0;
  private mode: 'drag' | 'plane' = 'drag';
  private planeKey: 'xw' | 'yw' | 'zw' | undefined;
  private gyroEnabled = false;
  private gyroBase: { beta: number; gamma: number } | null = null;
  private gyroHandler: ((ev: DeviceOrientationEvent) => void) | null = null;
  private keyHandler: ((ev: KeyboardEvent) => void) | null = null;
  private disposers: (() => void)[] = [];

  constructor(el: HTMLElement, handlers: InputHandlers) {
    this.el = el;
    this.handlers = handlers;
    this.attach();
  }

  private toNdc(clientX: number, clientY: number): { x: number; y: number } {
    const r = this.el.getBoundingClientRect();
    return {
      x: ((clientX - r.left) / Math.max(1, r.width)) * 2 - 1,
      y: -(((clientY - r.top) / Math.max(1, r.height)) * 2 - 1),
    };
  }

  private on<K extends keyof HTMLElementEventMap>(
    target: HTMLElement | Window | Document,
    type: K | string,
    fn: (ev: never) => void,
    opts?: AddEventListenerOptions
  ): void {
    target.addEventListener(type as string, fn as EventListener, opts);
    this.disposers.push(() => target.removeEventListener(type as string, fn as EventListener));
  }

  private attach(): void {
    const el = this.el;

    // ---- pointer (mouse + pen + touch unified) ---------------------------
    this.on(el, 'pointerdown', (ev: PointerEvent) => {
      el.setPointerCapture?.(ev.pointerId);
      this.pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY, t: performance.now(), moved: 0 });
      this.activeDrag = ev.pointerId;
      if (this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        this.pinchDist = Math.hypot((a?.x ?? 0) - (b?.x ?? 0), (a?.y ?? 0) - (b?.y ?? 0));
        this.pinchY = ((a?.y ?? 0) + (b?.y ?? 0)) / 2;
      }
      this.handlers.activity();
    });

    this.on(el, 'pointermove', (ev: PointerEvent) => {
      const rec = this.pointers.get(ev.pointerId);
      if (!rec) return;
      const dx = ev.clientX - rec.x;
      const dy = ev.clientY - rec.y;
      rec.moved += Math.abs(dx) + Math.abs(dy);
      rec.x = ev.clientX;
      rec.y = ev.clientY;

      if (this.pointers.size >= 2) {
        const pts = [...this.pointers.values()];
        const a = pts[0]!;
        const b = pts[1]!;
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        const midY = (a.y + b.y) / 2;
        if (this.pinchDist > 0) this.handlers.zoom((dist - this.pinchDist) * 2.4);
        this.handlers.slice(-(midY - this.pinchY) * 3.2);
        this.pinchDist = dist;
        this.pinchY = midY;
        return;
      }

      // Single pointer drag: 4D rotation. Shift promotes the drag to the
      // (y,w)/(z,w) planes, which are the ones that read as "the fourth axis".
      this.handlers.orbit(dx, dy, this.planeKey);
      // Wheel-style slice is available from the keyboard; drag stays rotation.
      this.mode = this.planeKey ? 'plane' : 'drag';
      void this.mode;
    });

    const release = (ev: PointerEvent) => {
      const rec = this.pointers.get(ev.pointerId);
      this.pointers.delete(ev.pointerId);
      if (this.activeDrag === ev.pointerId) this.activeDrag = null;
      if (this.pointers.size < 2) this.pinchDist = 0;
      if (!rec) return;
      const quick = performance.now() - rec.t < 400;
      if (quick && rec.moved < 6 && !this.planeKey) {
        const ndc = this.toNdc(ev.clientX, ev.clientY);
        this.handlers.ignite(ndc.x, ndc.y);
      }
    };
    this.on(el, 'pointerup', release);
    this.on(el, 'pointercancel', release);
    this.on(el, 'lostpointercapture', release);

    // ---- wheel: zoom, with shift/ctrl for hyperplane slicing -------------
    this.on(
      el,
      'wheel',
      (ev: WheelEvent) => {
        ev.preventDefault();
        if (ev.shiftKey) this.handlers.slice(ev.deltaY);
        else this.handlers.zoom(ev.deltaY);
        this.handlers.activity();
      },
      { passive: false }
    );

    // Safari still fires gesture events for trackpad pinch.
    this.on(el, 'gesturestart', (ev: Event) => ev.preventDefault());
    this.on(el, 'contextmenu', (ev: Event) => ev.preventDefault());
    this.on(el, 'touchstart', (ev: TouchEvent) => {
      // Prevent the page from scrolling/zooming under the canvas on iOS.
      if (ev.touches.length > 1) ev.preventDefault();
    }, { passive: false });

    // ---- keyboard --------------------------------------------------------
    this.keyHandler = (ev: KeyboardEvent) => {
      const target = ev.target as HTMLElement | null;
      if (target && /INPUT|TEXTAREA|SELECT/.test(target.tagName)) return;
      const step = ev.shiftKey ? 26 : 11;
      let action: KeyAction = 'none';
      switch (ev.key.toLowerCase()) {
        case 'w':
        case 'arrowup':
          this.handlers.orbit(0, -step, ev.altKey ? 'yw' : undefined);
          break;
        case 's':
        case 'arrowdown':
          if (ev.key.toLowerCase() === 's' && !ev.altKey && !ev.shiftKey && !ev.ctrlKey) action = 'export-png';
          else this.handlers.orbit(0, step, ev.altKey ? 'yw' : undefined);
          break;
        case 'a':
        case 'arrowleft':
          this.handlers.orbit(-step, 0, ev.altKey ? 'xw' : undefined);
          break;
        case 'd':
        case 'arrowright':
          this.handlers.orbit(step, 0, ev.altKey ? 'xw' : undefined);
          break;
        case 'q':
          this.handlers.slice(-40);
          break;
        case 'e':
          this.handlers.slice(40);
          break;
        case '+':
        case '=':
          this.handlers.zoom(-90);
          break;
        case '-':
        case '_':
          this.handlers.zoom(90);
          break;
        case ' ':
          action = 'pause';
          break;
        case 'r':
          action = 'reset';
          break;
        case 'h':
          action = 'toggle-ui';
          break;
        case 'v':
          action = 'record';
          break;
        case 'x':
          action = 'plane-xw';
          break;
        case 'y':
          action = 'plane-yw';
          break;
        case 'z':
          action = 'plane-zw';
          break;
        case '?':
        case 'f1':
          action = 'help';
          break;
        default:
          return;
      }
      this.handlers.activity();
      if (action !== 'none') {
        if (action === 'pause' || action === 'help') ev.preventDefault();
        this.handlers.key(action);
      }
    };
    window.addEventListener('keydown', this.keyHandler);
    this.disposers.push(() => window.removeEventListener('keydown', this.keyHandler!));

    // ---- device orientation ---------------------------------------------
    this.gyroHandler = (ev: DeviceOrientationEvent) => {
      if (ev.beta === null || ev.gamma === null) return;
      if (!this.gyroBase) this.gyroBase = { beta: ev.beta, gamma: ev.gamma };
      const dBeta = ev.beta - this.gyroBase.beta;
      const dGamma = ev.gamma - this.gyroBase.gamma;
      // Slow follow so the view glides instead of jittering with hand tremor.
      this.handlers.orbit(dGamma * 0.012, dBeta * 0.012, undefined);
    };
  }

  setPlane(plane: 'xw' | 'yw' | 'zw' | undefined): void {
    this.planeKey = plane;
  }

  get plane(): 'xw' | 'yw' | 'zw' | undefined {
    return this.planeKey;
  }

  get usingGyro(): boolean {
    return this.gyroEnabled;
  }

  /** iOS 13+ requires a user-gesture-triggered permission prompt. */
  async enableGyro(): Promise<boolean> {
    if (this.gyroEnabled) return true;
    const DOE = window.DeviceOrientationEvent as unknown as
      | { requestPermission?: () => Promise<'granted' | 'denied'> }
      | undefined;
    try {
      if (DOE && typeof DOE.requestPermission === 'function') {
        const res = await DOE.requestPermission();
        if (res !== 'granted') return false;
      }
      if (!('DeviceOrientationEvent' in window)) return false;
      window.addEventListener('deviceorientation', this.gyroHandler!);
      this.gyroEnabled = true;
      return true;
    } catch {
      return false;
    }
  }

  disableGyro(): void {
    if (this.gyroHandler) window.removeEventListener('deviceorientation', this.gyroHandler);
    this.gyroEnabled = false;
    this.gyroBase = null;
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.disableGyro();
    this.pointers.clear();
  }
}

import {
  AlphaType,
  Canvas,
  ColorType,
  Fill,
  FilterMode,
  ImageShader,
  MipmapMode,
  Shader,
  Skia,
  useImage,
  type SkImage,
} from '@shopify/react-native-skia';
import { useEffect, useMemo, useRef } from 'react';
import { StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  useAnimatedStyle,
  useFrameCallback,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

// ---------------------------------------------------------------------------
// TUNABLE PARAMETERS
// All of the "feel" of the water lives here. Austen fine-tunes on-device.
// ---------------------------------------------------------------------------

// --- Simulation grid -------------------------------------------------------

/** Width of the simulation grid in cells. Height is derived from screen aspect.
 *  Higher = finer ripples but more CPU per frame. 120 is a good 60fps target. */
const GRID_WIDTH = 120;

/** Energy retained each wave step (0..1). Lower = ripples die faster.
 *  0.95 matches the reference: damping = mix(0.8, 0.999, decay=0.75) ≈ 0.95. */
const DAMPING = 0.95;

/** Height pushed into the water per stamp (sim units). Negative = the finger
 *  dents the surface downward. Scaled by finger speed (see velocity gating). */
const IMPULSE_STRENGTH = -6.0;

/** Radius of the finger "brush" in grid cells. Larger = fatter wake. */
const BRUSH_RADIUS = 3.5;

/** Maps raw sim height into the 0..1 stored in the height texture.
 *  Keep IMPULSE_STRENGTH * HEIGHT_SCALE comfortably inside ~0.45. */
const HEIGHT_SCALE = 0.07;

// --- Velocity-gated ripple injection (matches reference) -------------------

/** Minimum finger travel per frame (grid cells) before ANY ripple is injected.
 *  Below this the water is untouched — a still fingertip makes no waves,
 *  mirroring the reference's `mouseSpeed > 0.0001` gate. */
const VELOCITY_THRESHOLD = 0.1;

/** Maps finger speed (grid cells/frame) to an impulse multiplier in 0..1.
 *  amount = clamp(speed * VELOCITY_GAIN, 0, 1). Higher = fainter drags ripple. */
const VELOCITY_GAIN = 0.4;

// --- Refraction / optics ---------------------------------------------------

/** Screen-px the surface gradient bends the image beneath (refraction strength). */
const REFRACTION_STRENGTH = 70.0;

/** Screen-px radius of the cheap disturbance blur where the water is choppy. */
const BLUR_STRENGTH = 4.0;

/** Chromatic aberration: R/B channels split along the gradient, scaled by this
 *  and the refraction. Reference chromaticDispersion ≈ 0.25. */
const CHROMATIC_DISPERSION = 0.25;

/** Specular glint strength on wave slopes. Reference lightingIntensity ≈ 0.21. */
const LIGHTING_INTENSITY = 0.21;

/** How much choppy water darkens the image (0..1). */
const SHADING_STRENGTH = 0.1;

// --- Blue tint (reference color rgb 0, 0.427, 0.827) -----------------------

/** Subtle water tint colour (reference `color`). */
const TINT_COLOR = [0.0, 0.427, 0.827] as const;

/** Overall tint strength. Kept low — most of it lands on disturbed water. */
const TINT_STRENGTH = 0.06;

// --- Ambient pool caustics + surface sway (always on, even with no touch) --
// ONE animated 3D noise field drives BOTH effects, mirroring the reference:
// its scalar value -> the faint caustic light, its gradient -> the gentle,
// spatially-varying surface warp. Domain warping (re-sampling offset by the
// previous sample's gradient) folds smooth noise into the caustic "web".

/** Caustic light colour (reference caustic tint ≈ 0.42, 0.82, 1.0). */
const CAUSTIC_COLOR = [0.4196, 0.8235, 1.0] as const;

/** Faint caustic light strength. VERY subtle — ambient pool shimmer, not a
 *  light show. Reference keeps the caustic itself faint under a 0.74 mix. */
const CAUSTIC_STRENGTH = 0.6;

/** Spatial frequency of the caustic field (roughly cycles across screen width).
 *  Higher = smaller, busier cells. Reference scales uv up by ~16 * 0.446. */
const CAUSTIC_SCALE = 3.5;

/** Drift speed of the field through time. Reference feeds `uTime * 0.25`. */
const CAUSTIC_SPEED = 0.25;

/** Domain-warp strength: how far the previous sample's gradient displaces the
 *  next sample point (in noise-space units). This folding is what turns smooth
 *  noise into the interconnected caustic web. Reference folds twice. */
const DOMAIN_WARP_STRENGTH = 0.35;

/** Peak screen-px the field gradient warps the image on its own — the gentle,
 *  spatially-varying "pool surface swaying" idle motion. Because it's the noise
 *  gradient, it differs everywhere and drifts with time (unlike a global sine).
 *  Added to the refraction offset only, so it never triggers choppy blur. */
const AMBIENT_DISTORTION_PX = 25.0;

// --- Background image rotation ---------------------------------------------

/** How long the very first image holds before the first crossfade (ms). */
const FIRST_HOLD_MS = 18000;

/** How long each subsequent image holds before crossfading (ms). */
const SLIDE_HOLD_MS = 5000;

/** Crossfade duration between images (ms). */
const CROSSFADE_MS = 300;

/** Number of hero images / slides in the rotation. */
const SLIDE_COUNT = 3;

// ---------------------------------------------------------------------------
// OVERLAY TUNABLES — the hero text + bottom progress bar rendered in the RN
// view tree ABOVE the Skia <Canvas>, so the water never distorts them.
// Sizes target a ~1080px-wide phone (Pixel 10 Pro XL); Austen fine-tunes
// exact px on-device.
// ---------------------------------------------------------------------------

// --- Overlay 1: centered hero headline -------------------------------------

const HERO_LINE_1 = 'ONE DREAM';
const HERO_LINE_2 = 'AT A TIME';
/** Big hero headline size. amaliproperties.com uses "aviano-sans" (a licensed
 *  commercial font we can't ship) — this uses the platform default sans-serif
 *  at weight '300' as a visual substitute. */
const HERO_FONT_SIZE = 48;
const HERO_LINE_HEIGHT = 60;
const HERO_LETTER_SPACING = 2;
const HERO_COLOR = '#FFFFFF';

// --- Overlay 2: bottom "DISCOVER [Residences|Island|Villa] LIVING" bar ------
// Matches amaliproperties.com's desktop hero bar: a translucent rounded pill
// holding the three slide labels (active = white, the other two faded), flanked
// by a static "DISCOVER" / "LIVING". A very faint wash sweeps across the pill
// as the current slide progresses.

/** Static words either side of the pill. Reads "DISCOVER <slide> LIVING". */
const SIDE_LABEL_LEFT = 'DISCOVER';
const SIDE_LABEL_RIGHT = 'LIVING';
/** Slide labels inside the pill — order matches the image rotation. */
const TAB_LABELS = ['RESIDENCES', 'ISLAND', 'VILLA'] as const;

/** px the bar sits above the bottom safe-area inset. */
const BAR_BOTTOM_OFFSET = 40;
/** Gap between DISCOVER / pill / LIVING. */
const BAR_GAP = 14;
/** Shared text style for every label in the bar. */
const BAR_FONT_SIZE = 12;
const BAR_LETTER_SPACING = 1.2;
const BAR_TEXT_COLOR = '#FFFFFF';
/** Opacity of the two non-active slide tabs. */
const TAB_INACTIVE_OPACITY = 0.35;
/** Fade duration when the active tab changes. */
const TAB_FADE_MS = 400;
/** The translucent pill behind the slide labels. */
const PILL_BG = 'rgba(255,255,255,0.08)';
const PILL_RADIUS = 10;
const PILL_PADDING_H = 12;
/** Tall vertical padding gives the pill its height. */
const TAB_PADDING_V = 16;
const TAB_PADDING_H = 12;
/** The faint wash that sweeps across the pill as the slide progresses. */
const PILL_SWEEP_BG = 'rgba(255,255,255,0.10)';

// ---------------------------------------------------------------------------
// SkSL runtime shader.
// Child 0 = current background, child 1 = next background (cover-fit),
// child 2 = height field (grid-pixel space).
// `fade` (0..1) crossfades current->next BEFORE refraction so the water
// refracts the blended result. `uTime` drives the ambient caustics.
// ---------------------------------------------------------------------------

const source = Skia.RuntimeEffect.Make(`
uniform shader image;      // current background, cover-fit (canvas space)
uniform shader imageNext;  // next background, cover-fit (canvas space)
uniform shader heightMap;  // height field, sampled in grid-pixel space
uniform float2 resolution; // screen size in px
uniform float2 gridSize;   // grid dimensions in cells
uniform float refraction;  // px offset per unit gradient
uniform float blurAmount;  // px blur radius at max disturbance
uniform float lighting;    // specular strength
uniform float shading;     // darkening strength
uniform float chromatic;   // chromatic dispersion
uniform float tintStrength;
uniform float3 tintColor;
uniform float causticStrength;
uniform float causticScale;
uniform float causticSpeed;
uniform float3 causticColor;
uniform float domainWarp;       // domain-warp fold strength (noise-space units)
uniform float ambientDistortion; // px of always-on spatially-varying surface warp
uniform float fade;        // 0..1 crossfade current -> next
uniform float uTime;       // seconds, drives caustic drift + ambient motion

// Decode height stored (biased around 0.5) in the red channel.
float sampleH(float2 g) {
  return heightMap.eval(g).r - 0.5;
}

// Blended background sample. Skips the second texture fetch when not fading.
half4 bg(float2 p) {
  half4 c = image.eval(p);
  if (fade > 0.0) {
    c = mix(c, imageNext.eval(p), half(fade));
  }
  return c;
}

// --- Cheap 3D value noise; scalar value + gradient via finite differences --
// The reference uses an analytic-derivative BCC noise (vec4: xyz = gradient,
// w = value). We don't need that beast: a cheap value noise plus a
// finite-difference gradient reproduces the same "value + gradient" field the
// domain warping and the two ambient effects need.
float hash13(float3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

float vnoise(float3 x) {
  float3 i = floor(x);
  float3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  float n000 = hash13(i + float3(0.0, 0.0, 0.0));
  float n100 = hash13(i + float3(1.0, 0.0, 0.0));
  float n010 = hash13(i + float3(0.0, 1.0, 0.0));
  float n110 = hash13(i + float3(1.0, 1.0, 0.0));
  float n001 = hash13(i + float3(0.0, 0.0, 1.0));
  float n101 = hash13(i + float3(1.0, 0.0, 1.0));
  float n011 = hash13(i + float3(0.0, 1.0, 1.0));
  float n111 = hash13(i + float3(1.0, 1.0, 1.0));
  float nx00 = mix(n000, n100, f.x);
  float nx10 = mix(n010, n110, f.x);
  float nx01 = mix(n001, n101, f.x);
  float nx11 = mix(n011, n111, f.x);
  float nxy0 = mix(nx00, nx10, f.y);
  float nxy1 = mix(nx01, nx11, f.y);
  return mix(nxy0, nxy1, f.z);
}

// Sample the noise and its xy gradient (time held fixed) via forward
// differences. Returns float3(dValue/dx, dValue/dy, value). 3 noise taps.
float3 noiseGrad(float3 p) {
  const float e = 0.06; // finite-difference step in noise space
  float c = vnoise(p);
  float nx = vnoise(p + float3(e, 0.0, 0.0));
  float ny = vnoise(p + float3(0.0, e, 0.0));
  return float3((nx - c) / e, (ny - c) / e, c);
}

// Animated caustic field with two domain-warp folds (matches the reference's
// base -> balance -> final re-sampling). Each fold offsets the sample point by
// the previous sample's gradient, folding smooth noise into the caustic web.
// Returns float3(gradient.x, gradient.y, value). 9 noise taps total.
float3 causticField(float2 aspectUv) {
  float3 p = float3(aspectUv * causticScale, uTime * causticSpeed);
  float3 n = noiseGrad(p);
  p.xy -= n.xy * domainWarp;      // fold 1 (like balanceNoise)
  n = noiseGrad(p);
  p.xy -= n.xy * domainWarp;      // fold 2 (like final noise)
  n = noiseGrad(p);
  return n;
}

half4 main(float2 xy) {
  float2 uv = xy / resolution;
  float2 g = uv * gridSize;

  // Central-difference gradient of the surface (one cell each way).
  float hL = sampleH(g - float2(1.0, 0.0));
  float hR = sampleH(g + float2(1.0, 0.0));
  float hU = sampleH(g - float2(0.0, 1.0));
  float hD = sampleH(g + float2(0.0, 1.0));
  float2 grad = float2(hR - hL, hD - hU);

  // Animated caustic field, evaluated once. Its gradient drives the ambient
  // surface sway; its scalar value drives the faint caustic light below.
  float2 aspectUv = float2(uv.x * (resolution.x / resolution.y), uv.y);
  float3 field = causticField(aspectUv);

  // Ambient surface sway — always on, even with no touch. It's the noise
  // gradient, so it's spatially varying (different regions warp in different
  // directions) and drifts with time, like a pool surface very gently swaying.
  // Added to the refraction offset only (NOT into grad), so it warps the image
  // without triggering the choppy-water blur/darkening.
  float2 amb = field.xy * ambientDistortion;

  float2 sampleXY = xy + grad * refraction + amb;
  float disturb = clamp(length(grad) * 8.0, 0.0, 1.0);
  half hd = half(disturb);

  half4 col;
  if (disturb > 0.003) {
    // Chromatic aberration: split R/B along the gradient (green stays put).
    float2 chroma = grad * refraction * chromatic;
    col = half4(
      bg(sampleXY + chroma).r,
      bg(sampleXY).g,
      bg(sampleXY - chroma).b,
      1.0
    );

    // Cheap 4-tap ring blur that only kicks in on choppy water.
    if (blurAmount > 0.0) {
      float r = blurAmount * disturb;
      half4 ring = bg(sampleXY + float2(r, 0.0))
                 + bg(sampleXY - float2(r, 0.0))
                 + bg(sampleXY + float2(0.0, r))
                 + bg(sampleXY - float2(0.0, r));
      col = mix(col, (col + ring) / 5.0, hd);
    }
  } else {
    // Calm water: a single blended sample (fast path for most of the screen).
    col = bg(sampleXY);
  }

  // Faint pool caustics: the field's scalar value, sharpened with pow(.,2)
  // (matching the reference's normalized = pow(0.5 + 0.5*noise.w, 2.0)), then
  // screen-blended in the caustic colour. Kept VERY subtle — ambient shimmer.
  float caustic = pow(clamp(field.z, 0.0, 1.0), 2.0);
  half3 causticCol = half3(causticColor) * half(caustic * causticStrength);
  col.rgb = half3(1.0) - (half3(1.0) - col.rgb) * (half3(1.0) - causticCol);

  // Specular glint from slopes facing a virtual top-left light.
  float spec = clamp((-grad.x - grad.y) * 0.5, 0.0, 1.0);
  col.rgb += half3(spec * lighting);

  // Choppy water reads slightly darker, like a broken reflection.
  col.rgb *= half(1.0 - disturb * shading);

  // Subtle blue tint, mostly on disturbed water plus a whisper globally.
  half tintAmt = half(tintStrength) * (half(0.3) + half(0.7) * hd);
  col.rgb = mix(col.rgb, half3(tintColor), tintAmt);

  return col;
}
`);

export function WaterSurface() {
  const { width, height } = useWindowDimensions();

  // The three hero images, rotated forever under the water.
  const img0 = useImage(require('@/assets/images/amali-residences.webp'));
  const img1 = useImage(require('@/assets/images/amali-island.jpeg'));
  const img2 = useImage(require('@/assets/images/amali-villas.jpeg'));

  // Grid dimensions: keep cells roughly square by matching screen aspect.
  const gridW = GRID_WIDTH;
  const gridH = Math.max(2, Math.round(GRID_WIDTH * (height / width)));
  const cellCount = gridW * gridH;

  // Simulation state lives in a ref only ever touched on the UI thread inside
  // the frame worklet (never during render). Buffers are allocated lazily on
  // the first frame and reused (mutated in place) every frame after.
  const buffersRef = useRef<{
    current: Float32Array;
    previous: Float32Array;
    rgba: Uint8Array;
  } | null>(null);

  // A flat (undisturbed) height image so the shader always has a valid child.
  const flatImage = useMemo(() => {
    const rgba = new Uint8Array(cellCount * 4);
    for (let i = 0; i < cellCount; i++) {
      rgba[i * 4] = 128; // 0.5 -> zero height
      rgba[i * 4 + 3] = 255;
    }
    const data = Skia.Data.fromBytes(rgba);
    return Skia.Image.MakeImage(
      { width: gridW, height: gridH, colorType: ColorType.RGBA_8888, alphaType: AlphaType.Opaque },
      data,
      gridW * 4,
    );
  }, [cellCount, gridW, gridH]);

  // The per-frame height texture handed to the shader.
  const heightImage = useSharedValue(flatImage);

  // Background rotation state, driven entirely on the UI thread (no per-frame
  // React state). currentImg/nextImg/fade are updated together in the frame
  // worklet from a pure function of elapsed time, so the crossfade endpoint
  // and the index swap are always consistent — no flash on wrap. They are the
  // worklet's alone to write (the React Compiler forbids writing a value that
  // an effect also touches), so we seed them with the flat placeholder and let
  // the effect only publish the loaded image list + flip `ready`.
  const imagesSV = useSharedValue<(SkImage | null)[] | null>(null);
  const currentImg = useSharedValue<SkImage | null>(flatImage);
  const nextImg = useSharedValue<SkImage | null>(flatImage);

  // Overlay sync: the frame worklet is the sole writer of these two, derived
  // as a pure function of the same elapsed-time clock that drives the image
  // rotation, so the overlays never drift from the background. Never written
  // in an effect / read in a conflicting hook (React Compiler rule) — the
  // overlays only READ them through useAnimatedStyle on the UI thread.
  const slideIndex = useSharedValue(0); // active slide, 0..SLIDE_COUNT-1
  const slideProgress = useSharedValue(0); // 0->1 through the current slide

  // Readiness is derived during render (no state, no cascading renders); the
  // effect only publishes the loaded image list to the worklet.
  const ready = !!(img0 && img1 && img2);

  useEffect(() => {
    if (img0 && img1 && img2) {
      imagesSV.value = [img0, img1, img2];
    }
  }, [img0, img1, img2, imagesSV]);

  // Touch bridge (gesture worklets -> frame worklet, all on the UI thread).
  const touchX = useSharedValue(0);
  const touchY = useSharedValue(0);
  const prevX = useSharedValue(0);
  const prevY = useSharedValue(0);
  const touching = useSharedValue(0);

  // The uniforms object lives in a single shared value the frame worklet owns
  // (the only writer), mirroring the heightImage pattern. fade + uTime are
  // refreshed every frame on the UI thread — no React re-render, and the
  // React Compiler is happy because nothing reads these in a hook.
  const uniforms = useSharedValue({
    resolution: [width, height],
    gridSize: [gridW, gridH],
    refraction: REFRACTION_STRENGTH,
    blurAmount: BLUR_STRENGTH,
    lighting: LIGHTING_INTENSITY,
    shading: SHADING_STRENGTH,
    chromatic: CHROMATIC_DISPERSION,
    tintStrength: TINT_STRENGTH,
    tintColor: TINT_COLOR as unknown as number[],
    causticStrength: CAUSTIC_STRENGTH,
    causticScale: CAUSTIC_SCALE,
    causticSpeed: CAUSTIC_SPEED,
    causticColor: CAUSTIC_COLOR as unknown as number[],
    domainWarp: DOMAIN_WARP_STRENGTH,
    ambientDistortion: AMBIENT_DISTORTION_PX,
    fade: 0,
    uTime: 0,
  });

  const fullScreenRect = useMemo(
    () => ({ x: 0, y: 0, width, height }),
    [width, height],
  );

  const pan = Gesture.Pan()
    .minDistance(0)
    .onBegin((e) => {
      'worklet';
      prevX.value = e.x;
      prevY.value = e.y;
      touchX.value = e.x;
      touchY.value = e.y;
      touching.value = 1;
    })
    .onUpdate((e) => {
      'worklet';
      touchX.value = e.x;
      touchY.value = e.y;
    })
    .onFinalize(() => {
      'worklet';
      touching.value = 0;
    });

  useFrameCallback((frameInfo) => {
    'worklet';
    // Lazily allocate (or resize) the simulation buffers on the UI thread.
    if (buffersRef.current === null || buffersRef.current.rgba.length !== cellCount * 4) {
      buffersRef.current = {
        current: new Float32Array(cellCount),
        previous: new Float32Array(cellCount),
        rgba: new Uint8Array(cellCount * 4),
      };
    }
    const buffers = buffersRef.current;
    const current = buffers.current;
    const previous = buffers.previous;
    const rgba = buffers.rgba;

    // --- 0. Clock + background rotation (pure function of elapsed time).
    // idx/fadeValue drive the crossfade; slideIndex/slideProgress feed the
    // overlays. All four come from this one clock so they stay in lockstep.
    // The first slide holds FIRST_HOLD_MS, the rest SLIDE_HOLD_MS, so slide 0's
    // slot (and thus its progress fill duration) is longer than the others.
    const tms = frameInfo.timeSinceFirstFrame;
    let fadeValue = 0;
    let idx = 0;
    let progressValue = 0;

    const firstSlot = FIRST_HOLD_MS + CROSSFADE_MS;
    if (tms < firstSlot) {
      // Slide 0: idx stays 0 through its own crossfade; progress spans the
      // whole slot (hold + crossfade) so it hits 1 exactly as slide 1 takes over.
      idx = 0;
      const held = tms - FIRST_HOLD_MS;
      fadeValue = held > 0 ? held / CROSSFADE_MS : 0;
      progressValue = tms / firstSlot;
    } else {
      const cycle = SLIDE_HOLD_MS + CROSSFADE_MS;
      const t = tms - firstSlot;
      const nCycles = Math.floor(t / cycle);
      const r = t - nCycles * cycle;
      idx = (1 + nCycles) % SLIDE_COUNT;
      fadeValue = r < SLIDE_HOLD_MS ? 0 : (r - SLIDE_HOLD_MS) / CROSSFADE_MS;
      progressValue = r / cycle; // resets to 0 exactly when idx changes
    }

    slideIndex.value = idx;
    slideProgress.value = progressValue;

    const imgs = imagesSV.value;
    if (imgs) {
      currentImg.value = imgs[idx];
      nextImg.value = imgs[(idx + 1) % imgs.length];
    }

    // Refresh the per-frame uniforms (fade + animated caustic clock).
    uniforms.value = {
      resolution: [width, height],
      gridSize: [gridW, gridH],
      refraction: REFRACTION_STRENGTH,
      blurAmount: BLUR_STRENGTH,
      lighting: LIGHTING_INTENSITY,
      shading: SHADING_STRENGTH,
      chromatic: CHROMATIC_DISPERSION,
      tintStrength: TINT_STRENGTH,
      tintColor: TINT_COLOR as unknown as number[],
      causticStrength: CAUSTIC_STRENGTH,
      causticScale: CAUSTIC_SCALE,
      causticSpeed: CAUSTIC_SPEED,
      causticColor: CAUSTIC_COLOR as unknown as number[],
      domainWarp: DOMAIN_WARP_STRENGTH,
      ambientDistortion: AMBIENT_DISTORTION_PX,
      fade: fadeValue,
      uTime: tms / 1000,
    };

    // --- 1. Velocity-gated impulse: inject only where the finger is MOVING,
    //        amount proportional to finger speed (matches the reference).
    if (touching.value === 1) {
      const cxGrid = (touchX.value / width) * gridW;
      const cyGrid = (touchY.value / height) * gridH;
      const pxGrid = (prevX.value / width) * gridW;
      const pyGrid = (prevY.value / height) * gridH;

      const dx = cxGrid - pxGrid;
      const dy = cyGrid - pyGrid;
      const segLen = Math.sqrt(dx * dx + dy * dy);

      if (segLen > VELOCITY_THRESHOLD) {
        // Impulse scales with speed, clamped so a fast flick can't explode.
        const speedFactor = Math.min(1, segLen * VELOCITY_GAIN);
        const impulse = IMPULSE_STRENGTH * speedFactor;

        const stamps = Math.max(1, Math.ceil(segLen));
        const r2 = BRUSH_RADIUS * BRUSH_RADIUS;
        const rCeil = Math.ceil(BRUSH_RADIUS);

        for (let s = 0; s < stamps; s++) {
          const t = stamps === 1 ? 0 : s / (stamps - 1);
          const sx = pxGrid + dx * t;
          const sy = pyGrid + dy * t;
          const minX = Math.max(1, Math.floor(sx) - rCeil);
          const maxX = Math.min(gridW - 2, Math.floor(sx) + rCeil);
          const minY = Math.max(1, Math.floor(sy) - rCeil);
          const maxY = Math.min(gridH - 2, Math.floor(sy) + rCeil);
          for (let y = minY; y <= maxY; y++) {
            for (let x = minX; x <= maxX; x++) {
              const ox = x - sx;
              const oy = y - sy;
              const d2 = ox * ox + oy * oy;
              if (d2 <= r2) {
                const falloff = 1 - d2 / r2;
                current[y * gridW + x] += impulse * falloff;
              }
            }
          }
        }
      }

      prevX.value = touchX.value;
      prevY.value = touchY.value;
    }

    // --- 2. Wave-equation step (Hugo Elias scheme). Borders stay 0 -> reflect.
    for (let y = 1; y < gridH - 1; y++) {
      const row = y * gridW;
      for (let x = 1; x < gridW - 1; x++) {
        const i = row + x;
        const next =
          ((current[i - 1] + current[i + 1] + current[i - gridW] + current[i + gridW]) * 0.5 -
            previous[i]) *
          DAMPING;
        previous[i] = next;
      }
    }
    // Swap: `previous` now holds the newest field.
    buffers.current = previous;
    buffers.previous = current;

    // --- 3. Encode the newest field into the RGBA texture.
    const field = buffers.current;
    for (let i = 0; i < cellCount; i++) {
      let v = 0.5 + field[i] * HEIGHT_SCALE;
      if (v < 0) v = 0;
      else if (v > 1) v = 1;
      rgba[i * 4] = (v * 255) | 0;
      rgba[i * 4 + 3] = 255;
    }

    // --- 4. Hand the height field to the shader as a fresh Skia image.
    const data = Skia.Data.fromBytes(rgba);
    const img = Skia.Image.MakeImage(
      { width: gridW, height: gridH, colorType: ColorType.RGBA_8888, alphaType: AlphaType.Opaque },
      data,
      gridW * 4,
    );
    if (img) {
      heightImage.value = img;
    }
  });

  const insets = useSafeAreaInsets();

  // Faint wash sweeping left->right across the pill as the current slide
  // progresses; the hard reset to 0 at each slide change comes from the worklet.
  const sweepStyle = useAnimatedStyle(() => ({
    width: `${slideProgress.value * 100}%`,
  }));

  // Per-tab opacity: the active slide's label is full white, the other two fade.
  const tab0Style = useAnimatedStyle(() => ({
    opacity: withTiming(slideIndex.value === 0 ? 1 : TAB_INACTIVE_OPACITY, {
      duration: TAB_FADE_MS,
    }),
  }));
  const tab1Style = useAnimatedStyle(() => ({
    opacity: withTiming(slideIndex.value === 1 ? 1 : TAB_INACTIVE_OPACITY, {
      duration: TAB_FADE_MS,
    }),
  }));
  const tab2Style = useAnimatedStyle(() => ({
    opacity: withTiming(slideIndex.value === 2 ? 1 : TAB_INACTIVE_OPACITY, {
      duration: TAB_FADE_MS,
    }),
  }));
  const tabStyles = [tab0Style, tab1Style, tab2Style];

  // Wait for all three images to decode and the shader to compile before
  // drawing so the shader never samples a null child.
  if (!ready || !source) {
    return <View style={styles.placeholder} />;
  }

  return (
    <View style={styles.root}>
      <GestureDetector gesture={pan}>
        <Canvas style={styles.canvas}>
          <Fill>
            <Shader source={source} uniforms={uniforms}>
              <ImageShader
                image={currentImg}
                fit="cover"
                rect={fullScreenRect}
                tx="clamp"
                ty="clamp"
              />
              <ImageShader
                image={nextImg}
                fit="cover"
                rect={fullScreenRect}
                tx="clamp"
                ty="clamp"
              />
              <ImageShader
                image={heightImage}
                fit="none"
                tx="clamp"
                ty="clamp"
                sampling={{ filter: FilterMode.Linear, mipmap: MipmapMode.None }}
              />
            </Shader>
          </Fill>
        </Canvas>
      </GestureDetector>

      {/* Overlays live in the RN view tree ABOVE the Canvas, so the water shader
          never distorts them. pointerEvents="none" lets the finger disturb the
          water everywhere, including under the text. */}
      <View style={styles.overlay} pointerEvents="none">
        {/* Overlay 1: centered hero headline. */}
        <View style={styles.heroWrap}>
          <Text style={styles.hero}>
            {HERO_LINE_1}
            {'\n'}
            {HERO_LINE_2}
          </Text>
        </View>

        {/* Overlay 2: "DISCOVER [Residences|Island|Villa] LIVING" bar. */}
        <View
          style={[styles.bottomBar, { bottom: insets.bottom + BAR_BOTTOM_OFFSET }]}
        >
          <Text style={styles.barLabel}>{SIDE_LABEL_LEFT}</Text>
          <View style={styles.pill}>
            <Animated.View style={[styles.pillSweep, sweepStyle]} />
            {TAB_LABELS.map((label, i) => (
              <Animated.View key={label} style={[styles.tab, tabStyles[i]]}>
                <Text style={styles.barLabel}>{label}</Text>
              </Animated.View>
            ))}
          </View>
          <Text style={styles.barLabel}>{SIDE_LABEL_RIGHT}</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
  canvas: {
    flex: 1,
  },
  placeholder: {
    flex: 1,
    backgroundColor: '#000',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  // Hero headline — vertically & horizontally centered on the screen.
  heroWrap: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  hero: {
    color: HERO_COLOR,
    fontSize: HERO_FONT_SIZE,
    lineHeight: HERO_LINE_HEIGHT,
    letterSpacing: HERO_LETTER_SPACING,
    // aviano-sans (licensed) is the real font on amaliproperties.com; the
    // platform default sans-serif at weight '300' stands in for it here.
    fontWeight: '300',
    textAlign: 'center',
  },
  // Bottom "DISCOVER [tabs] LIVING" bar — centered horizontally.
  bottomBar: {
    position: 'absolute',
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: BAR_GAP,
  },
  // Translucent rounded pill holding the three slide tabs.
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: PILL_PADDING_H,
    backgroundColor: PILL_BG,
    borderRadius: PILL_RADIUS,
    overflow: 'hidden',
  },
  // Faint wash sweeping left->right across the pill with slide progress.
  pillSweep: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    backgroundColor: PILL_SWEEP_BG,
  },
  tab: {
    paddingVertical: TAB_PADDING_V,
    paddingHorizontal: TAB_PADDING_H,
  },
  // Shared label style. aviano-sans (licensed) on the reference; platform
  // sans-serif stands in here.
  barLabel: {
    color: BAR_TEXT_COLOR,
    fontSize: BAR_FONT_SIZE,
    letterSpacing: BAR_LETTER_SPACING,
    fontWeight: '400',
  },
});

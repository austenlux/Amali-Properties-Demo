/* eslint-disable react-hooks/immutability, react-hooks/refs --
   This component is built on Reanimated shared values (written in worklets AND
   in JS gesture/press callbacks) and expo-audio players (settable .loop/.volume
   props), and reads a ref inside a gesture callback. All are runtime-safe,
   idiomatic patterns; the React Compiler's experimental immutability/refs lint
   rules false-flag them, but the compiler (reactCompiler: true) still runs and
   handles them correctly. Scoped to this file only. */
import {
  AlphaType,
  Blur,
  Canvas,
  ColorType,
  Fill,
  FilterMode,
  Group,
  ImageShader,
  LinearGradient,
  MipmapMode,
  Paint,
  Rect,
  Shader,
  Skia,
  useImage,
  vec,
  type SkImage,
} from '@shopify/react-native-skia';
import { useAudioPlayer } from 'expo-audio';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  Easing,
  runOnJS,
  useAnimatedStyle,
  useDerivedValue,
  useFrameCallback,
  useSharedValue,
  withDelay,
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
const CAUSTIC_SCALE = 8.0;

/** Drift speed of the field through time. Reference feeds `uTime * 0.25`. */
const CAUSTIC_SPEED = 0.25;

/** Domain-warp strength: how far the previous sample's gradient displaces the
 *  next sample point (in noise-space units). This folding is what turns smooth
 *  noise into the interconnected caustic web — more folding = more organic,
 *  less grid-like. Reference folds twice. */
const DOMAIN_WARP_STRENGTH = 0.55;

/** Peak screen-px the field gradient warps the image on its own — the gentle,
 *  spatially-varying "pool surface swaying" idle motion. Because it's the noise
 *  gradient, it differs everywhere and drifts with time (unlike a global sine).
 *  Added to the refraction offset only, so it never triggers choppy blur. */
const AMBIENT_DISTORTION_PX = 3.0;

/** Barely-visible bright rim traced along the edges of each ambient distortion
 *  shape — the thin caustic "veins" you see on a pool surface. Derived from the
 *  field gradient (largest at shape boundaries). */
const EDGE_GAIN = 1.6;      // maps gradient magnitude toward the rim
const EDGE_SHARP = 2.0;     // higher = thinner, crisper rim lines
const EDGE_STRENGTH = 0.07; // brightness of the rim

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
/** The translucent WHITE pill behind the slide labels (as on the reference).
 *  It reads because the bottom gradient scrim darkens the area behind it. */
const PILL_BG = 'rgba(255,255,255,0.15)';
const PILL_RADIUS = 10;
const PILL_PADDING_H = 0; // 0 so the progress sweep spans the pill's FULL width
                          // (an absolute child's 100% is the content box); tab
                          // padding still provides the inset around the labels.
/** Tall vertical padding gives the pill its height. */
const TAB_PADDING_V = 16;
const TAB_PADDING_H = 12;
/** The faint wash that sweeps across the pill as the slide progresses. */
const PILL_SWEEP_BG = 'rgba(255,255,255,0.10)';

/** Bottom gradient scrim (transparent at top -> dark at the very bottom) so the
 *  whole bar stays legible over any image, bright or busy. Rendered with Skia
 *  (no extra dependency). */
const SCRIM_HEIGHT = 260;
const SCRIM_COLOR = 'rgba(0,0,0,0.6)';

// --- Overlay 3: top logo wordmark -------------------------------------------
// The white transparent "AMÁLI / PROPERTIES" wordmark, centered near the top.

/** Logo width as a fraction of screen width. */
const LOGO_WIDTH_FRACTION = 0.46;
/** Wordmark aspect ratio (width : height) ≈ 2.94:1. Drives the fixed height. */
const LOGO_ASPECT = 2.94;
/** px below the safe-area top inset the logo (and equalizer) sit. */
const TOP_OFFSET = 14;

// --- Overlay 4: top-right equalizer play/pause button -----------------------
// A 40×40 circle framing three vertical bars that bounce like an equalizer
// while the ambient loop plays, and flatten when paused.

/** Diameter of the circular button. */
const EQ_BUTTON_SIZE = 40;
/** px from the right edge. */
const EQ_RIGHT_OFFSET = 20;
/** Bar geometry inside the button. */
const EQ_BAR_WIDTH = 2.5;
const EQ_BAR_HEIGHT = 12;
const EQ_BAR_GAP = 6;
/** scaleY range each bar bounces between while playing / rests at when paused. */
const EQ_MIN_SCALE = 0.25;
const EQ_MAX_SCALE = 1.0;
/** Cadence of the per-bar random bounce (ms) — mirrors the reference's 300ms. */
const EQ_STEP_MS = 300;
/** Per-bar period offset so the three bars drift out of phase (uncoordinated). */
const EQ_STAGGER_MS = 23;
/** How quickly the bars settle flat when paused (ms). */
const EQ_FLAT_MS = 200;

// --- Ambient audio ----------------------------------------------------------

/** One-shot UI click volume (0..1). The ambient loop stays at full volume. */
const CLICK_VOLUME = 0.25;

// --- Launch intro animation --------------------------------------------------
// On every app launch the whole water/background is Gaussian-blurred (Skia
// layer blur) with a centered two-line headline over it; after a brief hold the
// blur eases to zero and the intro text fades out AS the normal overlays fade
// in — landing exactly on the current experience. Driven by ONE shared value.

/** Intro headline, line 1 (hard break after this line). */
const INTRO_LINE_1 = 'REDEFINING';
/** Intro headline, line 2. */
const INTRO_LINE_2 = 'LUXURY LIVING';
/** How long the fully-blurred intro holds before the transition (ms). */
const INTRO_HOLD_MS = 600;
/** Ease-out duration for blur->0 + intro-text fade-out + overlays fade-in (ms). */
const INTRO_FADE_MS = 1800;
/** Peak Gaussian blur sigma (px) applied to the rendered water at t=0. */
const INTRO_MAX_BLUR = 30;
/** Handoff point (in `intro` units, 1→0) between the two headlines. The intro
 *  text fades out over intro 1→SPLIT, then the experience fades in over
 *  SPLIT→0 — so "REDEFINING / LUXURY LIVING" and "ONE DREAM / AT A TIME" never
 *  overlap (they'd otherwise cross-fade at the same screen position). */
const INTRO_SPLIT = 0.5;
/** Intro headline font size. Matches the hero's thin, centered treatment. */
const INTRO_FONT_SIZE = 44;
const INTRO_LINE_HEIGHT = 56;
const INTRO_LETTER_SPACING = 2;

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
uniform float edgeGain;      // maps field gradient magnitude toward the rim
uniform float edgeSharp;     // sharpness (thinness) of the caustic edge rim
uniform float edgeStrength;  // brightness of the caustic edge rim
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

// Two-octave fractal value noise: the second octave is rotated off-axis and
// higher frequency so the base cubic lattice never reads as square/diamond
// cells — the shapes come out organic and random.
float fnoise(float3 x) {
  float v = vnoise(x);
  float3 x2 = float3(
    x.x * 0.5403 - x.y * 0.8415,
    x.x * 0.8415 + x.y * 0.5403,
    x.z
  ) * 2.03 + 11.5;
  v += 0.5 * vnoise(x2);
  return v / 1.5; // normalize back to ~0..1
}

// Sample the fractal noise and its xy gradient (time held fixed) via forward
// differences. Returns float3(dValue/dx, dValue/dy, value).
float3 noiseGrad(float3 p) {
  const float e = 0.06; // finite-difference step in noise space
  float c = fnoise(p);
  float nx = fnoise(p + float3(e, 0.0, 0.0));
  float ny = fnoise(p + float3(0.0, e, 0.0));
  return float3((nx - c) / e, (ny - c) / e, c);
}

// Animated caustic field with two domain-warp folds (matches the reference's
// base -> balance -> final re-sampling). Each fold offsets the sample point by
// the previous sample's gradient, folding smooth noise into the caustic web.
// Returns float3(gradient.x, gradient.y, value). 9 noise taps total.
float3 causticField(float2 aspectUv) {
  // Rotate the sample domain off-axis (~0.5 rad) so the value-noise lattice
  // doesn't read as regular axis-aligned diamonds.
  float2 r = float2(
    aspectUv.x * 0.87758 - aspectUv.y * 0.47943,
    aspectUv.x * 0.47943 + aspectUv.y * 0.87758
  );
  float3 p = float3(r * causticScale, uTime * causticSpeed);
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

  // Barely-visible bright rim on the edges of each ambient distortion shape.
  // The field gradient is largest at the shape boundaries, so |field.xy| traces
  // the edges; sharpen it into thin veins and add a whisper of white.
  float ambEdge = pow(clamp(length(field.xy) * edgeGain, 0.0, 1.0), edgeSharp);
  col.rgb += half3(ambEdge * edgeStrength);

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

  // Overridable rotation state (worklet-owned). Instead of deriving the slide
  // from absolute elapsed time, the worklet tracks the active slide and the
  // clock at which its hold began, so a tab tap can reset the cycle. All three
  // are written ONLY inside the frame worklet.
  const activeIndex = useSharedValue(0); // slide the worklet is currently holding
  const cycleStartMs = useSharedValue(-1); // tms the current hold began (-1 = seed)
  const currentHoldMs = useSharedValue(FIRST_HOLD_MS); // hold for the active slide

  // JS -> worklet jump bridge for tab taps. JS sets the requested index; the
  // worklet consumes it on the next frame and clears it (-1 = nothing pending).
  // Written from the JS press handler AND cleared in the worklet — never in an
  // effect, so there's no React-Compiler write conflict and no JS/UI race.
  const pendingJump = useSharedValue(-1);

  // Readiness is derived during render (no state, no cascading renders); the
  // effect only publishes the loaded image list to the worklet.
  const ready = !!(img0 && img1 && img2);

  useEffect(() => {
    if (img0 && img1 && img2) {
      imagesSV.value = [img0, img1, img2];
    }
  }, [img0, img1, img2, imagesSV]);

  // --- Ambient audio (expo-audio) ------------------------------------------
  // Two auto-managed players: a looping beach ambience and a one-shot UI click.
  const beach = useAudioPlayer(require('@/assets/audio/beach-loop.mp3'));
  const click = useAudioPlayer(require('@/assets/audio/click.mp3'));

  // Configure the players once. `loop`/`volume` are plain settable properties on
  // the AudioPlayer instance; safe to set before the source finishes loading.
  useEffect(() => {
    beach.loop = true;
    click.volume = CLICK_VOLUME;
  }, [beach, click]);

  // isPlaying changes only on explicit user action (rarely) — plain React state
  // is fine here; it is NEVER written per frame.
  const [isPlaying, setIsPlaying] = useState(false);
  // The loop is gesture-gated: it starts on the FIRST user interaction only.
  const hasStartedRef = useRef(false);

  const startAudioIfNeeded = useCallback(() => {
    if (hasStartedRef.current) return;
    hasStartedRef.current = true;
    beach.play();
    setIsPlaying(true);
  }, [beach]);

  // One-shot click: rewind to the start, then play (so rapid taps re-trigger).
  const playClick = useCallback(() => {
    click.seekTo(0);
    click.play();
  }, [click]);

  // Equalizer button toggles the loop; the very first tap starts it.
  const toggleAudio = useCallback(() => {
    if (!hasStartedRef.current) {
      startAudioIfNeeded();
      return;
    }
    if (isPlaying) {
      beach.pause();
      setIsPlaying(false);
    } else {
      beach.play();
      setIsPlaying(true);
    }
  }, [beach, isPlaying, startAudioIfNeeded]);

  // Tab tap: request a jump to slide `i`, play the click, and (first time) start
  // the ambience. The worklet applies the jump on its next frame.
  const onTabPress = useCallback(
    (i: number) => {
      pendingJump.value = i;
      playClick();
      startAudioIfNeeded();
    },
    [pendingJump, playClick, startAudioIfNeeded],
  );

  // --- Equalizer bars: three UI-thread scaleY values -----------------------
  // Written from a JS timer (rare, ~300ms) or the play/pause effect — never
  // from the frame worklet — and read only in useAnimatedStyle.
  const bar0 = useSharedValue(EQ_MIN_SCALE);
  const bar1 = useSharedValue(EQ_MIN_SCALE);
  const bar2 = useSharedValue(EQ_MIN_SCALE);

  useEffect(() => {
    const bars = [bar0, bar1, bar2];
    if (!isPlaying) {
      // Paused: flatten all three to a short, equal, static height.
      bars.forEach((b) => {
        b.value = withTiming(EQ_MIN_SCALE, { duration: EQ_FLAT_MS });
      });
      return;
    }
    // Playing: each bar animates to a fresh random scaleY on its own cadence.
    // Slightly different periods (via EQ_STAGGER_MS) drift the bars out of phase
    // so the bounce reads jittery and uncoordinated, like the reference.
    const bounce = (b: typeof bar0) => {
      const target = EQ_MIN_SCALE + Math.random() * (EQ_MAX_SCALE - EQ_MIN_SCALE);
      b.value = withTiming(target, { duration: EQ_STEP_MS });
    };
    const timers = bars.map((b, i) => {
      bounce(b); // kick immediately so it reacts on tap
      return setInterval(() => bounce(b), EQ_STEP_MS + i * EQ_STAGGER_MS);
    });
    return () => timers.forEach(clearInterval);
  }, [isPlaying, bar0, bar1, bar2]);

  const bar0Style = useAnimatedStyle(() => ({ transform: [{ scaleY: bar0.value }] }));
  const bar1Style = useAnimatedStyle(() => ({ transform: [{ scaleY: bar1.value }] }));
  const bar2Style = useAnimatedStyle(() => ({ transform: [{ scaleY: bar2.value }] }));
  const barStyles = [bar0Style, bar1Style, bar2Style];

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
    edgeGain: EDGE_GAIN,
    edgeSharp: EDGE_SHARP,
    edgeStrength: EDGE_STRENGTH,
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
      // Gesture-gated audio: start the loop on the first water touch. The JS
      // guard makes every call after the first a no-op.
      runOnJS(startAudioIfNeeded)();
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

    // --- 0. Clock + OVERRIDABLE background rotation.
    // Rather than deriving the slide from absolute elapsed time, the worklet
    // holds `activeIndex` and `cycleStartMs` (the clock at which the active
    // slide's hold began) and advances them itself. This makes the rotation
    // resettable: a tab tap can drop a new index into `cycleStartMs`/activeIndex
    // via `pendingJump` and the hold restarts cleanly. idx/fadeValue drive the
    // crossfade; slideIndex/slideProgress feed the overlays — all from this one
    // clock so they stay in lockstep. The first slide holds FIRST_HOLD_MS, every
    // slide after (including a tapped one) holds SLIDE_HOLD_MS.
    const tms = frameInfo.timeSinceFirstFrame;

    // Seed the cycle clock on the very first frame.
    if (cycleStartMs.value < 0) {
      cycleStartMs.value = tms;
    }

    // Consume a pending tab jump (set from the JS thread): jump to the tapped
    // slide, restart its hold now, and clear the request.
    const jump = pendingJump.value;
    if (jump >= 0) {
      activeIndex.value = jump;
      cycleStartMs.value = tms;
      currentHoldMs.value = SLIDE_HOLD_MS;
      pendingJump.value = -1;
    }

    // Advance through any completed slots (usually zero or one per frame). Each
    // slot is hold + crossfade; after the first slide every hold is SLIDE_HOLD_MS.
    // Stepping cycleStartMs by whole slots (vs. resetting to tms) avoids drift.
    let hold = currentHoldMs.value;
    let slot = hold + CROSSFADE_MS;
    while (tms - cycleStartMs.value >= slot) {
      cycleStartMs.value += slot;
      activeIndex.value = (activeIndex.value + 1) % SLIDE_COUNT;
      currentHoldMs.value = SLIDE_HOLD_MS;
      hold = SLIDE_HOLD_MS;
      slot = hold + CROSSFADE_MS;
    }

    const idx = activeIndex.value;
    const elapsedInCycle = tms - cycleStartMs.value;
    // Progress fills across the hold and hits 1 exactly as the crossfade starts.
    const progressValue = Math.min(1, elapsedInCycle / hold);
    // Fade runs only during the trailing CROSSFADE_MS of the slot.
    const held = elapsedInCycle - hold;
    const fadeValue = held > 0 ? Math.min(1, held / CROSSFADE_MS) : 0;

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
      edgeGain: EDGE_GAIN,
      edgeSharp: EDGE_SHARP,
      edgeStrength: EDGE_STRENGTH,
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

  // --- Launch intro -------------------------------------------------------
  // ONE animation source: `intro` starts at 1 (fully blurred, headline shown)
  // and eases to 0 on mount after a short hold. Everything else derives from it:
  //   • blur sigma      = intro * INTRO_MAX_BLUR   (Skia layer Gaussian blur)
  //   • intro-text opacity = intro                 (fades out)
  //   • normal-overlay opacity = 1 - intro         (fades in)
  // No per-frame setState — the effect kicks a single withTiming; the UI thread
  // drives the rest through derived/animated styles.
  const intro = useSharedValue(1);
  useEffect(() => {
    intro.value = withDelay(
      INTRO_HOLD_MS,
      withTiming(0, { duration: INTRO_FADE_MS, easing: Easing.out(Easing.cubic) }),
    );
  }, [intro]);

  // Animated Gaussian blur sigma for the Skia layer wrapping the water.
  const introBlur = useDerivedValue(() => intro.value * INTRO_MAX_BLUR);
  // Intro headline fades out; normal overlays fade in — in lockstep.
  // Staggered handoff so the two headlines never overlap: intro text fades out
  // over intro 1→SPLIT, the experience fades in over SPLIT→0.
  const introTextStyle = useAnimatedStyle(() => ({
    opacity: Math.max(0, Math.min(1, (intro.value - INTRO_SPLIT) / (1 - INTRO_SPLIT))),
  }));
  const normalOverlayStyle = useAnimatedStyle(() => ({
    opacity: Math.max(0, Math.min(1, (INTRO_SPLIT - intro.value) / INTRO_SPLIT)),
  }));

  // Faint wash sweeping left->right across the pill as the current slide
  // progresses; the hard reset to 0 at each slide change comes from the worklet.
  const sweepStyle = useAnimatedStyle(() => ({
    transform: [{ scaleX: slideProgress.value }],
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
          {/* The whole water is rendered into an offscreen layer whose Paint
              carries an animated Gaussian <Blur> image filter, so the RENDERED
              water (sim + rotation, untouched underneath) is blurred as one — the
              launch-intro blur. `introBlur` eases 30->0, clearing the water. */}
          <Group layer={<Paint><Blur blur={introBlur} /></Paint>}>
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
          </Group>
        </Canvas>
      </GestureDetector>

      {/* Overlays live in the RN view tree ABOVE the Canvas, so the water shader
          never distorts them. The wrapper is "box-none": it passes touches
          through to the water Canvas everywhere EXCEPT on the few interactive
          controls (the tabs + the equalizer button), which catch their own
          touches. Every non-interactive child is pointerEvents="none" so drags
          over them still ripple the water. Its opacity fades 0->1 as the intro
          clears; "box-none" is preserved so, once faded in, the water stays
          touchable and only the controls catch — exactly as before the intro. */}
      <Animated.View
        style={[styles.overlay, normalOverlayStyle]}
        pointerEvents="box-none"
      >
        {/* Bottom gradient scrim (Skia) so the bar reads over any image. */}
        <Canvas style={styles.scrim} pointerEvents="none">
          <Rect x={0} y={0} width={width} height={SCRIM_HEIGHT}>
            <LinearGradient
              start={vec(0, 0)}
              end={vec(0, SCRIM_HEIGHT)}
              colors={['transparent', SCRIM_COLOR]}
            />
          </Rect>
        </Canvas>

        {/* Overlay 3: top logo wordmark — centered, touch-transparent. */}
        <View
          style={[styles.logoWrap, { top: insets.top + TOP_OFFSET }]}
          pointerEvents="none"
        >
          <Image
            source={require('@/assets/images/amali-logo.png')}
            resizeMode="contain"
            style={{
              width: width * LOGO_WIDTH_FRACTION,
              height: (width * LOGO_WIDTH_FRACTION) / LOGO_ASPECT,
            }}
          />
        </View>

        {/* Overlay 4: top-right equalizer play/pause button — catches touches. */}
        <Pressable
          onPress={toggleAudio}
          style={[
            styles.eqButton,
            { top: insets.top + TOP_OFFSET, right: EQ_RIGHT_OFFSET },
          ]}
          accessibilityRole="button"
          accessibilityLabel={isPlaying ? 'Pause ambient sound' : 'Play ambient sound'}
        >
          {barStyles.map((barStyle, i) => (
            <Animated.View key={i} style={[styles.eqBar, barStyle]} />
          ))}
        </Pressable>

        {/* Overlay 1: centered hero headline. */}
        <View style={styles.heroWrap} pointerEvents="none">
          <Text style={styles.hero}>
            {HERO_LINE_1}
            {'\n'}
            {HERO_LINE_2}
          </Text>
        </View>

        {/* Overlay 2: "DISCOVER [Residences|Island|Villa] LIVING" bar. The bar
            and pill are "box-none" so the labels/gaps pass touches through; only
            the tab pressables catch. */}
        <View
          style={[styles.bottomBar, { bottom: insets.bottom + BAR_BOTTOM_OFFSET }]}
          pointerEvents="box-none"
        >
          <Text style={styles.barLabel} pointerEvents="none">
            {SIDE_LABEL_LEFT}
          </Text>
          <View style={styles.pill} pointerEvents="box-none">
            <Animated.View style={[styles.pillSweep, sweepStyle]} pointerEvents="none" />
            {TAB_LABELS.map((label, i) => (
              <Pressable key={label} onPress={() => onTabPress(i)} style={styles.tab}>
                <Animated.View style={tabStyles[i]}>
                  <Text style={styles.barLabel}>{label}</Text>
                </Animated.View>
              </Pressable>
            ))}
          </View>
          <Text style={styles.barLabel} pointerEvents="none">
            {SIDE_LABEL_RIGHT}
          </Text>
        </View>
      </Animated.View>

      {/* Launch-intro headline: centered like the hero, white/thin with the same
          dark halo, over the blurred water. Opacity fades 1->0 as the intro
          clears. pointerEvents="none" so it never intercepts touches. */}
      <Animated.View
        style={[styles.introWrap, introTextStyle]}
        pointerEvents="none"
      >
        <Text style={styles.introText}>
          {INTRO_LINE_1}
          {'\n'}
          {INTRO_LINE_2}
        </Text>
      </Animated.View>
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
  // Bottom gradient scrim behind the bar.
  scrim: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: SCRIM_HEIGHT,
  },
  // Top logo wordmark — horizontally centered, pinned near the top.
  logoWrap: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  // Circular equalizer play/pause button, top-right.
  eqButton: {
    position: 'absolute',
    width: EQ_BUTTON_SIZE,
    height: EQ_BUTTON_SIZE,
    borderRadius: EQ_BUTTON_SIZE / 2,
    borderWidth: 1,
    borderColor: '#FFFFFF',
    backgroundColor: 'transparent',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: EQ_BAR_GAP,
  },
  // A single equalizer bar; scaleY is animated and anchored at the bottom.
  eqBar: {
    width: EQ_BAR_WIDTH,
    height: EQ_BAR_HEIGHT,
    borderRadius: 1,
    backgroundColor: '#FFFFFF',
    transformOrigin: 'bottom',
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
    // Dark halo so the headline stays readable over bright image areas (same
    // approach as the bottom bar labels).
    textShadowColor: 'rgba(0,0,0,0.55)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 10,
  },
  // Launch-intro headline — centered on screen over the blurred water.
  introWrap: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  introText: {
    color: '#FFFFFF',
    fontSize: INTRO_FONT_SIZE,
    lineHeight: INTRO_LINE_HEIGHT,
    letterSpacing: INTRO_LETTER_SPACING,
    fontWeight: '300',
    textAlign: 'center',
    // Same dark halo the hero uses, for legibility over bright/blurred imagery.
    textShadowColor: 'rgba(0,0,0,0.55)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 10,
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
    right: 0,
    backgroundColor: PILL_SWEEP_BG,
    // Fill from the left edge; scaleX (driven by slideProgress) reveals it to
    // the true right edge regardless of any padding.
    transformOrigin: 'left',
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
    // Dark halo so the (backgroundless) DISCOVER/LIVING and the tab labels stay
    // legible over bright image areas.
    textShadowColor: 'rgba(0,0,0,0.75)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 6,
  },
});

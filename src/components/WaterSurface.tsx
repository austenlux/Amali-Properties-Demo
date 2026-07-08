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
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { useFrameCallback, useSharedValue } from 'react-native-reanimated';

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

// --- Ambient caustics shimmer (always on, even with no touch) --------------

/** Caustic light colour (reference caustic tint ≈ 0.42, 0.82, 1.0). */
const CAUSTIC_COLOR = [0.4196, 0.8235, 1.0] as const;

/** Caustic brightness. SUBTLE — this is an ambient breeze, not a light show. */
const CAUSTIC_STRENGTH = 0.12;

/** Spatial frequency of the caustic noise (cells across the screen width). */
const CAUSTIC_SCALE = 3.5;

/** Drift speed of the caustic field. Reference feeds `uTime * 0.25`. */
const CAUSTIC_SPEED = 0.25;

// --- Ambient surface motion (always on, even with no touch) ----------------

/** Peak screen-px the image gently warps on its own — the "gentle breeze on
 *  the water" idle motion. A few drifting sine waves, independent of ripples,
 *  so the surface is always subtly alive. Keep it small and calm. */
const AMBIENT_STRENGTH = 4.0;

/** Drift speed of the ambient motion. Higher = busier; keep it slow/gentle. */
const AMBIENT_SPEED = 0.5;

// --- Background image rotation ---------------------------------------------

/** How long the very first image holds before the first crossfade (ms). */
const FIRST_HOLD_MS = 18000;

/** How long each subsequent image holds before crossfading (ms). */
const SLIDE_HOLD_MS = 5000;

/** Crossfade duration between images (ms). */
const CROSSFADE_MS = 300;

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
uniform float ambientStrength; // px of always-on gentle surface warp
uniform float ambientSpeed;    // drift speed of the ambient motion
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

// --- Cheap 3D value noise (2 octaves) for the caustic shimmer -------------
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

float fbm(float3 p) {
  float v = 0.5 * vnoise(p);
  v += 0.25 * vnoise(p * 2.0);
  return v / 0.75; // normalize back to ~0..1
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

  // Ambient gentle motion — always on, even with no touch. A few incommensurate
  // sine waves drifting over time nudge the sample point like a soft breeze on
  // the surface. Added to the refraction offset only (NOT into grad), so it
  // warps the image without triggering the choppy-water blur/darkening.
  float at = uTime * ambientSpeed;
  float2 amb = float2(
    sin(uv.y * 7.0 + at) + 0.6 * sin(uv.x * 5.0 - at * 0.8),
    cos(uv.x * 6.0 + at * 0.9) + 0.6 * cos(uv.y * 4.5 - at * 0.7)
  ) * ambientStrength;

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

  // Ambient caustics: slow-drifting 3D noise, screen-blended in caustic color.
  float2 aspectUv = float2(uv.x * (resolution.x / resolution.y), uv.y);
  float3 np = float3(aspectUv * causticScale, uTime * causticSpeed);
  float n = fbm(np);
  n = pow(clamp(n, 0.0, 1.0), 2.0);
  half3 caustic = half3(causticColor) * half(n * causticStrength);
  col.rgb = half3(1.0) - (half3(1.0) - col.rgb) * (half3(1.0) - caustic);

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
    ambientStrength: AMBIENT_STRENGTH,
    ambientSpeed: AMBIENT_SPEED,
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
    const tms = frameInfo.timeSinceFirstFrame;
    let fadeValue = 0;

    const imgs = imagesSV.value;
    if (imgs) {
      const count = imgs.length;
      let idx = 0;
      if (tms < FIRST_HOLD_MS) {
        idx = 0;
        fadeValue = 0;
      } else {
        let t = tms - FIRST_HOLD_MS;
        if (t < CROSSFADE_MS) {
          idx = 0;
          fadeValue = t / CROSSFADE_MS;
        } else {
          t -= CROSSFADE_MS;
          const cycle = SLIDE_HOLD_MS + CROSSFADE_MS;
          const nCycles = Math.floor(t / cycle);
          const r = t - nCycles * cycle;
          idx = (1 + nCycles) % count;
          fadeValue = r < SLIDE_HOLD_MS ? 0 : (r - SLIDE_HOLD_MS) / CROSSFADE_MS;
        }
      }
      currentImg.value = imgs[idx];
      nextImg.value = imgs[(idx + 1) % count];
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
      ambientStrength: AMBIENT_STRENGTH,
      ambientSpeed: AMBIENT_SPEED,
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

  // Wait for all three images to decode and the shader to compile before
  // drawing so the shader never samples a null child.
  if (!ready || !source) {
    return <View style={styles.placeholder} />;
  }

  return (
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
  );
}

const styles = StyleSheet.create({
  canvas: {
    flex: 1,
  },
  placeholder: {
    flex: 1,
    backgroundColor: '#000',
  },
});
